"""Train the masked discrete diffusion Transformer with loss-free registers
on text (wikitext-103, GPT-2 BPE).

Same register mechanism as the Sudoku task -- K learned embeddings appended
past the end of the sequence, re-read every forward pass, no prediction target
and no direct loss. Only the data and the sequence shape change.

The metric is validation denoising cross-entropy on a fixed batch. Unlike
Sudoku exact-solve accuracy it has no ceiling, so a real effect has room to
appear and a null is harder to explain away.

    python src/text_train.py --k 0 --name text_k0_s0 --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, RegisterDiffusionTransformer

DATA_RNG_OFFSET = 1000
MASK_RNG_OFFSET = 2000
VAL_EVAL_SEED = 987654321  # arm- and seed-independent, as in the Sudoku runs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=str, default="/data/ryan.kim/registers_text_data")
    p.add_argument("--out", type=str, default="/data/ryan.kim/registers_runs")
    # model
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_tie", action="store_true")
    # optim
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    # eval
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--val_batches", type=int, default=32)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)


def sample_windows(stream, n, seq_len, rng):
    """n random windows of seq_len tokens from a flat token stream."""
    starts = rng.integers(0, len(stream) - seq_len - 1, size=n)
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    return stream[idx].astype(np.int64)


def make_batch(stream, n, seq_len, mask_id, data_rng, mask_rng):
    """Absorbing-state corruption: t ~ U(0,1], mask each position with prob t,
    always at least one masked position."""
    x = sample_windows(stream, n, seq_len, data_rng)
    t = np.clip(mask_rng.random(n).astype(np.float32), 1.0 / seq_len, 1.0)
    m = mask_rng.random((n, seq_len)).astype(np.float32) < t[:, None]
    short = m.sum(1) < 1
    if short.any():
        for i in np.where(short)[0]:
            m[i, mask_rng.integers(0, seq_len)] = True
    return x, m, t


def loss_on_batch(model, x, m, t, device, head_chunk=8192):
    """Cross-entropy at masked positions only.

    The head is applied to the GATHERED masked positions rather than to every
    position: at 50k classes that is the difference between a few hundred MB
    and several GB of logits.
    """
    xb = torch.as_tensor(x, device=device)
    mb = torch.as_tensor(m, device=device)
    inp = torch.where(mb, torch.full_like(xb, mask_id_of(model)), xb)
    h = model(inp, return_hidden=True)          # (B, L, D)
    sel = mb.reshape(-1)
    hsel = h.reshape(-1, h.shape[-1])[sel]      # (n_masked, D)
    tgt = xb.reshape(-1)[sel]
    total, n = 0.0, hsel.shape[0]
    ce_sum = None
    for i in range(0, n, head_chunk):
        lg = model.head(hsel[i:i + head_chunk].float())
        part = F.cross_entropy(lg, tgt[i:i + head_chunk], reduction="sum")
        ce_sum = part if ce_sum is None else ce_sum + part
    mean_ce = ce_sum / max(n, 1)
    # MDLM ELBO weighting, reported for reference only
    with torch.no_grad():
        per_ex = mb.float().sum(1) / mb.shape[1]
        elbo = (mean_ce.detach() * per_ex / torch.as_tensor(t, device=device)).mean()
    return mean_ce, elbo, n


def mask_id_of(model):
    """The MASK id is the extra vocabulary slot appended past the tokenizer."""
    return model.cfg.vocab_size - 1


@torch.no_grad()
def val_loss(model, val_stream, args, device):
    model.eval()
    rng = np.random.default_rng(VAL_EVAL_SEED)
    tot, ntok = 0.0, 0
    for _ in range(args.val_batches):
        x, m, t = make_batch(val_stream, args.bs, args.seq_len, None, rng, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ce, _, n = loss_on_batch(model, x, m, t, device)
        tot += ce.item() * n
        ntok += n
    model.train()
    return tot / max(ntok, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed + DATA_RNG_OFFSET)
    mask_rng = np.random.default_rng(args.seed + MASK_RNG_OFFSET)

    with open(os.path.join(args.data, "meta.json")) as f:
        dmeta = json.load(f)
    train_stream = np.load(os.path.join(args.data, "train.npy"), mmap_mode="r")
    val_stream = np.load(os.path.join(args.data, "val.npy"), mmap_mode="r")
    # one extra id past the tokenizer vocabulary is the MASK token
    vocab = dmeta["vocab_size"] + 1

    cfg = ModelConfig(
        n_registers=args.k, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
        vocab_size=vocab, seq_len=args.seq_len, out_start=0,
        out_len=args.seq_len, n_classes=vocab, tie_head=not args.no_tie,
    )
    model = RegisterDiffusionTransformer(cfg).to(device)
    cfg.save(os.path.join(run_dir, "model_config.json"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": model.n_params(),
                   "data_meta": dmeta}, f, indent=2)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    t0 = time.time()

    def log(rec):
        rec["wall_s"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    print(f"[{args.name}] K={args.k} params={model.n_params():,} "
          f"vocab={vocab} seq={args.seq_len} "
          f"train_tokens={len(train_stream):,} device={device}", flush=True)

    run_loss, run_n = 0.0, 0
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, m, t = make_batch(train_stream, args.bs, args.seq_len, None,
                             data_rng, mask_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ce, elbo, _ = loss_on_batch(model, x, m, t, device)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run_loss += ce.item()
        run_n += 1

        if (step + 1) % 100 == 0:
            log({"step": step + 1, "split": "train", "ce": run_loss / run_n,
                 "elbo": float(elbo), "lr": lr_at(step, args),
                 "grad_norm": float(gn)})
            run_loss, run_n = 0.0, 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vce = val_loss(model, val_stream, args, device)
            log({"step": step + 1, "split": "val", "ce": vce,
                 "ppl": math.exp(min(vce, 20))})

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "step": step + 1, "args": vars(args)},
                       os.path.join(run_dir, "ckpt.pt"))

    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                "step": args.steps, "args": vars(args)},
               os.path.join(run_dir, "ckpt.pt"))
    vce = val_loss(model, val_stream, args, device)
    final = {"k": args.k, "seed": args.seed, "val_ce": vce,
             "val_ppl": math.exp(min(vce, 20)), "n_params": model.n_params(),
             "steps": args.steps, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL " + json.dumps(final), flush=True)
    logf.close()


if __name__ == "__main__":
    main()
