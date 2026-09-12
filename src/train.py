"""Train a masked discrete diffusion Transformer on synthetic Sudoku.

Reproducibility contract for the K-sweep comparison: the data order and the
masking process are driven by RNG streams that do NOT depend on K, so every
arm sees exactly the same puzzles in the same order with the same masks. Only
model initialisation consumes the torch RNG (whose draw count necessarily
differs with K, since K changes the parameter count).

Usage:
    python src/train.py --k 0 --name k0_s0 --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from diffusion import (
    denoising_loss,
    load_split,
    make_inputs,
    read_meta,
    sample_masks,
    solve,
    solve_metrics,
)
from model import MASK, N_CELLS, ModelConfig, RegisterDiffusionTransformer

DATA_RNG_OFFSET = 1000
MASK_RNG_OFFSET = 2000
VAL_EVAL_SEED = 987654321  # arm- and seed-independent: identical val batches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, help="number of register tokens")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=str, default="/data/ryan.kim/registers_data")
    p.add_argument("--out", type=str, default="/data/ryan.kim/registers_runs")
    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    # optim
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    # eval
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--val_loss_n", type=int, default=8192)
    p.add_argument("--val_solve_n", type=int, default=256,
                   help="puzzles decoded at each mid-training eval")
    p.add_argument("--final_solve_n", type=int, default=20000,
                   help="puzzles decoded for the final val/test numbers")
    p.add_argument("--reveal_per_step", type=int, default=1)
    p.add_argument("--solve_chunk", type=int, default=1024)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    prog = min(1.0, prog)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)


@torch.no_grad()
def val_denoising_loss(model, val, fixed, device, reg_mode="normal", chunk=2048):
    model.eval()
    tot_ce, tot_elbo, nb = 0.0, 0.0, 0
    for i in range(0, len(fixed["idx"]), chunk):
        sl = slice(i, i + chunk)
        idx = fixed["idx"][sl]
        tokens, sol, m = make_inputs(
            val["puzzles"][idx], val["solutions"][idx], fixed["mask"][sl], device
        )
        t = torch.as_tensor(fixed["t"][sl], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens, reg_mode=reg_mode)
        ce, elbo = denoising_loss(logits.float(), sol, m, t)
        tot_ce += ce.item()
        tot_elbo += elbo.item()
        nb += 1
    return tot_ce / nb, tot_elbo / nb


@torch.no_grad()
def register_stats(model, val, device, n=256):
    """Mean L2 norm of register hidden states per layer, plus the same for
    real tokens as a scale reference. Cheap; logged at every eval."""
    if model.K == 0:
        return None
    n = min(n, len(val["puzzles"]))
    p = val["puzzles"][:n]
    # measure at the fully-masked state (the first denoising step)
    s = np.full((n, N_CELLS), MASK, dtype=np.int64)
    from model import build_tokens  # local import: token assembly only
    tok = build_tokens(
        torch.as_tensor(p, device=device, dtype=torch.long),
        torch.as_tensor(s, device=device, dtype=torch.long),
    )
    hs = model.hidden_states(tok)
    out = {"reg_norm": [], "tok_norm": []}
    for h in hs:
        out["reg_norm"].append(h[:, -model.K:, :].float().norm(dim=-1).mean().item())
        out["tok_norm"].append(h[:, :-model.K, :].float().norm(dim=-1).mean().item())
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed + DATA_RNG_OFFSET)
    mask_rng = np.random.default_rng(args.seed + MASK_RNG_OFFSET)

    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    test = load_split(args.data, "test")
    n_train = len(train["puzzles"])

    # Fixed validation batch for the denoising-loss curve: identical across
    # arms, seeds and steps so the curves are directly comparable.
    veval_rng = np.random.default_rng(VAL_EVAL_SEED)
    vn = min(args.val_loss_n, len(val["puzzles"]))
    v_idx = np.sort(veval_rng.choice(len(val["puzzles"]), size=vn, replace=False))
    v_t, v_mask = sample_masks(vn, veval_rng)
    fixed = {"idx": v_idx, "t": v_t, "mask": v_mask}

    cfg = ModelConfig(
        n_registers=args.k, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
    )
    model = RegisterDiffusionTransformer(cfg).to(device)
    cfg.save(os.path.join(run_dir, "model_config.json"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": model.n_params(),
                   "data_meta": read_meta(args.data)}, f, indent=2)

    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        (no_decay if p_.ndim < 2 else decay).append(p_)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    log_path = os.path.join(run_dir, "log.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        rec["wall_s"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.time()
    print(f"[{args.name}] K={args.k} params={model.n_params():,} "
          f"train={n_train:,} device={device}", flush=True)

    run_loss, run_n = 0.0, 0
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        idx = np.sort(data_rng.integers(0, n_train, size=args.bs))
        t_np, mask_np = sample_masks(args.bs, mask_rng)
        tokens, sol, m = make_inputs(
            train["puzzles"][idx], train["solutions"][idx], mask_np, device
        )
        t = torch.as_tensor(t_np, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens)
        ce, elbo = denoising_loss(logits.float(), sol, m, t)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run_loss += ce.item()
        run_n += 1

        if (step + 1) % 100 == 0:
            log({"step": step + 1, "split": "train", "ce": run_loss / run_n,
                 "elbo": elbo.item(), "lr": lr_at(step, args),
                 "grad_norm": float(gn)})
            run_loss, run_n = 0.0, 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vce, velbo = val_denoising_loss(model, val, fixed, device)
            dec = solve(model, val["puzzles"][:args.val_solve_n], device,
                        reveal_per_step=args.reveal_per_step,
                        chunk=args.solve_chunk)
            sm = solve_metrics(dec, val["solutions"][:args.val_solve_n])
            rec = {"step": step + 1, "split": "val", "ce": vce, "elbo": velbo,
                   "exact_solve_acc": sm["exact_solve_acc"],
                   "cell_acc": sm["cell_acc"], "solve_n": sm["n"]}
            rs = register_stats(model, val, device)
            if rs:
                rec["reg_norm_last"] = rs["reg_norm"][-1]
                rec["tok_norm_last"] = rs["tok_norm"][-1]
            log(rec)

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "step": step + 1, "args": vars(args)},
                       os.path.join(run_dir, "ckpt.pt"))

    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                "step": args.steps, "args": vars(args)},
               os.path.join(run_dir, "ckpt.pt"))

    final = {}
    for split_name, split in (("val", val), ("test", test)):
        n = min(args.final_solve_n, len(split["puzzles"]))
        dec = solve(model, split["puzzles"][:n], device,
                    reveal_per_step=args.reveal_per_step, chunk=args.solve_chunk)
        final[split_name] = solve_metrics(
            dec, split["solutions"][:n], split["n_clues"][:n]
        )
    vce, velbo = val_denoising_loss(model, val, fixed, device)
    final["val_ce"] = vce
    final["val_elbo"] = velbo
    final["k"] = args.k
    final["seed"] = args.seed
    final["n_params"] = model.n_params()
    final["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL " + json.dumps({k: v for k, v in final.items() if k != "val"}),
          flush=True)
    log({"step": args.steps, "split": "final",
         "val_exact": final["val"]["exact_solve_acc"],
         "test_exact": final["test"]["exact_solve_acc"], "val_ce": vce})
    logf.close()


if __name__ == "__main__":
    main()
