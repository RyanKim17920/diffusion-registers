"""Train the block-diffusion Sudoku model with carried register state.

Same comparison hygiene as src/train.py: the data order and the block plan
(which block, in what reveal order) come from RNG streams seeded independently
of K, so every arm sees identical examples with identical block plans. The
validation loss uses a fixed, arm-independent plan.

Cost note: one training step unrolls a whole generation block, so it runs
`block_size` forward passes. Step counts here are correspondingly lower than
in the non-block runs.

    python src/train_block.py --k 0 --name blk_k0_s0 --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from blockdiff import (
    block_solve,
    n_blocks_for,
    sample_block_plan,
    unrolled_block_loss,
)
from diffusion import load_split, read_meta, solve_metrics
from model import ModelConfig, RegisterDiffusionTransformer

DATA_RNG_OFFSET = 1000
PLAN_RNG_OFFSET = 2000
VAL_EVAL_SEED = 987654321


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=str, default="/data/ryan.kim/registers_data")
    p.add_argument("--out", type=str, default="/data/ryan.kim/registers_runs")
    p.add_argument("--block_size", type=int, default=9)
    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    # optim
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    # eval
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--val_loss_n", type=int, default=2048)
    p.add_argument("--val_solve_n", type=int, default=256)
    p.add_argument("--final_solve_n", type=int, default=10000)
    p.add_argument("--solve_chunk", type=int, default=512)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) *
                      0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def val_block_loss(model, val, fixed, args, device, chunk=512):
    model.eval()
    tot, ntot = 0.0, 0
    for i in range(0, len(fixed["idx"]), chunk):
        sl = slice(i, i + chunk)
        idx = fixed["idx"][sl]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ce, n = unrolled_block_loss(
                model, val["puzzles"][idx], val["solutions"][idx],
                fixed["b"][sl], fixed["order"][sl], args.block_size, device,
            )
        tot += ce.item() * n
        ntot += n
    model.train()
    return tot / max(ntot, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed + DATA_RNG_OFFSET)
    plan_rng = np.random.default_rng(args.seed + PLAN_RNG_OFFSET)

    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    test = load_split(args.data, "test")
    n_train = len(train["puzzles"])

    veval = np.random.default_rng(VAL_EVAL_SEED)
    vn = min(args.val_loss_n, len(val["puzzles"]))
    v_idx = np.sort(veval.choice(len(val["puzzles"]), size=vn, replace=False))
    v_b, v_order = sample_block_plan(vn, args.block_size, veval)
    fixed = {"idx": v_idx, "b": v_b, "order": v_order}

    cfg = ModelConfig(n_registers=args.k, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=args.n_heads,
                      d_ff=args.d_ff, dropout=args.dropout)
    model = RegisterDiffusionTransformer(cfg).to(device)
    cfg.save(os.path.join(run_dir, "model_config.json"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": model.n_params(),
                   "scheme": "block_diffusion_carried_registers",
                   "n_blocks": n_blocks_for(args.block_size),
                   "data_meta": read_meta(args.data)}, f, indent=2)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    t0 = time.time()

    def log(rec):
        rec["wall_s"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    print(f"[{args.name}] K={args.k} block_size={args.block_size} "
          f"n_blocks={n_blocks_for(args.block_size)} "
          f"params={model.n_params():,} device={device}", flush=True)

    run_loss, run_n = 0.0, 0
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        idx = np.sort(data_rng.integers(0, n_train, size=args.bs))
        b, order = sample_block_plan(args.bs, args.block_size, plan_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ce, _ = unrolled_block_loss(
                model, train["puzzles"][idx], train["solutions"][idx],
                b, order, args.block_size, device)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run_loss += ce.item()
        run_n += 1

        if (step + 1) % 100 == 0:
            log({"step": step + 1, "split": "train", "ce": run_loss / run_n,
                 "lr": lr_at(step, args), "grad_norm": float(gn)})
            run_loss, run_n = 0.0, 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vce = val_block_loss(model, val, fixed, args, device)
            dec = block_solve(model, val["puzzles"][:args.val_solve_n], device,
                              args.block_size, chunk=args.solve_chunk)
            sm = solve_metrics(dec, val["solutions"][:args.val_solve_n])
            log({"step": step + 1, "split": "val", "ce": vce,
                 "exact_solve_acc": sm["exact_solve_acc"],
                 "cell_acc": sm["cell_acc"], "solve_n": sm["n"]})

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "step": step + 1, "args": vars(args)},
                       os.path.join(run_dir, "ckpt.pt"))

    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                "step": args.steps, "args": vars(args)},
               os.path.join(run_dir, "ckpt.pt"))

    final = {}
    for name, split in (("val", val), ("test", test)):
        n = min(args.final_solve_n, len(split["puzzles"]))
        dec = block_solve(model, split["puzzles"][:n], device, args.block_size,
                          chunk=args.solve_chunk)
        final[name] = solve_metrics(dec, split["solutions"][:n],
                                    split["n_clues"][:n])
    final["val_ce"] = val_block_loss(model, val, fixed, args, device)
    final.update({"k": args.k, "seed": args.seed, "block_size": args.block_size,
                  "n_params": model.n_params(),
                  "scheme": "block_diffusion_carried_registers",
                  "wall_s": round(time.time() - t0, 1)})
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL " + json.dumps({k: v for k, v in final.items()
                                 if k not in ("val", "test")}), flush=True)
    log({"step": args.steps, "split": "final",
         "val_exact": final["val"]["exact_solve_acc"],
         "test_exact": final["test"]["exact_solve_acc"],
         "val_ce": final["val_ce"]})
    logf.close()


if __name__ == "__main__":
    main()
