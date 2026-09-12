"""Analysis for block-diffusion runs with carried register state.

The question this answers is narrower than the stateless analysis: does the
model actually USE the carry?

  1. Ablations at decode time:
       normal    -- registers carried across the steps of a block
       no_carry  -- registers re-initialised from the learned embeddings at
                    every step (severs the scratchpad channel, changes nothing
                    else). This is the decisive one.
       zero      -- registers zeroed at every step
       shuffle   -- register slots permuted per example
  2. Carry dynamics: how the carried state moves from step to step within a
     block (norm, step-to-step distance, cosine similarity to the previous
     state and to the learned initialisation). A state that never moves, or
     that instantly forgets its initialisation, is not a scratchpad.

    python src/analyze_block.py --run runs/blk_k16_s0
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from blockdiff import block_solve, n_blocks_for
from diffusion import load_split, solve_metrics
from model import MASK, N_CELLS, ModelConfig, RegisterDiffusionTransformer, build_tokens
import paths


def load_run(run_dir, device):
    cfg = ModelConfig.load(os.path.join(run_dir, "model_config.json"))
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    model = RegisterDiffusionTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("step")


@torch.no_grad()
def carry_dynamics(model, puzzles, device, block_size, n=128, block=4):
    """Track the carried register state across the steps of one block."""
    if model.K == 0:
        return None
    p = torch.as_tensor(np.ascontiguousarray(puzzles[:n]), device=device,
                        dtype=torch.long)
    B = p.shape[0]
    sol = torch.full((B, N_CELLS), MASK, dtype=torch.long, device=device)
    cell_block = torch.arange(N_CELLS, device=device) // block_size
    # bring the grid up to the start of `block` by decoding earlier blocks
    for b in range(block):
        in_block = (cell_block == b)[None, :].expand(B, -1)
        st = None
        for _ in range(block_size):
            logits, st = model(build_tokens(p, sol), reg_state=st,
                               return_reg_state=True)
            conf, digit = logits.float().softmax(-1).max(-1)
            conf = conf.masked_fill(~in_block | (sol != MASK), -1.0)
            pick = conf.argmax(-1, keepdim=True)
            sol.scatter_(1, pick, torch.gather(digit, 1, pick) + 1)

    init = model.reg_emb.unsqueeze(0).expand(B, -1, -1).float()
    in_block = (cell_block == block)[None, :].expand(B, -1)
    st, prev = None, None
    out = {"step_norm": [], "delta_from_prev": [], "cos_to_prev": [],
           "cos_to_init": []}
    for _ in range(block_size):
        logits, st = model(build_tokens(p, sol), reg_state=st,
                           return_reg_state=True)
        cur = st.float()
        out["step_norm"].append(cur.norm(dim=-1).mean().item())
        out["cos_to_init"].append(
            F.cosine_similarity(cur, init, dim=-1).mean().item())
        if prev is None:
            out["delta_from_prev"].append(None)
            out["cos_to_prev"].append(None)
        else:
            out["delta_from_prev"].append((cur - prev).norm(dim=-1).mean().item())
            out["cos_to_prev"].append(
                F.cosine_similarity(cur, prev, dim=-1).mean().item())
        prev = cur
        conf, digit = logits.float().softmax(-1).max(-1)
        conf = conf.masked_fill(~in_block | (sol != MASK), -1.0)
        pick = conf.argmax(-1, keepdim=True)
        sol.scatter_(1, pick, torch.gather(digit, 1, pick) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=paths.SUDOKU_DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_solve", type=int, default=2000)
    ap.add_argument("--n_dyn", type=int, default=128)
    ap.add_argument("--solve_chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_run(args.run, device)
    with open(os.path.join(args.run, "config.json")) as f:
        rcfg = json.load(f)
    block_size = rcfg.get("block_size", 9)
    split = load_split(args.data, args.split)
    torch.manual_seed(args.seed)

    n = min(args.n_solve, len(split["puzzles"]))
    res = {"run": args.run, "k": cfg.n_registers, "step": step,
           "block_size": block_size, "n_blocks": n_blocks_for(block_size),
           "n_solve": n, "ablations": {}}

    modes = ["normal"] if cfg.n_registers == 0 else \
        ["normal", "no_carry", "zero", "shuffle"]
    for mode in modes:
        dec = block_solve(model, split["puzzles"][:n], device, block_size,
                          reg_mode=mode, chunk=args.solve_chunk)
        res["ablations"][mode] = solve_metrics(
            dec, split["solutions"][:n], split["n_clues"][:n])

    dyn = carry_dynamics(model, split["puzzles"], device, block_size,
                         n=args.n_dyn)
    if dyn:
        res["carry_dynamics"] = dyn

    out = args.out or os.path.join(args.run, "analysis_block.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {os.path.basename(args.run)}  K={cfg.n_registers}  "
          f"block_size={block_size}  step={step} ===")
    base = res["ablations"]["normal"]["exact_solve_acc"]
    for mode, m in res["ablations"].items():
        d = m["exact_solve_acc"] - base
        print(f"  {mode:<9} exact={m['exact_solve_acc']:.4f} ({d:+.4f})  "
              f"cell={m['cell_acc']:.4f}  (n={m['n']})")
    if dyn:
        print("  carry dynamics within one block:")
        print("    step   |state|   d(prev)   cos(prev)   cos(init)")
        for j in range(len(dyn["step_norm"])):
            dp = dyn["delta_from_prev"][j]
            cp = dyn["cos_to_prev"][j]
            print(f"    {j:<6} {dyn['step_norm'][j]:>7.2f}   "
                  f"{'-' if dp is None else f'{dp:7.2f}'}   "
                  f"{'-' if cp is None else f'{cp:9.4f}'}   "
                  f"{dyn['cos_to_init'][j]:>9.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
