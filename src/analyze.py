"""Register analysis for a trained checkpoint (phase 2).

Reports, for one run:
  * text->register and register->text attention mass, per layer (and per head)
  * register hidden-state L2 norms per layer, against the token-norm baseline
  * causal-importance ablation: exact-solve accuracy with reg_mode
    normal / zero / shuffle
  * exact-solve accuracy by clue count

    python src/analyze.py --run /data/ryan.kim/registers_runs/k4_s0 --n_solve 2000
"""

import argparse
import json
import os

import numpy as np
import torch

from diffusion import load_split, solve, solve_metrics
from model import (
    MASK,
    N_CELLS,
    SEQ_REAL,
    ModelConfig,
    RegisterDiffusionTransformer,
    build_tokens,
)


def load_run(run_dir, device):
    cfg = ModelConfig.load(os.path.join(run_dir, "model_config.json"))
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    model = RegisterDiffusionTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("step")


def make_state(val, n, frac_revealed, rng, device):
    """Tokens at a partially-denoised state: `frac_revealed` of the 81 solution
    positions teacher-forced to the true digit, the rest MASK."""
    p = torch.as_tensor(val["puzzles"][:n], device=device, dtype=torch.long)
    s = torch.as_tensor(val["solutions"][:n], device=device, dtype=torch.long)
    keep = torch.as_tensor(
        rng.random((n, N_CELLS)) < frac_revealed, device=device
    )
    sol = torch.where(keep, s, torch.full_like(s, MASK))
    return build_tokens(p, sol)


@torch.no_grad()
def attention_stats(model, tokens, chunk=64):
    """Attention mass between the text block and the register block.

    text2reg[l] = mean over heads and text queries of the attention mass a
    text position puts on register keys.
    reg2text[l] = mean over heads and register queries of the mass a register
    puts on text keys.
    Also returns the per-head text2reg matrix (layers x heads).
    """
    K = model.K
    if K == 0:
        return None
    L = len(model.blocks)
    t2r = np.zeros(L)
    r2t = np.zeros(L)
    t2r_head = None
    nb = 0
    for i in range(0, tokens.shape[0], chunk):
        tk = tokens[i:i + chunk]
        _, att = model(tk, collect_attn=True)  # list of (B, H, T, T)
        if t2r_head is None:
            t2r_head = np.zeros((L, att[0].shape[1]))
        for l, a in enumerate(att):
            a = a.float()
            # queries = text rows [:SEQ_REAL], keys = register cols [-K:]
            m = a[:, :, :SEQ_REAL, -K:].sum(-1)      # (B, H, SEQ_REAL)
            t2r[l] += m.mean().item()
            t2r_head[l] += m.mean(dim=(0, 2)).cpu().numpy()
            rr = a[:, :, -K:, :SEQ_REAL].sum(-1)     # (B, H, K)
            r2t[l] += rr.mean().item()
        nb += 1
    return {
        "text_to_register": (t2r / nb).tolist(),
        "register_to_text": (r2t / nb).tolist(),
        "text_to_register_per_head": (t2r_head / nb).tolist(),
        "uniform_baseline": K / (SEQ_REAL + K),
    }


@torch.no_grad()
def norm_stats(model, tokens, chunk=256):
    """Per-layer mean L2 norm of register vs text hidden states."""
    reg, tok = None, None
    nb = 0
    for i in range(0, tokens.shape[0], chunk):
        hs = model.hidden_states(tokens[i:i + chunk])
        r = [h[:, SEQ_REAL:, :].float().norm(dim=-1).mean().item() if model.K else 0.0
             for h in hs]
        t = [h[:, :SEQ_REAL, :].float().norm(dim=-1).mean().item() for h in hs]
        reg = np.array(r) if reg is None else reg + np.array(r)
        tok = np.array(t) if tok is None else tok + np.array(t)
        nb += 1
    return {"register_norm": (reg / nb).tolist(),
            "token_norm": (tok / nb).tolist(),
            "layers": "index 0 is the embedding output, then one per block"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="/data/ryan.kim/registers_data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_solve", type=int, default=2000,
                    help="puzzles decoded per ablation arm")
    ap.add_argument("--n_attn", type=int, default=256,
                    help="examples used for attention / norm statistics")
    ap.add_argument("--solve_chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_run(args.run, device)
    split = load_split(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    res = {"run": args.run, "k": cfg.n_registers, "step": step,
           "split": args.split, "n_solve": min(args.n_solve, len(split["puzzles"]))}

    # --- attention and norms at two denoising states
    res["states"] = {}
    for tag, frac in (("all_masked", 0.0), ("half_revealed", 0.5)):
        tokens = make_state(split, min(args.n_attn, len(split["puzzles"])),
                            frac, rng, device)
        entry = {"norms": norm_stats(model, tokens)}
        a = attention_stats(model, tokens)
        if a:
            entry["attention"] = a
        res["states"][tag] = entry

    # --- causal importance of the registers
    n = res["n_solve"]
    modes = ["normal"] if cfg.n_registers == 0 else ["normal", "zero", "shuffle"]
    res["ablations"] = {}
    for mode in modes:
        dec = solve(model, split["puzzles"][:n], device, reg_mode=mode,
                    chunk=args.solve_chunk)
        res["ablations"][mode] = solve_metrics(
            dec, split["solutions"][:n], split["n_clues"][:n]
        )

    out = args.out or os.path.join(args.run, f"analysis_{args.split}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    # --- human-readable summary
    print(f"\n=== {os.path.basename(args.run)}  K={cfg.n_registers}  step={step} ===")
    for mode, m in res["ablations"].items():
        print(f"  reg_mode={mode:<8} exact={m['exact_solve_acc']:.4f}  "
              f"cell={m['cell_acc']:.4f}  (n={m['n']})")
    if cfg.n_registers:
        for tag, e in res["states"].items():
            a = e["attention"]
            print(f"  [{tag}] uniform-attention baseline = "
                  f"{a['uniform_baseline']:.4f}")
            print("    layer  text->reg  reg->text   reg_norm  tok_norm")
            for l in range(len(a["text_to_register"])):
                print(f"    {l:<6} {a['text_to_register'][l]:>9.4f}  "
                      f"{a['register_to_text'][l]:>9.4f}  "
                      f"{e['norms']['register_norm'][l + 1]:>9.2f}  "
                      f"{e['norms']['token_norm'][l + 1]:>8.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
