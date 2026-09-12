"""Register analysis for text MDLM checkpoints.

The measurement this exists for: does TEXT show the high-norm artifact tokens
that Sudoku lacked? That is the standing explanation for the Sudoku null -- no
artifact for a register to absorb -- and it is only testable by looking at the
K=0 token-norm profile on a setting where the pathology is reported.

Also reports text->register attention and the zero/shuffle ablations, scored
by validation denoising CE (no ceiling, unlike exact-solve accuracy).

    python src/analyze_text.py --run /data/ryan.kim/registers_runs/text_k16_s0
"""

import argparse
import json
import os

import numpy as np
import torch

from analyze import attention_stats, norm_stats
from model import ModelConfig, RegisterDiffusionTransformer
from text_train import VAL_EVAL_SEED, loss_on_batch, make_batch, mask_id_of


def load_run(run_dir, device):
    cfg = ModelConfig.load(os.path.join(run_dir, "model_config.json"))
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    model = RegisterDiffusionTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("step")


@torch.no_grad()
def val_ce(model, stream, seq_len, bs, batches, device, reg_mode="normal"):
    rng = np.random.default_rng(VAL_EVAL_SEED)
    tot, ntok = 0.0, 0
    for _ in range(batches):
        x, m, _ = make_batch(stream, bs, seq_len, rng, rng)
        xb = torch.as_tensor(x, device=device)
        mb = torch.as_tensor(m, device=device)
        inp = torch.where(mb, torch.full_like(xb, mask_id_of(model)), xb)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            h = model(inp, reg_mode=reg_mode, return_hidden=True)
            sel = mb.reshape(-1)
            hsel = h.reshape(-1, h.shape[-1])[sel]
            tgt = xb.reshape(-1)[sel]
            import torch.nn.functional as F
            ce = F.cross_entropy(model.head(hsel).float(), tgt, reduction="sum")
        tot += ce.item()
        ntok += int(hsel.shape[0])
    return tot / max(ntok, 1)


def fixed_tokens(model, stream, seq_len, n, mask_frac, device, seed=0):
    """A fixed batch of partially masked text windows for the norm/attention
    statistics."""
    rng = np.random.default_rng(seed)
    x = np.stack([stream[s:s + seq_len] for s in
                  rng.integers(0, len(stream) - seq_len - 1, size=n)]
                 ).astype(np.int64)
    xb = torch.as_tensor(x, device=device)
    m = torch.as_tensor(rng.random((n, seq_len)) < mask_frac, device=device)
    return torch.where(m, torch.full_like(xb, mask_id_of(model)), xb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="/data/ryan.kim/registers_text_data")
    ap.add_argument("--n_stats", type=int, default=64)
    ap.add_argument("--mask_frac", type=float, default=0.5)
    ap.add_argument("--val_batches", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_run(args.run, device)
    with open(os.path.join(args.run, "config.json")) as f:
        rcfg = json.load(f)
    val_stream = np.load(os.path.join(args.data, "val.npy"), mmap_mode="r")
    seq_len, bs = cfg.seq_len, rcfg.get("bs", 32)

    res = {"run": args.run, "k": cfg.n_registers, "step": step,
           "seq_len": seq_len}

    tokens = fixed_tokens(model, val_stream, seq_len, args.n_stats,
                          args.mask_frac, device)
    res["norms"] = norm_stats(model, tokens, chunk=16)
    a = attention_stats(model, tokens, chunk=8)
    if a:
        res["attention"] = a

    res["ablations"] = {}
    modes = ["normal"] if cfg.n_registers == 0 else ["normal", "zero", "shuffle"]
    for mode in modes:
        res["ablations"][mode] = val_ce(model, val_stream, seq_len, bs,
                                        args.val_batches, device, mode)

    out = args.out or os.path.join(args.run, "analysis_text.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {os.path.basename(args.run)}  K={cfg.n_registers}  step={step} ===")
    for mode, v in res["ablations"].items():
        d = v - res["ablations"]["normal"]
        print(f"  val CE  {mode:<8} {v:.4f} ({d:+.4f})")
    n = res["norms"]
    print(f"  token norms by layer (outlier = >{n['outlier_mult']}x median):")
    print("    layer   tok_norm   tok_max   p99.9   outlier_frac   argmax_pos")
    for l in range(len(n["token_norm"])):
        print(f"    {l:<6} {n['token_norm'][l]:>9.2f} {n['token_norm_max'][l]:>9.2f} "
              f"{n['token_norm_p999'][l]:>7.1f} {n['token_outlier_frac'][l]:>14.4f} "
              f"{n['token_norm_argmax_pos'][l]:>12.1f}")
    if a:
        print(f"  text->register attention (uniform baseline "
              f"{a['uniform_baseline']:.4f}):")
        print("    " + "  ".join(f"L{i}:{v:.4f}"
                                 for i, v in enumerate(a["text_to_register"])))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
