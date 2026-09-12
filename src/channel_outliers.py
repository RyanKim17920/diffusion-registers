"""Per-CHANNEL activation outlier statistics (Phase 0 gate).

Everything measured so far is the per-TOKEN hidden-state norm, and registers
cut that by ~40%. But per-tensor activation quantization fails on per-CHANNEL
outliers: a single input channel with an extreme absmax sets the scale for the
whole tensor and collapses every other channel into a few levels. A model can
have well-behaved token norms and still carry one catastrophic channel.

So this measures, at the input of every quantized Linear, across tokens:

    chan_absmax        per-input-channel |x| max
    max/median ratio   how far the worst channel is above a typical one --
                       this is what sets per-tensor quantization error
    kurtosis           heavy-tailedness of the channel-absmax distribution
    top1_share         worst channel's absmax / sum over channels

If registers do NOT reduce the max/median channel ratio, they cannot help
W4A4, regardless of what they do to token norms, and the quantization framing
has to be abandoned.

    python src/channel_outliers.py --runs text_k0_s0 text_k16_s0
"""

import argparse
import json
import os

import numpy as np
import torch

from model import ModelConfig, RegisterDiffusionTransformer
from text_train import VAL_EVAL_SEED, make_batch, mask_id_of
import paths


class ChannelObserver:
    """Records per-input-channel absmax at a module's input."""

    def __init__(self, name):
        self.name = name
        self.absmax = None
        self.n = 0

    def __call__(self, module, inputs):
        x = inputs[0].detach()
        c = x.abs().reshape(-1, x.shape[-1]).amax(0).float()
        self.absmax = c if self.absmax is None else torch.maximum(self.absmax, c)
        self.n += 1

    def stats(self, n_top=32):
        a = self.absmax.cpu().numpy().astype(np.float64)
        med = float(np.median(a))
        mx = float(a.max())
        mean, sd = a.mean(), a.std()
        kurt = float((((a - mean) / (sd + 1e-12)) ** 4).mean()) if sd > 0 else 0.0
        srt = np.sort(a)[::-1]
        tot = max(a.sum(), 1e-12)
        # The SHAPE matters, not just the summary: a handful of catastrophic
        # channels and a broad heavy tail give the same max/median but call
        # for different fixes (per-channel handling vs rotation). Keep the
        # sorted head and the quantile profile so the distribution is visible.
        return {
            "max": mx,
            "median": med,
            "max_over_median": mx / max(med, 1e-12),
            "kurtosis": kurt,
            "top1_share": mx / tot,
            "top8_share": float(srt[:8].sum() / tot),
            "top32_share": float(srt[:32].sum() / tot),
            "n_over_4x_median": int((a > 4 * med).sum()),
            "n_over_10x_median": int((a > 10 * med).sum()),
            "quantiles": {q: float(np.quantile(a, q))
                          for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
            "top_channels": [float(x) for x in srt[:n_top]],
            "n_channels": int(a.size),
        }


def attach(model):
    obs, handles = {}, []
    for li, blk in enumerate(model.blocks):
        for name in ("qkv", "proj", "fc1", "fc2"):
            o = ChannelObserver(f"L{li}.{name}")
            handles.append(getattr(blk, name).register_forward_pre_hook(o))
            obs[o.name] = o
    return obs, handles


@torch.no_grad()
def collect(run_dir, data, device, batches, bs):
    cfg = ModelConfig.load(os.path.join(run_dir, "model_config.json"))
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    model = RegisterDiffusionTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    obs, handles = attach(model)
    stream = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    rng = np.random.default_rng(VAL_EVAL_SEED)
    for _ in range(batches):
        x, m, _ = make_batch(stream, bs, cfg.seq_len, rng, rng)
        xb = torch.as_tensor(x, device=device)
        mb = torch.as_tensor(m, device=device)
        inp = torch.where(mb, torch.full_like(xb, mask_id_of(model)), xb)
        model(inp, return_hidden=True)
    for h in handles:
        h.remove()
    return cfg.n_registers, {k: o.stats() for k, o in obs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directory names under --root")
    ap.add_argument("--root", default=paths.RUNS)
    ap.add_argument("--data", default=paths.TEXT_DATA)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_res = {}
    for name in args.runs:
        d = os.path.join(args.root, name)
        k, st = collect(d, args.data, device, args.batches, args.bs)
        all_res[name] = {"k": k, "modules": st}
        ratios = [v["max_over_median"] for v in st.values()]
        kurts = [v["kurtosis"] for v in st.values()]
        print(f"\n=== {name}  K={k} ===")
        print(f"  per-channel max/median  mean {np.mean(ratios):8.2f}   "
              f"worst module {np.max(ratios):8.2f}")
        print(f"  channel kurtosis        mean {np.mean(kurts):8.2f}   "
              f"worst module {np.max(kurts):8.2f}")
        worst = max(st.items(), key=lambda kv: kv[1]["max_over_median"])
        w = worst[1]
        print(f"  worst module: {worst[0]}  max/median {w['max_over_median']:.1f}")
        print(f"    spike shape: top1 {w['top1_share']:.4f} of total, "
              f"top8 {w['top8_share']:.4f}, top32 {w['top32_share']:.4f}")
        print(f"    channels over 4x median: {w['n_over_4x_median']:>4} / "
              f"{w['n_channels']}   over 10x: {w['n_over_10x_median']}")
        qs = w["quantiles"]
        print("    channel-absmax quantiles  " + "  ".join(
            f"p{int(q * 100) if q < 1 else 'max'}={v:.1f}" for q, v in qs.items()))
        print("    top 12 channels: " + " ".join(f"{x:.0f}" for x in
                                                 w["top_channels"][:12]))
        n_spiky = sum(1 for v in st.values() if v["n_over_10x_median"] > 0)
        print(f"  modules with any channel >10x median: {n_spiky}/{len(st)}")

    # K=0 vs K>0 comparison, which is the actual gate
    zero = [n for n, v in all_res.items() if v["k"] == 0]
    reg = [n for n, v in all_res.items() if v["k"] > 0]
    if zero and reg:
        def agg(names, field):
            return np.mean([np.mean([m[field] for m in all_res[n]["modules"].values()])
                            for n in names])
        print("\n=== GATE: does the per-channel outlier shrink? ===")
        for field in ("max_over_median", "kurtosis", "top1_share",
                      "top8_share", "n_over_4x_median", "n_over_10x_median"):
            a, b = agg(zero, field), agg(reg, field)
            print(f"  {field:<16} K=0 {a:10.3f}   K>0 {b:10.3f}   "
                  f"change {100 * (b - a) / a:+7.1f}%")
        print("\n  (per-TOKEN max norm fell ~40% with registers; if these "
              "per-CHANNEL\n   numbers do not move, registers cannot help W4A4)")

    out = args.out or os.path.join(args.root, "channel_outliers.json")
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
