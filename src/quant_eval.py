"""Post-training quantization evaluation.

Tests the claim the activation-norm result implies: if registers pull
high-norm outliers off the ordinary tokens, then a register model should
survive low-bit *activation* quantization better than its K=0 baseline.
Activation outliers are what break per-tensor INT8 -- one extreme channel
sets the scale and everything else collapses into a few levels.

Simulated (fake) quantization, which is the standard way to measure PTQ
degradation without a kernel:

  weights      per-output-channel symmetric absmax  (outliers are not the
               problem on the weight side, so this is the easy axis)
  activations  per-tensor symmetric, calibrated on a held-out split; this is
               the axis activation outliers destroy

Reported as validation denoising CE at full precision vs quantized, and the
degradation between them. The degradation is the number that matters -- a
register model could be worse in absolute CE yet degrade far less.

    python src/quant_eval.py --run runs/text_k16_s0 \
        --w_bits 8 --a_bits 8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig, RegisterDiffusionTransformer
from text_train import VAL_EVAL_SEED, make_batch, mask_id_of
import paths


def quantize_per_tensor(x, scale, bits):
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


class QuantLinear(nn.Module):
    """Wraps an nn.Linear with fake quantization.

    Two passes: `calibrating=True` records the activation absmax (or a high
    percentile, which is the usual way to blunt a few extreme values), then
    `calibrating=False` quantizes with the recorded scale.
    """

    def __init__(self, lin, w_bits=8, a_bits=8, a_percentile=100.0):
        super().__init__()
        self.lin = lin
        self.w_bits, self.a_bits = w_bits, a_bits
        self.a_percentile = a_percentile
        self.register_buffer("a_scale", torch.zeros(1))
        self.calibrating = False
        self._obs = []
        self._wq = None
        self.act_div = None        # SmoothQuant per-channel divisor
        self.chan_absmax = None    # per-input-channel absmax, for SmoothQuant

    def _quant_weight(self):
        if self._wq is not None:
            return self._wq
        w = self.lin.weight
        if self.w_bits >= 16:
            self._wq = w
            return w
        qmax = 2 ** (self.w_bits - 1) - 1
        # per-output-channel scale
        s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        self._wq = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
        return self._wq

    def forward(self, x):
        if self.act_div is not None:
            x = x / self.act_div
        if self.calibrating:
            with torch.no_grad():
                c = x.detach().abs().reshape(-1, x.shape[-1]).amax(0)
                self.chan_absmax = c if self.chan_absmax is None else \
                    torch.maximum(self.chan_absmax, c)
                if self.a_percentile >= 100.0:
                    v = x.abs().max()
                else:
                    v = torch.quantile(
                        x.abs().float().flatten()[:1_000_000],
                        self.a_percentile / 100.0)
                self._obs.append(v.item())
            return self.lin(x)
        if self.a_bits < 16:
            qmax = 2 ** (self.a_bits - 1) - 1
            x = quantize_per_tensor(x, self.a_scale.clamp(min=1e-8) / qmax,
                                    self.a_bits)
        return F.linear(x, self._quant_weight(), self.lin.bias)

    def finish_calibration(self):
        if self._obs:
            self.a_scale.fill_(float(np.mean(self._obs)))
        self._obs = []


@torch.no_grad()
def apply_smoothquant(model, acts, alpha):
    """Per-channel activation->weight magnitude migration (SmoothQuant).

    `acts` maps each wrapped module to the per-input-channel absmax observed
    during calibration. Dividing the activation by s and multiplying the
    weight column by s leaves the product unchanged in full precision, but
    moves outlier magnitude from the activation (per-tensor quantized, and
    therefore fragile) into the weights (per-channel quantized, and robust).
    """
    for q, a_absmax in acts.items():
        w_absmax = q.lin.weight.abs().amax(dim=0).clamp(min=1e-5)
        s = (a_absmax.clamp(min=1e-5) ** alpha) / (w_absmax ** (1 - alpha))
        s = s.clamp(min=1e-5)
        q.lin.weight.mul_(s.unsqueeze(0))
        q.act_div = s
        q._wq = None


def wrap_model(model, w_bits, a_bits, a_percentile):
    """Replace every Linear inside the transformer blocks. The embedding and
    the output head are left alone: PTQ studies quantize the body, and tying
    the head to the embedding makes head quantization a separate question."""
    wrapped = []
    for blk in model.blocks:
        for name in ("qkv", "proj", "fc1", "fc2"):
            lin = getattr(blk, name)
            q = QuantLinear(lin, w_bits, a_bits, a_percentile).to(lin.weight.device)
            setattr(blk, name, q)
            wrapped.append(q)
    return wrapped


@torch.no_grad()
def val_ce(model, stream, seq_len, bs, batches, device, calibrate=None):
    rng = np.random.default_rng(VAL_EVAL_SEED)
    tot, ntok = 0.0, 0
    for _ in range(batches):
        x, m, _ = make_batch(stream, bs, seq_len, rng, rng)
        xb = torch.as_tensor(x, device=device)
        mb = torch.as_tensor(m, device=device)
        inp = torch.where(mb, torch.full_like(xb, mask_id_of(model)), xb)
        h = model(inp, return_hidden=True)
        if calibrate:
            continue
        sel = mb.reshape(-1)
        hsel = h.reshape(-1, h.shape[-1])[sel]
        tgt = xb.reshape(-1)[sel]
        ce = F.cross_entropy(model.head(hsel).float(), tgt, reduction="sum")
        tot += ce.item()
        ntok += int(hsel.shape[0])
    return tot / max(ntok, 1) if ntok else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=paths.TEXT_DATA)
    # W8A8 is too easy to separate anything: the dLLM PTQ literature
    # (arXiv 2508.14896) reports that dLLMs break at W4A4, where SmoothQuant
    # falls to near-zero. The interesting operating points are the low-bit
    # ACTIVATION ones, since activations are what outliers corrupt.
    ap.add_argument("--w_bits", type=int, nargs="*",
                    default=[8, 8, 8, 4, 4])
    ap.add_argument("--a_bits", type=int, nargs="*",
                    default=[16, 8, 6, 8, 4])
    ap.add_argument("--a_percentile", type=float, default=100.0)
    ap.add_argument("--calib_batches", type=int, default=8)
    ap.add_argument("--val_batches", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="SmoothQuant-style migration strength alpha in [0,1): "
                         "scale activations down and weights up by "
                         "s = max|X|^alpha / max|W|^(1-alpha), moving outlier "
                         "magnitude off the activations. 0 disables it. This is "
                         "the post-training REPAIR baseline that registers "
                         "(a training-time fix) have to be compared against.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert len(args.w_bits) == len(args.a_bits), \
        "--w_bits and --a_bits must be the same length (they pair up)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig.load(os.path.join(args.run, "model_config.json"))
    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location="cpu")
    val_stream = np.load(os.path.join(args.data, "val.npy"), mmap_mode="r")

    res = {"run": args.run, "k": cfg.n_registers, "seq_len": cfg.seq_len,
           "a_percentile": args.a_percentile, "smooth_alpha": args.smooth,
           "settings": []}

    # full precision reference, from a clean copy of the weights
    model = RegisterDiffusionTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    fp = val_ce(model, val_stream, cfg.seq_len, args.bs, args.val_batches, device)
    res["fp_ce"] = fp
    print(f"\n=== {os.path.basename(args.run)}  K={cfg.n_registers} ===")
    print(f"  full precision val CE {fp:.4f}")

    for wb, ab in zip(args.w_bits, args.a_bits):
        model = RegisterDiffusionTransformer(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        qs = wrap_model(model, wb, ab, args.a_percentile)
        for q in qs:
            q.calibrating = True
        val_ce(model, val_stream, cfg.seq_len, args.bs, args.calib_batches,
               device, calibrate=True)
        if args.smooth > 0:
            # migrate outlier magnitude activation->weight, then RE-calibrate:
            # the activation distribution has changed, so the old per-tensor
            # scale no longer describes it
            apply_smoothquant(model, {q: q.chan_absmax for q in qs},
                              args.smooth)
            for q in qs:
                q._obs, q.chan_absmax = [], None
            val_ce(model, val_stream, cfg.seq_len, args.bs, args.calib_batches,
                   device, calibrate=True)
        for q in qs:
            q.calibrating = False
            q.finish_calibration()
        ce = val_ce(model, val_stream, cfg.seq_len, args.bs, args.val_batches,
                    device)
        entry = {"w_bits": wb, "a_bits": ab, "ce": ce, "degradation": ce - fp,
                 "smooth_alpha": args.smooth}
        res["settings"].append(entry)
        print(f"  W{wb}A{ab:<3} val CE {ce:.4f}   degradation {ce - fp:+.4f}")

    tag = "" if args.smooth == 0 else f"_sq{args.smooth:g}"
    out = args.out or os.path.join(args.run, f"quant_eval{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
