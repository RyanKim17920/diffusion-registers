"""Activation-outlier and PTQ profile of REAL released models.

Motivation for the register work rests on a claim that has to be true outside
our own 51M toys: deployed masked-diffusion LMs carry high-norm activation
outliers, and those outliers are what breaks per-tensor INT8.

This profiles any HF causal/diffusion LM with the same statistics used on our
models, so the numbers are comparable:

  * per-layer hidden-state norms: mean, max, 99.9th percentile
  * outlier fraction (tokens above `mult` x the per-sequence median)
  * WHERE the outliers sit -- argmax-position entropy and the mass on position
    0. This separates the autoregressive attention-sink story (positional,
    position 0 elected because causal masking makes it visible to every query)
    from a content-dependent one. A bidirectional diffusion LM has no
    privileged position, so if its outliers are content-dependent the AR fix
    "keep token 0" has nothing to keep.
  * simulated W8A8 degradation on the same text

    python src/profile_real_models.py --model GSAI-ML/LLaDA-8B-Base \
        --diffusion --n_seq 16 --seq_len 1024
"""

import argparse
import json
import math
import os

import numpy as np
import torch


def layer_stats(h, mult=3.0):
    """h: (B, T, D) hidden states -> norm statistics over token positions."""
    n = h[:, :, :].float().norm(dim=-1)               # (B, T)
    med = n.median(dim=1, keepdim=True).values
    am = n.argmax(dim=1)
    return {
        "mean": n.mean().item(),
        "max": n.max(dim=1).values.mean().item(),
        "p999": n.flatten().quantile(0.999).item(),
        "outlier_frac": (n > mult * med).float().mean().item(),
        "argmax": am.cpu(),
    }


def position_entropy(argmax_all, seq_len):
    c = torch.bincount(torch.cat(argmax_all), minlength=seq_len).float()
    p = c / c.sum()
    nz = p[p > 0]
    return {
        "normalised_entropy": float(-(nz * nz.log()).sum() / math.log(seq_len)),
        "frac_pos0": float(p[0]),
        "frac_first4": float(p[:4].sum()),
        "top5_positions": [int(i) for i in torch.topk(c, 5).indices],
    }


class FakeQuantLinear(torch.nn.Module):
    """Per-channel INT-N weights, per-tensor INT-N activations, calibrated.

    Architecture-agnostic: wraps whatever nn.Linear modules a released model
    happens to use, so the same PTQ measurement applies to our models and to
    an 8B diffusion LM. Per-tensor activation quantization is the setting that
    activation outliers destroy -- one extreme value sets the scale.
    """

    def __init__(self, lin, w_bits, a_bits, conv1d=False):
        super().__init__()
        self.lin, self.w_bits, self.a_bits = lin, w_bits, a_bits
        # transformers' Conv1D stores weight as (in, out) and computes
        # x @ W + b, so the output channel axis is 0 rather than 1
        self.conv1d = conv1d
        self.register_buffer("a_absmax", torch.zeros(1, device=lin.weight.device))
        self.calibrating = True
        self._wq = None

    def _qw(self):
        if self._wq is None:
            w = self.lin.weight.float()
            qmax = 2 ** (self.w_bits - 1) - 1
            ch = 0 if self.conv1d else 1
            sc = w.abs().amax(dim=ch, keepdim=True).clamp(min=1e-8) / qmax
            self._wq = (torch.clamp(torch.round(w / sc), -qmax - 1, qmax) * sc
                        ).to(self.lin.weight.dtype)
        return self._wq

    def forward(self, x):
        if self.calibrating:
            with torch.no_grad():
                self.a_absmax.fill_(max(self.a_absmax.item(),
                                        x.detach().abs().max().float().item()))
            return self.lin(x)
        qmax = 2 ** (self.a_bits - 1) - 1
        sc = (self.a_absmax / qmax).clamp(min=1e-8).to(x.dtype)
        xq = torch.clamp(torch.round(x / sc), -qmax - 1, qmax) * sc
        if self.conv1d:
            return torch.addmm(self.lin.bias, xq.view(-1, xq.shape[-1]),
                               self._qw()).view(*xq.shape[:-1], -1)
        return torch.nn.functional.linear(xq, self._qw(), self.lin.bias)


def _is_conv1d(m):
    return type(m).__name__ == "Conv1D" and hasattr(m, "weight")


def swap_linears(module, w_bits, a_bits, skip=("lm_head", "embed", "router",
                                               "gate")):
    """Wrap every projection except the head/embedding/MoE-router.

    Handles nn.Linear (LLaMA/Qwen/LLaDA) and transformers' Conv1D (GPT-2).
    Callers must check the returned count: a silent zero-wrap would report
    zero quantization degradation, which reads as "quantization is free"
    rather than "nothing was quantized".
    """
    wrapped = []
    for name, child in module.named_children():
        full = name.lower()
        hit = isinstance(child, torch.nn.Linear) or _is_conv1d(child)
        if hit and not any(s in full for s in skip):
            q = FakeQuantLinear(child, w_bits, a_bits, conv1d=_is_conv1d(child))
            setattr(module, name, q)
            wrapped.append(q)
        else:
            wrapped += swap_linears(child, w_bits, a_bits, skip)
    return wrapped


@torch.no_grad()
def masked_ce(model, tok, texts, seq_len, mask_id, mask_frac, batch, device,
              seed=0):
    """Denoising CE at masked positions -- the same quantity our runs report."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot, n = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        ids = enc["input_ids"].to(device)
        m = (torch.rand(ids.shape, generator=g) < mask_frac).to(device)
        inp = torch.where(m, torch.full_like(ids, mask_id), ids)
        out = model(input_ids=inp)
        lg = out.logits if hasattr(out, "logits") else out[0]
        sel = m.reshape(-1)
        if sel.sum() == 0:
            continue
        ce = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1])[sel].float(),
            ids.reshape(-1)[sel], reduction="sum")
        tot += ce.item()
        n += int(sel.sum())
    return tot / max(n, 1)


@torch.no_grad()
def channel_stats(model, tok, texts, seq_len, mask_id, mask_frac, batch,
                  device, diffusion):
    """Per-input-channel absmax at every Linear, the axis per-tensor INT
    quantization fails on. Same statistic as src/channel_outliers.py so our
    models and released ones are directly comparable.

    The point of running this on a released model: it separates "our models
    are too SMALL to show massive outliers" from "our models are too
    UNDERTRAINED to show them" -- mdlm-owt is comparable in size to our rungs
    but trained on far more tokens.
    """
    obs, handles = {}, []

    def mk(name):
        def hook(mod, inp):
            x = inp[0].detach()
            c = x.abs().reshape(-1, x.shape[-1]).amax(0).float()
            obs[name] = c if name not in obs else torch.maximum(obs[name], c)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) or type(mod).__name__ == "Conv1D":
            if any(k in name.lower() for k in ("lm_head", "embed", "router")):
                continue
            handles.append(mod.register_forward_pre_hook(mk(name)))
    if not handles:
        raise RuntimeError("no projections hooked -- unrecognised module types")

    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        ids = enc["input_ids"].to(device)
        if diffusion:
            m = torch.rand(ids.shape, device=device) < mask_frac
            ids = torch.where(m, torch.full_like(ids, mask_id), ids)
        model(input_ids=ids)
    for h in handles:
        h.remove()

    out = {}
    for name, a in obs.items():
        v = a.cpu().numpy().astype(np.float64)
        med = float(np.median(v))
        out[name] = {
            "max_over_median": float(v.max()) / max(med, 1e-12),
            "n_over_10x_median": int((v > 10 * med).sum()),
            "n_channels": int(v.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--diffusion", action="store_true",
                    help="feed a partially MASKED sequence, as a diffusion LM "
                         "sees at inference, rather than clean text")
    ap.add_argument("--mask_frac", type=float, default=0.5)
    ap.add_argument("--mask_token", default=None,
                    help="mask token id or string; defaults to the "
                         "tokenizer's mask token, else 126336 (LLaDA)")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer repo to use instead of the model's own; "
                         "some checkpoints (e.g. mdlm-owt) ship a config the "
                         "fast tokenizer cannot instantiate but are plain "
                         "GPT-2 BPE underneath")
    ap.add_argument("--data", default="/data/ryan.kim/registers_text_data")
    ap.add_argument("--out", default="/data/ryan.kim/registers_runs/real_models")
    ap.add_argument("--channels", action="store_true",
                    help="per-input-channel outlier stats (comparable to "
                         "src/channel_outliers.py on our own runs)")
    ap.add_argument("--ptq", action="store_true",
                    help="also measure W8A8 / W8A6 PTQ degradation")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data/huggingface")
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    tok_src = args.tokenizer or args.model
    try:
        tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    except Exception as e:
        print(f"tokenizer {tok_src} failed ({type(e).__name__}); falling back "
              f"to gpt2 -- verify this matches the model's vocabulary")
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    # Checkpoints with custom configs are not registered for every AutoModel
    # class, so try the plausible ones in order rather than assuming one.
    from transformers import AutoModelForMaskedLM
    model, last = None, None
    for cls in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            model = cls.from_pretrained(args.model, dtype=dtype,
                                        trust_remote_code=True).to(device)
            print(f"loaded with {cls.__name__}")
            break
        except Exception as e:
            last = e
    if model is None:
        raise RuntimeError(f"could not load {args.model}: {type(last).__name__}: {last}")
    model.eval()

    # same text for every model: the wikitext-103 validation stream
    val = np.load(os.path.join(args.data, "val.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - args.seq_len - 1, size=args.n_seq)
    # our stream is GPT-2 BPE; re-decode and re-encode so each model sees text
    from transformers import AutoTokenizer as AT
    gpt2 = AT.from_pretrained("gpt2")
    texts = [gpt2.decode(val[s:s + args.seq_len].tolist()) for s in starts]

    if args.mask_token is not None:
        mask_id = int(args.mask_token) if str(args.mask_token).isdigit() else \
            tok.convert_tokens_to_ids(args.mask_token)
    else:
        mask_id = tok.mask_token_id if tok.mask_token_id is not None else 126336

    agg, argmax_all, nlayers = {}, {}, None
    for i in range(0, len(texts), args.batch):
        enc = tok(texts[i:i + args.batch], return_tensors="pt",
                  truncation=True, max_length=args.seq_len,
                  padding="max_length")
        ids = enc["input_ids"].to(device)
        if args.diffusion:
            m = torch.rand(ids.shape, device=device) < args.mask_frac
            ids = torch.where(m, torch.full_like(ids, mask_id), ids)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states
        nlayers = len(hs)
        for l, h in enumerate(hs):
            st = layer_stats(h, args.mult)
            argmax_all.setdefault(l, []).append(st.pop("argmax"))
            for k, v in st.items():
                agg.setdefault(l, {}).setdefault(k, []).append(v)

    res = {"model": args.model, "diffusion_input": args.diffusion,
           "seq_len": args.seq_len, "n_seq": args.n_seq, "mult": args.mult,
           "layers": {}}
    for l in range(nlayers):
        res["layers"][l] = {k: float(np.mean(v)) for k, v in agg[l].items()}
        res["layers"][l].update(position_entropy(argmax_all[l], args.seq_len))

    os.makedirs(args.out, exist_ok=True)
    name = args.model.replace("/", "__") + ("_diff" if args.diffusion else "_clean")
    path = os.path.join(args.out, name + ".json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {args.model} ({'masked' if args.diffusion else 'clean'} input) ===")
    print("layer    mean      max     p99.9   outlier_frac   pos_entropy  frac_pos0")
    for l in range(0, nlayers, max(1, nlayers // 12)):
        e = res["layers"][l]
        print(f"{l:<6} {e['mean']:>8.1f} {e['max']:>9.1f} {e['p999']:>9.1f} "
              f"{e['outlier_frac']:>13.4f} {e['normalised_entropy']:>13.4f} "
              f"{e['frac_pos0']:>10.4f}")
    if args.channels:
        cs = channel_stats(model, tok, texts, args.seq_len, mask_id,
                           args.mask_frac, args.batch, device, args.diffusion)
        res["channels"] = cs
        r = [v["max_over_median"] for v in cs.values()]
        print(f"\n  PER-CHANNEL max/median: mean {np.mean(r):.2f}  "
              f"median {np.median(r):.2f}  worst {np.max(r):.2f}  "
              f"({len(cs)} projections)")
        print(f"  projections with any channel >10x median: "
              f"{sum(1 for v in cs.values() if v['n_over_10x_median'] > 0)}/{len(cs)}")
        print("  (our from-scratch 51M runs: mean 4.13, worst 9.44)")

    if args.ptq:
        res["ptq"] = {}
        fp = masked_ce(model, tok, texts, args.seq_len, mask_id,
                       args.mask_frac, args.batch, device)
        res["ptq"]["fp_ce"] = fp
        print(f"\n  full-precision masked CE {fp:.4f}")
        for wb, ab in ((8, 8), (8, 6)):
            qs = swap_linears(model, wb, ab)
            if len(qs) < nlayers:
                raise RuntimeError(
                    f"only {len(qs)} projections wrapped for {args.model} "
                    f"({nlayers} hidden states) -- the module types are not "
                    "recognised, and reporting this as PTQ degradation would "
                    "read as 'quantization is free'")
            for q in qs:
                q.calibrating = True
            masked_ce(model, tok, texts[:args.batch * 2], args.seq_len, mask_id,
                      args.mask_frac, args.batch, device)
            for q in qs:
                q.calibrating = False
            ce = masked_ce(model, tok, texts, args.seq_len, mask_id,
                           args.mask_frac, args.batch, device)
            res["ptq"][f"W{wb}A{ab}"] = {"ce": ce, "degradation": ce - fp}
            print(f"  W{wb}A{ab} masked CE {ce:.4f}  degradation {ce - fp:+.4f}")
            for q in qs:  # restore fp weights before the next setting
                q._wq = None
            del qs
            import gc; gc.collect(); torch.cuda.empty_cache()

    mx = max(res["layers"][l]["max"] / max(res["layers"][l]["mean"], 1e-6)
             for l in range(nlayers))
    print(f"\npeak max/mean norm ratio across layers: {mx:.1f}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
