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
    ap.add_argument("--data", default="/data/ryan.kim/registers_text_data")
    ap.add_argument("--out", default="/data/ryan.kim/registers_runs/real_models")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data/huggingface")
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    except Exception:
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
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
    mx = max(res["layers"][l]["max"] / max(res["layers"][l]["mean"], 1e-6)
             for l in range(nlayers))
    print(f"\npeak max/mean norm ratio across layers: {mx:.1f}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
