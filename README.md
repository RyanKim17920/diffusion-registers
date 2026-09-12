# diffusion-registers

**Do loss-free register tokens improve masked diffusion language models?**
Across 76 training runs and two register designs: **no.**

Register tokens are extra learned vectors appended to the sequence that carry
no prediction target and receive no direct loss — they can only influence the
model by being attended to. They help vision transformers by absorbing
high-norm "artifact" activations. This tests whether the same idea helps a
masked discrete diffusion LM, on synthetic Sudoku (where correctness is
unambiguous) and on wikitext-103.

Every arm shares architecture, data, optimizer and step count; only `K` (the
number of registers) and the seed vary. Runs sharing a seed see byte-identical
data order and masks, so comparisons are paired by seed.

## Sudoku — test exact-solve accuracy, 3 seeds/arm

| K | 0 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| **stateless** — registers reset every forward pass | .9715 | .9700 | .9719 | .9719 | .9680 |
| **carried** — state persists across a generation block | .1440 | .1556 | .1499 | .1436 | .1551 |

No `K` beats `K=0` under either design. The carried registers are genuinely
*load-bearing* — severing the carry at decode time drops exact-solve to
**0.0000** — and still never beat a model trained without them. Necessity is
not usefulness, and an ablation-only study here would have concluded the
opposite.

## Text — validation denoising CE, 51M params, wikitext-103

| comparison | result |
|---|---|
| K=16 vs K=0, n=8 paired seeds | −0.0049 nats, t=−1.24, **p≈0.22** |
| the same at n=3 | −0.0086, t=−4.76 — dissolved with more seeds |
| compute-matched: +6% wall clock spent on extra steps instead | registers **0.0061 nats worse** |

## Why it fails

Registers do absorb activation outliers — per-**token** max norm falls 40%.
But per-tensor quantization fails on per-**channel** outliers, and there the
reduction is only 4.5%, against a baseline ratio of **4.2×**. Low-bit
quantization breaks at 100–1000×; GPT-2 (124M) sits at 8.75×. **These models
never had the pathology registers are meant to fix**, which also explains the
Sudoku null.

The useful takeaway is a cheap pre-check: measure your baseline's outlier
severity at `K=0` first. It correctly predicted the null in both settings here.

## Layout

`src/model.py` registers + transformer · `src/diffusion.py` masking and decoding
· `src/blockdiff.py` carried-register block diffusion · `src/train*.py` training
· `src/analyze*.py`, `src/channel_outliers.py`, `src/quant_eval.py` analysis ·
`src/gen_sudoku.py`, `src/verify_sudoku.py` data + independent verification.

```bash
scripts/launch.sh --ks "0 4 16" --seeds "0 1 2" --tag phase1 | bash
python src/compare.py --glob 'phase1_*'
```

Paths default to the repo root; override with `REG_RUNS`, `REG_SUDOKU_DATA`,
`REG_TEXT_DATA`. Full write-up in [RESULTS.md](RESULTS.md), scope and stop
criteria in [PLAN.md](PLAN.md).
