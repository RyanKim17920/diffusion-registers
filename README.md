# diffusion-registers

**Do loss-free register tokens improve masked diffusion language models?**
Across 59 reported training runs and two register designs: **no.**

Register tokens are extra learned vectors appended to the sequence that carry
no prediction target and receive no direct loss — they influence the model only
by being attended to. They help vision transformers by absorbing high-norm
"artifact" activations. This tests the same idea in a masked discrete diffusion
LM, on synthetic Sudoku (where correctness is unambiguous) and wikitext-103.

Every arm shares architecture, data, optimizer and step count; only `K` (the
number of registers) and the seed vary. Runs sharing a seed see byte-identical
data order and masks, so all comparisons are paired by seed.

## Sudoku — test exact-solve accuracy

3 seeds per arm, except carried K=0 and K=64 which have 8.

| K | 0 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| **stateless** — registers reset every forward pass | .9715 | .9700 | .9719 | .9719 | .9680 |
| **carried** — state persists across a generation block | .1440 | .1556 | .1499 | .1436 | .1551 |

No `K` separates from `K=0`. The largest paired gap is carried K=64 at
**+0.011 ± 0.024 (t=1.31, n=8, p≈0.23)**; at n=3 that same arm looked like
+0.012 ± 0.007 (p≈0.09) and dissolved when seeds were added.

The carried registers are genuinely *load-bearing* — severing the carry at
decode time drops exact-solve to ≈0 (.0000 for K=1–16, .0017 for K=64) — and
still never beat a model trained without them. **Necessity is not usefulness**;
an ablation-only study here would have concluded the opposite.

## Text — validation denoising CE, 51M params, wikitext-103

| comparison | result |
|---|---|
| K=16 vs K=0, n=8 paired seeds | −0.0049 nats, t=−1.24, **p≈0.22** |
| the same at n=3 | −0.0086, t=−4.76 — dissolved with more seeds |
| compute-matched: +6% wall clock spent on extra steps instead | registers 0.0061 nats worse (n=3, 2/3 seeds — provisional) |

## Why it fails

Registers do absorb activation outliers: per-**token** max norm falls 40%. But
per-tensor quantization fails on per-**channel** outliers, and there the
reduction is only 4.5%, against a baseline ratio of **4.13×**. Low-bit
quantization breaks in the 100–1000× regime
([arXiv:2508.14896](https://arxiv.org/abs/2508.14896)); GPT-2 (124M) sits at
8.75×. **These models never had the pathology registers are meant to fix**,
which is consistent with the Sudoku null too.

The project stopped at that gate. `src/quant_eval.py` ships and implements the
W4A4 + SmoothQuant sweep, **but that sweep was never run** — the thesis died on
per-channel statistics before quantization was attempted.

The reusable takeaway is a cheap pre-check: measure your baseline's outlier
severity at `K=0` first. It is consistent with both nulls here, though it was
formulated after seeing them and has not been tested prospectively.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python src/gen_sudoku.py --out-dir data/sudoku          # ~90 min, 64 procs
python src/verify_sudoku.py --data data/sudoku          # independent checks
scripts/launch.sh --ks "0 4 16" --seeds "0 1 2" --tag phase1 > jobs.sh
bash jobs.sh                                            # or your scheduler
python src/compare.py --glob 'phase1_*'
```

Text runs additionally need `python src/text_data.py --out data/text`. Paths
default to the repo root; override with `REG_RUNS`, `REG_SUDOKU_DATA`,
`REG_TEXT_DATA`. Run artifacts are not committed, so the numbers above are not
independently checkable from this repo alone.

Full write-up in [RESULTS.md](RESULTS.md); scope and stop criteria in
[PLAN.md](PLAN.md).
