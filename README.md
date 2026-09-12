# diffusion-registers

**Do loss-free register tokens improve masked diffusion language models? No.**

76 runs, two register designs, Sudoku and wikitext-103. Paired by seed: arms
sharing a seed see identical data order and masks.

### Sudoku — test exact-solve accuracy (3 seeds/arm)

| K | 0 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| stateless registers | .9715 | .9700 | .9719 | .9719 | .9680 |
| registers carried across a block | .1440 | .1556 | .1499 | .1436 | .1551 |

No K beats K=0. Carried registers are *load-bearing* — severing the carry gives
**0.0000** — yet never beat a K=0 model trained without them. Necessity is not
usefulness.

### Text — validation denoising CE, 51M params, paired

| comparison | result |
|---|---|
| K=16 vs K=0, n=8 | −0.0049 nats, t=−1.24, **p≈0.22** |
| same at n=3 | −0.0086, t=−4.76 (dissolved with more seeds) |
| compute-matched (+6% wall clock as extra steps) | **+0.0061 nats worse** |

### Why

Registers cut per-**token** max activation norm 40%, but per-**channel**
max/median only 4.5% — and that baseline is **4.2×**, where low-bit
quantization breaks at 100–1000×. GPT-2 (124M): 8.75×. No pathology to fix.

[RESULTS.md](RESULTS.md) · [PLAN.md](PLAN.md) · paths configurable via
`REG_RUNS`, `REG_SUDOKU_DATA`, `REG_TEXT_DATA`.
