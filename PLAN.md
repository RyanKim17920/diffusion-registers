# Plan

## Thesis

Activation outliers in diffusion LMs are currently handled by **post-training
repair** (SmoothQuant, DuQuant, GPTQ, FAIR-Calib — all applied to a frozen
model). We ask whether a **training-time architectural fix** — loss-free
register tokens — stops the outliers forming in the first place, and whether
that is worth the FLOPs.

This is a claim only controlled training can make, so a ≤1B budget is the
natural scope rather than a limitation.

## Positioning

| | covered by | our relation |
| --- | --- | --- |
| dLLMs have activation outliers | [2508.14896](https://arxiv.org/abs/2508.14896) (LLaDA, Dream-7B) | **cite, do not re-measure** |
| PTQ repair methods for dLLMs | same, + [FAIR-Calib](https://arxiv.org/pdf/2606.06547) | baselines to beat / compose with |
| registers as a carry channel for dLLM reasoning | `lbertge/d1-registers` (concurrent, ICLR 2027, LLaDA-8B) | **avoid this framing** |
| registers as outlier prevention | — | **ours** |

## Status

Done:
- Sudoku, 40 runs: registers do nothing, with a mechanistic reason (the
  pathology is absent — no high-norm outlier tokens at K=0). Keep as the
  criterion's falsification test.
- Necessity ≠ usefulness: `no_carry` → 0 accuracy on a model that is
  nevertheless indistinguishable from K=0. Methods contribution.
- Text (51M, wikitext-103): registers halve the outlier fraction, cut max
  token norm 40%, are attended 7–27× above baseline, and give 0.0086 nats
  (n=3, provisional).
- Cost accounting: +6% wall clock, +2% FLOPs; the gap is 1040-token tiling.
- Harnesses: `quant_eval.py` (W8A16→W4A4, SmoothQuant), `analyze_text.py`,
  `profile_real_models.py`.

Running: n=8 seed confirmation, compute-matched control (K=0 @ 53k steps),
scaling rungs at d=256/4L, 768/12L, 1024/16L on `L = d/64`.

## Phase 0 — the gate (do first, ~1 GPU-hr)

**Does the register effect touch the axis that actually breaks quantization?**

We measured **per-token** norms. The PTQ failure mode is **per-channel**
outliers. A model can have well-behaved token norms and one catastrophic
channel. If registers only fix the token axis, they cannot help W4A4 and the
thesis is dead.

- Measure per-channel activation absmax, kurtosis, and the
  max-channel/median-channel ratio at every Linear input, K=0 vs K=16.
- **Gate:** if the per-channel ratio does not drop measurably, stop and
  rewrite the paper around the Sudoku + necessity≠usefulness results instead.

## Phase 1 — does it help PTQ? (~4 GPU-hr)

`quant_eval.py` over the existing text runs, 8 seeds, W8A8 / W8A6 / W4A8 /
W4A4. Report degradation (quantized − FP), paired by seed.

**Gate:** registers must reduce W4A4 degradation. Absolute CE is secondary.

## Phase 2 — against and with the repair baselines (~8 GPU-hr)

Four arms at W4A4: K=0, K=0 + SmoothQuant, K=16, K=16 + SmoothQuant.

Answers the two questions a reviewer will ask: does a training-time fix beat a
post-training one, and do they compose? Add DuQuant-style rotation if Phase 2
is promising — the literature reports it as the strongest repair method.

## Phase 3 — scale (~30 GPU-hr, partly running)

Ladder on `L = d/64`: 7M / 51M / 115M / 300M, K ∈ {0,16}, 3 seeds, fixed data
and steps. Report outlier reduction and W4A4 degradation per scale.

**The question is the trend, not the level:** does the outlier gap widen with
scale? Three points show direction only; 4 points with 3 seeds is the minimum
for a claim, and the effect must exceed seed noise at each rung.

## Phase 4 — retrofit into a pretrained dLLM (~20 GPU-hr)

Only if Phases 0–2 pass. Add K registers to `dllm-hub/Qwen3-0.6B-diffusion-mdlm`
(0.6B, under budget), short continued fine-tune, measure outlier reduction and
W4A4 degradation against the same model fine-tuned without registers.

This is the bridge from "works when trained from scratch at 300M" to "can be
applied to a real pretrained dLLM", without pretraining at 8B.

## Phase 5 — write-up

Sudoku negative → criterion → text positive → scale → retrofit. The negative
results are the control that makes the positive interpretable; they are not
filler.

## Out of scope (and why)

- **8B training.** Not affordable; not needed — the intervention claim lives
  in controlled training, and Phase 4 covers pretrained models.
- **Re-profiling released dLLMs for outliers.** Published already.
- **Carried registers / reasoning.** Concurrent group is there at 8B.
- **Hard Sudoku.** The generator cannot reach low clue counts (0/80 at
  [18,21]); block decoding supplied difficulty instead.

## Standing rules

- Paired-by-seed comparisons; arms sharing a seed see identical data order.
- Never conclude from n=3 (the Sudoku K=64 arm looked significant at n=3 and
  dissolved at n=8).
- Report FLOP-matched and wall-clock-matched cost side by side.
- Every ablation needs a from-scratch baseline, not just the ablated model.
