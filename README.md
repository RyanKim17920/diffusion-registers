# Loss-free register tokens in a masked diffusion language model

Minimal research prototype testing one question: **do K learned register
tokens, which have no prediction target and receive no direct loss, improve a
masked discrete diffusion Transformer?** The testbed is synthetic Sudoku,
where "did it work" has an unambiguous answer (exact-solve accuracy).

## The setup

Every forward pass operates on

```
[ puzzle: 81 tokens ] [ SEP ] [ solution: 81 tokens ] [ R_0 .. R_{K-1} ]
  0..80                81      82..162                 163..163+K-1
```

with full bidirectional attention. Vocabulary is 12 ids: `0` = blank puzzle
cell, `1..9` = digits, `10` = MASK, `11` = SEP.

Three properties are load-bearing and are enforced in `src/model.py`:

1. **No direct loss.** The output head is applied only to the 81 solution
   positions (`logits` is `(B, 81, 9)`). Registers have no target and appear
   in no loss term; they can only affect the loss by being attended to.
2. **Re-initialised every forward pass.** Registers are K rows of a dedicated
   learned embedding table (`reg_emb`), read fresh on each call to `forward`.
   Nothing is carried between denoising steps: step 40 of a decode starts its
   registers from exactly the same learned vectors as step 0.
3. **K=0 is the same code path.** With `--k 0` there is no register parameter
   at all (`reg_emb is None`) and the sequence is the plain 163 tokens, so the
   baseline is not a register model with registers disabled.

The forward process is absorbing-state masking: draw `t ~ U(0,1]` per example,
mask each solution position independently with probability `t` (at least one
always masked), and take cross-entropy over masked positions only. Puzzle
clues are always visible in the puzzle block; the model must still reproduce
them in the solution block.

Decoding matches the protocol in the brief: start from all 81 solution
positions MASKed, run the model, reveal the single highest-confidence masked
position, repeat for 81 steps.

## Comparison hygiene

The point of this prototype is a comparison that can be trusted, so:

- Data order and mask sampling are driven by RNG streams seeded independently
  of K, so every arm sees **identical puzzles in identical order with
  identical masks**. Only model init consumes the torch RNG, whose draw count
  necessarily differs with K.
- The validation denoising loss is measured on a **fixed batch** (fixed
  puzzle indices, fixed `t`, fixed masks, from an arm- and seed-independent
  seed), so val-CE curves are comparable across arms, seeds and steps.
- Every arm shares architecture, optimizer, schedule, step count and batch
  size. Only `--k` and `--seed` vary.
- Multiple seeds per arm, because a one-seed difference on this kind of
  comparison is not evidence.

## Data

`src/gen_sudoku.py` generates complete grids by randomized backtracking, then
digs holes while verifying with a solution-counting solver that the puzzle
keeps a **unique** solution. Clue counts are drawn uniformly from [24, 45] so
difficulty can be stratified later. Solution grids are deduplicated globally,
so **no solution grid in val/test appears in train under any clue mask** —
the holdout tests generalisation, not puzzle memorisation.

```
/data/ryan.kim/registers_data/{train,val,test}/{puzzles,solutions,n_clues}.npy
```

`puzzles` is uint8 `(N, 81)` with 0 for blanks; `solutions` is uint8 `(N, 81)`
in 1..9; cell (r, c) is index `r*9 + c`. `src/verify_sudoku.py` re-checks
validity, puzzle/solution agreement, split disjointness, and uniqueness
(with an independently written solver).

## Two register designs

`src/train.py` trains the **stateless** design: registers re-read from the
learned embeddings on every forward pass, free-order decoding.

`src/train_block.py` trains the **carried** design: registers persist across the
denoising steps within a generation block and reset at the block boundary, with
semi-autoregressive block decoding and backprop through the carry. See
`src/blockdiff.py`.

Results for both are in [RESULTS.md](RESULTS.md).

## Running

```bash
# phase 1: baseline vs registers
scripts/launch.sh --ks "0 4 16" --seeds "0 1 2" --tag phase1

# compare
.venv/bin/python src/compare.py --glob 'phase1_*'

# phase 2 (only if registers are competitive): attention, norms, ablations
.venv/bin/python src/analyze.py --run /data/ryan.kim/registers_runs/phase1_k4_s0
```

Runs live on `/data/ryan.kim/registers_runs` (`runs/` in the repo is a symlink
there); the repo itself stays in home. The GPU venv is `.venv` — the
system-wide torch is built against CUDA 13 and will not initialise on this
cluster's 12.8 driver.

## Layout

| file | role |
| --- | --- |
| `src/model.py` | transformer + register block, `reg_mode` normal/zero/shuffle |
| `src/diffusion.py` | data loading, masking process, loss, 81-step decoder |
| `src/train.py` | training loop, curves, checkpoints, final val/test metrics |
| `src/gen_sudoku.py` | unique-solution puzzle generator |
| `src/verify_sudoku.py` | independent dataset verification |
| `src/compare.py` | cross-run table, per-clue-count breakdown, curve plots |
| `src/analyze.py` | register attention/norms + causal ablations (stateless) |
| `src/blockdiff.py` | block plan, unrolled per-block loss, block decoder |
| `src/train_block.py` | training for the carried-register design |
| `src/analyze_block.py` | no_carry/zero/shuffle ablations + carry dynamics |
| `src/text_train.py` | text MDLM (wikitext-103), same register mechanism |
| `scripts/launch.sh` | build a jobs file and submit the array |
