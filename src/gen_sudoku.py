#!/usr/bin/env python3
"""
gen_sudoku.py -- Synthetic Sudoku dataset generator.

Generates random, complete, valid 9x9 Sudoku solution grids via randomized
backtracking, "digs holes" out of each grid to produce a puzzle with a
UNIQUE solution (verified with an exact solver that early-exits as soon as
a second solution is found), and writes train/val/test splits of
(puzzle, solution, n_clues) triples to disk as numpy arrays.

Reproducibility: every worker process derives its RNG stream from
`numpy.random.SeedSequence(master_seed, spawn_key=(job_index,))`, which is
a pure function of (master_seed, job_index) -- independent of which OS
process executes the job or how work is scheduled. Results are drained from
the worker pool via `pool.imap` (ORDERED, not `imap_unordered`), so batches
are consumed in job_index order regardless of which worker happens to
finish first. Combined, this makes the entire pipeline -- which candidate
puzzles are generated, the order they're deduped, and which split each one
lands in -- a pure, deterministic function of (master_seed, workers,
batch_size, min_clues, max_clues): two runs with identical arguments
produce byte-identical output files. Worker COUNT is allowed to change the
output (a different `workers` value changes nothing about per-job seeding,
but changes wall-clock scheduling only in principle -- job order is fixed
by `imap` either way, so in this implementation output is actually
independent of `workers` too; only the CLI arguments to `generate_dataset`
determine the result).

Deduplication is on the SOLUTION grid (81-char string of digits 1-9,
row-major index r*9+c): no solution may appear twice anywhere across
train/val/test combined. Dedup happens in the parent process against a
single global set, since workers run independently and can occasionally
emit colliding solutions.

Usage:
    python gen_sudoku.py --master-seed 1234 --workers 64 \\
        --train-n 1000000 --val-n 20000 --test-n 20000 \\
        --out-dir data/sudoku
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

GENERATOR_VERSION = "1.0.0"

FULL_MASK = 0b111111111  # 9 bits, one per digit 1-9

# ----------------------------------------------------------------------------
# Core Sudoku logic (pure Python; bitmask backtracking with MRV heuristic).
# ----------------------------------------------------------------------------


def _box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def generate_full_grid(rng: np.random.Generator) -> list[int]:
    """Generate one random, complete, valid 9x9 Sudoku solution.

    Uses randomized backtracking with a minimum-remaining-values (MRV)
    heuristic: at each step, fill the empty cell with the fewest legal
    candidates first (in a random order), which makes full-grid generation
    for a fully-constrained puzzle like Sudoku essentially backtrack-free in
    practice. Returns a flat 81-length list of ints 1-9, row-major
    (index = r * 9 + c).
    """
    grid = [0] * 81
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9

    def backtrack() -> bool:
        best_idx = -1
        best_count = 10
        best_mask = 0
        for i in range(81):
            if grid[i] != 0:
                continue
            r, c = divmod(i, 9)
            used = rows[r] | cols[c] | boxes[_box_index(r, c)]
            avail = FULL_MASK & ~used
            cnt = bin(avail).count("1")
            if cnt < best_count:
                best_count = cnt
                best_idx = i
                best_mask = avail
                if cnt == 0:
                    return False  # dead end, no need to scan further
        if best_idx == -1:
            return True  # every cell filled -> solved
        if best_mask == 0:
            return False

        r, c = divmod(best_idx, 9)
        b = _box_index(r, c)
        digits = [d for d in range(1, 10) if best_mask & (1 << (d - 1))]
        order = rng.permutation(len(digits))
        for k in order:
            d = digits[int(k)]
            bit = 1 << (d - 1)
            grid[best_idx] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if backtrack():
                return True
            grid[best_idx] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
        return False

    ok = backtrack()
    if not ok:
        # Should never happen for a Sudoku grid started empty, but guard
        # against silent corruption rather than returning a partial grid.
        raise RuntimeError("failed to generate a complete Sudoku solution")
    return grid


def count_solutions(grid: list[int], limit: int = 2) -> int:
    """Exact solution counter with early exit at `limit`.

    `grid` is a flat 81-length list with 0 = blank, 1-9 = filled. Never
    fully enumerates once `limit` solutions have been found. Mutates a
    local copy only.
    """
    work = list(grid)
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for i, v in enumerate(work):
        if v:
            r, c = divmod(i, 9)
            b = _box_index(r, c)
            bit = 1 << (v - 1)
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    count = 0

    def backtrack() -> None:
        nonlocal count
        best_idx = -1
        best_count = 10
        best_mask = 0
        for i in range(81):
            if work[i] != 0:
                continue
            r, c = divmod(i, 9)
            used = rows[r] | cols[c] | boxes[_box_index(r, c)]
            avail = FULL_MASK & ~used
            cnt = bin(avail).count("1")
            if cnt < best_count:
                best_count = cnt
                best_idx = i
                best_mask = avail
                if cnt == 0:
                    return  # dead end
        if best_idx == -1:
            count += 1
            return
        if best_mask == 0:
            return

        r, c = divmod(best_idx, 9)
        b = _box_index(r, c)
        for d in range(1, 10):
            bit = 1 << (d - 1)
            if not (best_mask & bit):
                continue
            work[best_idx] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            backtrack()
            work[best_idx] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
            if count >= limit:
                return

    backtrack()
    return count


def dig_holes(
    solution: list[int], target_clues: int, rng: np.random.Generator
) -> list[int] | None:
    """Remove cells from a complete grid down to `target_clues`, preserving
    a unique solution at every step.

    Attempts to remove cells in a single random order (without replacement).
    A removal is kept only if the resulting puzzle still has exactly one
    solution (checked via `count_solutions(..., limit=2)`, which never
    fully enumerates). Stops as soon as `target_clues` is reached.

    Returns the puzzle (flat 81-length list, 0 = blank) if `target_clues`
    was reached, or None if the digging got stuck above `target_clues`
    (no remaining cell can be removed without breaking uniqueness) --
    callers MUST discard puzzles that return None, never accept a puzzle
    with more clues than the target.
    """
    puzzle = list(solution)
    clues = 81
    order = rng.permutation(81).tolist()

    for pos in order:
        if clues <= target_clues:
            break
        if puzzle[pos] == 0:
            continue
        backup = puzzle[pos]
        puzzle[pos] = 0
        if count_solutions(puzzle, limit=2) == 1:
            clues -= 1
        else:
            puzzle[pos] = backup  # revert: removing this cell breaks uniqueness

    if clues == target_clues:
        return puzzle
    return None  # could not reach target while preserving uniqueness -> discard


def grid_to_str(grid: list[int]) -> str:
    return "".join(str(v) for v in grid)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


def _worker_job(args: tuple[int, int, int, int, int]) -> list[tuple[list[int], list[int]]]:
    """Run one batch job: generate up to `batch_size` valid, unique-solution
    puzzles. Seed is a pure function of (master_seed, job_index), so this is
    reproducible independent of process/worker scheduling.

    Returns a list of (puzzle, solution) flat-81 int lists. May be shorter
    than `batch_size` if some attempts were discarded (digging could not
    reach target clue count).
    """
    job_index, master_seed, batch_size, min_clues, max_clues = args
    seed_seq = np.random.SeedSequence(master_seed, spawn_key=(job_index,))
    rng = np.random.default_rng(seed_seq)

    results: list[tuple[list[int], list[int]]] = []
    for _ in range(batch_size):
        solution = generate_full_grid(rng)
        target = int(rng.integers(min_clues, max_clues + 1))  # inclusive [min, max]
        puzzle = dig_holes(solution, target, rng)
        if puzzle is None:
            continue  # discard: couldn't reach target clue count uniquely
        results.append((puzzle, solution))
    return results


# ----------------------------------------------------------------------------
# Parent: orchestration, dedup, split assignment, I/O
# ----------------------------------------------------------------------------


def _clue_histogram(n_clues: np.ndarray) -> dict[str, int]:
    hist: dict[str, int] = {}
    for v, c in zip(*np.unique(n_clues, return_counts=True)):
        hist[str(int(v))] = int(c)
    return hist


def _write_split(out_dir: str, name: str, puzzles: list[list[int]], solutions: list[list[int]]) -> np.ndarray:
    split_dir = os.path.join(out_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    puzzles_arr = np.asarray(puzzles, dtype=np.uint8).reshape(-1, 81)
    solutions_arr = np.asarray(solutions, dtype=np.uint8).reshape(-1, 81)
    n_clues_arr = np.count_nonzero(puzzles_arr, axis=1).astype(np.uint8)

    np.save(os.path.join(split_dir, "puzzles.npy"), puzzles_arr)
    np.save(os.path.join(split_dir, "solutions.npy"), solutions_arr)
    np.save(os.path.join(split_dir, "n_clues.npy"), n_clues_arr)
    return n_clues_arr


def generate_dataset(
    master_seed: int,
    workers: int,
    split_ns: dict[str, int],
    out_dir: str,
    min_clues: int = 24,
    max_clues: int = 45,
    batch_size: int = 20,
) -> dict:
    """Drive the whole pipeline. `split_ns` must have keys 'train', 'val',
    'test' (order determines fill order: train first, then val, then test).
    Returns the meta dict that was also written to <out_dir>/meta.json.
    """
    start = time.time()
    total_target = sum(split_ns.values())

    seen_solutions: set[str] = set()
    split_order = [k for k in ("train", "val", "test") if k in split_ns]
    split_data: dict[str, dict[str, list]] = {
        name: {"puzzles": [], "solutions": []} for name in split_order
    }

    def total_collected() -> int:
        return sum(len(split_data[name]["puzzles"]) for name in split_order)

    def current_open_split() -> str | None:
        for name in split_order:
            if len(split_data[name]["puzzles"]) < split_ns[name]:
                return name
        return None

    job_indices = itertools.count()

    def job_iterable():
        for j in job_indices:
            yield (j, master_seed, batch_size, min_clues, max_clues)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        # NOTE: `imap` (ORDERED), not `imap_unordered`. Each job's seed is a
        # pure function of job_index (see _worker_job), and job_iterable()
        # emits job_index 0, 1, 2, ... in order -- so imap yields batches in
        # that same deterministic order regardless of which worker process
        # actually completed which job first. That makes dedup + split
        # assignment (which depends on arrival order) a pure function of
        # (master_seed, workers, batch_size), independent of OS scheduling.
        for batch in pool.imap(_worker_job, job_iterable()):
            for puzzle, solution in batch:
                sol_str = grid_to_str(solution)
                if sol_str in seen_solutions:
                    continue  # duplicate solution grid: drop, keep first occurrence
                dest = current_open_split()
                if dest is None:
                    break  # all splits full
                seen_solutions.add(sol_str)
                split_data[dest]["puzzles"].append(puzzle)
                split_data[dest]["solutions"].append(solution)
            if total_collected() >= total_target:
                break
        pool.terminate()
        pool.join()

    os.makedirs(out_dir, exist_ok=True)
    clue_histograms: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    for name in split_order:
        n_clues_arr = _write_split(
            out_dir, name, split_data[name]["puzzles"], split_data[name]["solutions"]
        )
        clue_histograms[name] = _clue_histogram(n_clues_arr)
        counts[name] = int(len(split_data[name]["puzzles"]))

    wall_clock_s = time.time() - start
    meta = {
        "seed": master_seed,
        "counts": counts,
        "clue_count_histogram": clue_histograms,
        "min_clues": min_clues,
        "max_clues": max_clues,
        "generator_version": GENERATOR_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": wall_clock_s,
        "workers": workers,
        "batch_size": batch_size,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a synthetic Sudoku dataset.")
    p.add_argument("--master-seed", type=int, required=True, help="Master RNG seed.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--train-n", type=int, default=1_000_000)
    p.add_argument("--val-n", type=int, default=20_000)
    p.add_argument("--test-n", type=int, default=20_000)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--min-clues", type=int, default=24)
    p.add_argument("--max-clues", type=int, default=45)
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Puzzles attempted per worker job (a job may return fewer due to discards).",
    )
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    split_ns = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    meta = generate_dataset(
        master_seed=args.master_seed,
        workers=args.workers,
        split_ns=split_ns,
        out_dir=args.out_dir,
        min_clues=args.min_clues,
        max_clues=args.max_clues,
        batch_size=args.batch_size,
    )
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
