"""Independent verification of a generated Sudoku dataset.

Deliberately shares no code with src/gen_sudoku.py -- the solution counter
here is written from scratch (bitmask backtracking) so that a bug in the
generator's solver cannot be masked by reusing it.

    python src/verify_sudoku.py --data /data/ryan.kim/registers_data
"""

import argparse
import json
import os

import numpy as np

ROWS = np.arange(81) // 9
COLS = np.arange(81) % 9
BOXES = (ROWS // 3) * 3 + (COLS // 3)


# ------------------------------------------------------- vectorised checks

def _group_is_permutation(grids, group_ids):
    """grids (N, 81) in 1..9; group_ids (81,) in 0..8.
    True per row iff each of the 9 groups holds each digit exactly once."""
    n = len(grids)
    counts = np.zeros((n, 9, 10), dtype=np.int16)
    for g in range(9):
        sel = grids[:, group_ids == g]  # (N, 9)
        for d in range(1, 10):
            counts[:, g, d] = (sel == d).sum(1)
    return (counts[:, :, 1:] == 1).all(axis=(1, 2))


def check_valid_solutions(sol):
    ok = np.ones(len(sol), bool)
    for gid in (ROWS, COLS, BOXES):
        ok &= _group_is_permutation(sol, gid)
    return ok


# -------------------------------------------------- independent solver

def count_solutions(puzzle, limit=2):
    """Count solutions of an 81-length puzzle (0 = blank), stopping at
    `limit`. Bitmask backtracking with most-constrained-cell selection."""
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    blanks = []
    grid = list(int(v) for v in puzzle)
    for i, v in enumerate(grid):
        r, c, b = i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3
        if v:
            bit = 1 << v
            if row[r] & bit or col[c] & bit or box[b] & bit:
                return 0  # the givens already conflict
            row[r] |= bit
            col[c] |= bit
            box[b] |= bit
        else:
            blanks.append(i)

    count = 0

    def rec():
        nonlocal count
        if not blanks:
            count += 1
            return
        # most-constrained blank first
        best_i, best_cands, best_pos = -1, None, -1
        for pos, i in enumerate(blanks):
            r, c, b = i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3
            used = row[r] | col[c] | box[b]
            cands = [d for d in range(1, 10) if not used & (1 << d)]
            if best_cands is None or len(cands) < len(best_cands):
                best_i, best_cands, best_pos = i, cands, pos
                if len(cands) <= 1:
                    break
        if not best_cands:
            return
        i, pos = best_i, best_pos
        r, c, b = i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3
        blanks.pop(pos)
        for d in best_cands:
            bit = 1 << d
            row[r] |= bit
            col[c] |= bit
            box[b] |= bit
            rec()
            row[r] &= ~bit
            col[c] &= ~bit
            box[b] &= ~bit
            if count >= limit:
                break
        blanks.insert(pos, i)

    rec()
    return count


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/ryan.kim/registers_data")
    ap.add_argument("--min_clues", type=int, default=24)
    ap.add_argument("--max_clues", type=int, default=45)
    ap.add_argument("--uniq_per_split", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = ["train", "val", "test"]
    data = {}
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    for s in splits:
        d = os.path.join(args.data, s)
        p = np.load(os.path.join(d, "puzzles.npy"))
        sol = np.load(os.path.join(d, "solutions.npy"))
        nc = np.load(os.path.join(d, "n_clues.npy"))
        data[s] = (p, sol, nc)
        print(f"\n=== {s}: {len(p):,} rows ===")
        check(f"{s} dtypes uint8",
              p.dtype == np.uint8 and sol.dtype == np.uint8 and nc.dtype == np.uint8,
              f"{p.dtype}/{sol.dtype}/{nc.dtype}")
        check(f"{s} shapes",
              p.shape == sol.shape == (len(p), 81) and nc.shape == (len(p),),
              f"{p.shape} {sol.shape} {nc.shape}")
        check(f"{s} puzzle values in 0..9",
              bool(p.min() >= 0 and p.max() <= 9), f"[{p.min()}, {p.max()}]")
        check(f"{s} solution values in 1..9",
              bool(sol.min() >= 1 and sol.max() <= 9), f"[{sol.min()}, {sol.max()}]")

        valid = check_valid_solutions(sol.astype(np.int16))
        check(f"{s} all solutions are valid complete grids",
              bool(valid.all()), f"{int((~valid).sum())} invalid")

        agree = ((p == 0) | (p == sol)).all(1)
        check(f"{s} puzzles agree with solutions on every given",
              bool(agree.all()), f"{int((~agree).sum())} mismatched")

        real_nc = (p != 0).sum(1)
        check(f"{s} n_clues matches nonzero count",
              bool((real_nc == nc).all()))
        in_range = (nc >= args.min_clues) & (nc <= args.max_clues)
        check(f"{s} clue counts in [{args.min_clues}, {args.max_clues}]",
              bool(in_range.all()), f"observed [{nc.min()}, {nc.max()}]")

        keys = np.ascontiguousarray(sol).view([("", np.uint8)] * 81).ravel()
        n_uniq = len(np.unique(keys))
        check(f"{s} no duplicate solution grid within split",
              n_uniq == len(sol), f"{len(sol) - n_uniq} duplicates")

    # cross-split disjointness
    print("\n=== cross-split ===")
    keysets = {}
    for s in splits:
        sol = data[s][1]
        keysets[s] = set(map(bytes, np.ascontiguousarray(sol)))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = keysets[a] & keysets[b]
        check(f"{a} and {b} share no solution grid", not inter, f"{len(inter)} shared")

    # independent uniqueness spot-check
    print("\n=== uniqueness spot-check (independent solver) ===")
    for s in splits:
        p, sol, _ = data[s]
        idx = rng.choice(len(p), size=min(args.uniq_per_split, len(p)), replace=False)
        bad, wrong_sol = 0, 0
        for i in idx:
            n = count_solutions(p[i].tolist(), limit=2)
            if n != 1:
                bad += 1
            if count_solutions(sol[i].tolist(), limit=2) != 1:
                wrong_sol += 1
        check(f"{s}: {len(idx)} sampled puzzles have exactly 1 solution",
              bad == 0, f"{bad} non-unique")
        check(f"{s}: sampled solutions are self-consistent", wrong_sol == 0)

    # histograms
    print("\n=== clue-count histograms ===")
    for s in splits:
        nc = data[s][2]
        vals, cnts = np.unique(nc, return_counts=True)
        print(f"  {s}: " + " ".join(f"{v}:{c}" for v, c in zip(vals, cnts)))

    meta_p = os.path.join(args.data, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            print("\nmeta.json counts:", json.load(f).get("counts"))

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
