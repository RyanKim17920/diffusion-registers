"""Aggregate the block-diffusion analysis_block.json files across seeds.

    python src/summarize_block_analysis.py --runs runs
"""

import argparse
import glob
import json
import os
import statistics as st
import paths


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def cell(m, s, nd=4):
    return "-" if m is None else f"{m:.{nd}f} +/- {s:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=paths.RUNS)
    ap.add_argument("--pattern", default="blk_k*_s*/analysis_block.json")
    args = ap.parse_args()

    by_k = {}
    for p in sorted(glob.glob(os.path.join(args.runs, args.pattern))):
        with open(p) as f:
            a = json.load(f)
        by_k.setdefault(a["k"], []).append(a)
    if not by_k:
        print(f"no analysis files under {args.runs}")
        return

    print("=== Decode-time ablations (exact-solve accuracy) ===")
    print("  no_carry = registers re-initialised every step instead of carried;")
    print("  it severs the scratchpad channel and changes nothing else.")
    modes = ["normal", "no_carry", "zero", "shuffle"]
    print(f"{'K':>4} {'seeds':>6}  " + "  ".join(f"{m:>17}" for m in modes))
    for k in sorted(by_k):
        rs = by_k[k]
        cells = []
        for m in modes:
            vals = [r["ablations"][m]["exact_solve_acc"] for r in rs
                    if m in r["ablations"]]
            cells.append(cell(*ms(vals)) if vals else "-")
        print(f"{k:>4} {len(rs):>6}  " + "  ".join(f"{c:>17}" for c in cells))

    print("\n=== Cost of severing the carry (no_carry - normal) ===")
    for k in sorted(by_k):
        rs = [r for r in by_k[k] if "no_carry" in r["ablations"]]
        if not rs:
            continue
        d = [r["ablations"]["no_carry"]["exact_solve_acc"]
             - r["ablations"]["normal"]["exact_solve_acc"] for r in rs]
        print(f"  K={k:<4} {cell(*ms(d))}")

    print("\n=== Carry dynamics (mean over seeds, one block) ===")
    for k in sorted(by_k):
        rs = [r for r in by_k[k] if "carry_dynamics" in r]
        if not rs:
            continue
        n = len(rs[0]["carry_dynamics"]["step_norm"])
        print(f"  K={k}")
        print("    step   |state|   d(prev)   cos(prev)   cos(init)")
        for j in range(n):
            sn = ms([r["carry_dynamics"]["step_norm"][j] for r in rs])[0]
            dp = ms([r["carry_dynamics"]["delta_from_prev"][j] for r in rs])[0]
            cp = ms([r["carry_dynamics"]["cos_to_prev"][j] for r in rs])[0]
            ci = ms([r["carry_dynamics"]["cos_to_init"][j] for r in rs])[0]
            print(f"    {j:<6} {sn:>7.2f}   "
                  f"{'-' if dp is None else f'{dp:7.2f}'}   "
                  f"{'-' if cp is None else f'{cp:9.4f}'}   {ci:>9.4f}")


if __name__ == "__main__":
    main()
