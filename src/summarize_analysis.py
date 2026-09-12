"""Aggregate the per-run analysis_*.json files into the register report.

Answers the three phase-2 questions across seeds:
  1. Do the registers carry anything? (zero / shuffle ablation vs normal)
  2. Does the text stream read them? (text->register attention vs the
     uniform-attention baseline K/(163+K))
  3. Do they accumulate activation norm anyway?

    python src/summarize_analysis.py --runs runs
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
    ap.add_argument("--pattern", default="phase1_k*_s*/analysis_val.json")
    args = ap.parse_args()

    by_k = {}
    for p in sorted(glob.glob(os.path.join(args.runs, args.pattern))):
        with open(p) as f:
            a = json.load(f)
        by_k.setdefault(a["k"], []).append(a)
    if not by_k:
        print(f"no analysis files matching {args.pattern} under {args.runs}")
        return

    print("=== 1. Causal importance: exact-solve accuracy under register ablation ===")
    print(f"{'K':>4} {'seeds':>6}  {'normal':>17}  {'zero':>17}  {'shuffle':>17}  "
          f"{'zero - normal':>17}")
    for k in sorted(by_k):
        rs = by_k[k]
        got = {m: [r["ablations"][m]["exact_solve_acc"] for r in rs
                   if m in r["ablations"]] for m in ("normal", "zero", "shuffle")}
        d_zero = [z - n for z, n in zip(got["zero"], got["normal"])]
        print(f"{k:>4} {len(rs):>6}  {cell(*ms(got['normal'])):>17}  "
              f"{cell(*ms(got['zero'])):>17}  {cell(*ms(got['shuffle'])):>17}  "
              f"{cell(*ms(d_zero)):>17}")
    print("  (a register that carries information should make 'zero' collapse;")
    print("   'zero - normal' near 0 means the registers are causally inert)")

    for state in ("all_masked", "half_revealed"):
        print(f"\n=== 2. Attention mass, state = {state} ===")
        print(f"{'K':>4} {'uniform':>9}  " +
              "  ".join(f"{'L' + str(i):>7}" for i in range(8)))
        for k in sorted(by_k):
            rs = [r for r in by_k[k] if state in r.get("states", {})]
            if not rs or "attention" not in rs[0]["states"][state]:
                continue
            base = rs[0]["states"][state]["attention"]["uniform_baseline"]
            nl = len(rs[0]["states"][state]["attention"]["text_to_register"])
            t2r = [ms([r["states"][state]["attention"]["text_to_register"][l]
                       for r in rs])[0] for l in range(nl)]
            r2t = [ms([r["states"][state]["attention"]["register_to_text"][l]
                       for r in rs])[0] for l in range(nl)]
            print(f"{k:>4} {base:>9.4f}  text->reg  " +
                  "  ".join(f"{v:>7.4f}" for v in t2r))
            print(f"{'':>4} {'':>9}  reg->text  " +
                  "  ".join(f"{v:>7.4f}" for v in r2t))
        print("  (text->reg below the uniform baseline = the text stream is actively")
        print("   ignoring the registers)")

    print("\n=== 3. Activation norms (state = all_masked, mean L2 by layer) ===")
    print("  K=0 is included: it is the baseline the register models are judged")
    print("  against. The register hypothesis predicts that adding registers")
    print("  DRAINS high-norm artifacts out of the text tokens, so K>0 should")
    print("  show lower token_max / outlier_frac than K=0.")
    keys = [("register_norm", "reg ", "{:>7.1f}"),
            ("token_norm", "tok ", "{:>7.1f}"),
            ("token_norm_max", "tmax", "{:>7.1f}"),
            ("token_outlier_frac", "out%", "{:>7.4f}")]
    for k in sorted(by_k):
        rs = [r for r in by_k[k] if "all_masked" in r.get("states", {})]
        if not rs:
            continue
        nrm = rs[0]["states"]["all_masked"]["norms"]
        nl = len(nrm["token_norm"])
        print(f"  K={k}")
        vals = {}
        for key, lbl, fmtstr in keys:
            if key not in nrm:
                continue
            v = [ms([r["states"]["all_masked"]["norms"][key][l]
                     for r in rs])[0] for l in range(nl)]
            vals[key] = v
            print(f"    {lbl}  " + "  ".join(fmtstr.format(x) for x in v))
        if "register_norm" in vals and k:
            print("    ratio " + "  ".join(
                f"{r / t:>7.2f}" for r, t in zip(vals["register_norm"],
                                                 vals["token_norm"])))

    print("\n=== 4. Ablation effect on hard puzzles (<=27 clues) ===")
    print(f"{'K':>4}  {'normal':>17}  {'zero':>17}")
    for k in sorted(by_k):
        rs = by_k[k]

        def hard(r, mode):
            b = r["ablations"].get(mode, {}).get("by_clues")
            if not b:
                return None
            num = den = 0
            for c, v in b.items():
                if int(c) <= 27:
                    num += v["exact_solve_acc"] * v["n"]
                    den += v["n"]
            return num / den if den else None

        print(f"{k:>4}  {cell(*ms([hard(r, 'normal') for r in rs])):>17}  "
              f"{cell(*ms([hard(r, 'zero') for r in rs])):>17}")


if __name__ == "__main__":
    main()
