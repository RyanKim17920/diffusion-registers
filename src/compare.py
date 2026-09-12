"""Compare runs: print a table of final metrics and plot the curves.

    python src/compare.py --runs runs --names k0_s0 k4_s0 k16_s0
    python src/compare.py --runs runs --glob 'k*_s0'
"""

import argparse
import glob as globmod
import json
import os
import paths


def read_run(run_dir):
    out = {"name": os.path.basename(run_dir.rstrip("/")), "dir": run_dir}
    for fn, key in (("config.json", "config"), ("final.json", "final")):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            with open(p) as f:
                out[key] = json.load(f)
    log = []
    p = os.path.join(run_dir, "log.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    out["log"] = log
    out["val"] = [r for r in log if r.get("split") == "val"]
    out["train"] = [r for r in log if r.get("split") == "train"]
    return out


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def table(runs):
    hdr = ["run", "K", "seed", "params", "step", "val CE", "val exact", "test exact",
           "val cell", "test cell"]
    rows = []
    for r in runs:
        cfg, fin = r.get("config", {}), r.get("final")
        last_val = r["val"][-1] if r["val"] else {}
        if fin:
            rows.append([r["name"], cfg.get("k"), cfg.get("seed"),
                         f"{cfg.get('n_params', 0):,}", cfg.get("steps"),
                         fmt(fin.get("val_ce")),
                         fmt(fin["val"]["exact_solve_acc"]),
                         fmt(fin["test"]["exact_solve_acc"]),
                         fmt(fin["val"]["cell_acc"]),
                         fmt(fin["test"]["cell_acc"])])
        else:
            rows.append([r["name"] + " (running)", cfg.get("k"), cfg.get("seed"),
                         f"{cfg.get('n_params', 0):,}", last_val.get("step"),
                         fmt(last_val.get("ce")),
                         fmt(last_val.get("exact_solve_acc")), "-",
                         fmt(last_val.get("cell_acc")), "-"])
    w = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(hdr)]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(row)))


def by_clue_table(runs):
    have = [r for r in runs if r.get("final", {}).get("test", {}).get("by_clues")]
    if not have:
        return
    print("\nTest exact-solve accuracy by clue count")
    clues = sorted({int(c) for r in have for c in r["final"]["test"]["by_clues"]})
    print("clues  " + "  ".join(f"{r['name']:>12}" for r in have) + "      n")
    for c in clues:
        cells, n = [], 0
        for r in have:
            b = r["final"]["test"]["by_clues"].get(str(c)) or \
                r["final"]["test"]["by_clues"].get(c)
            cells.append(f"{b['exact_solve_acc']:>12.4f}" if b else f"{'-':>12}")
            n = b["n"] if b else n
        print(f"{c:<5}  " + "  ".join(cells) + f"  {n:>5}")


def plot(runs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for r in runs:
        lbl = f"{r['name']} (K={r.get('config', {}).get('k')})"
        tr = r["train"]
        if tr:
            axes[0].plot([x["step"] for x in tr], [x["ce"] for x in tr],
                         alpha=0.75, label=lbl)
        v = r["val"]
        if v:
            axes[1].plot([x["step"] for x in v], [x["ce"] for x in v],
                         marker="o", ms=2.5, label=lbl)
            axes[2].plot([x["step"] for x in v],
                         [x["exact_solve_acc"] for x in v],
                         marker="o", ms=2.5, label=lbl)
    axes[0].set_title("train denoising CE (masked positions)")
    axes[1].set_title("val denoising CE (fixed batch)")
    axes[2].set_title("val exact-solve accuracy")
    for ax, yl in zip(axes, ["CE", "CE", "exact-solve acc"]):
        ax.set_xlabel("step")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=paths.RUNS)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    if args.names:
        dirs = [os.path.join(args.runs, n) for n in args.names]
    else:
        pat = args.glob or "*"
        dirs = sorted(d for d in globmod.glob(os.path.join(args.runs, pat))
                      if os.path.isdir(d) and
                      os.path.exists(os.path.join(d, "config.json")))
    runs = [read_run(d) for d in dirs]
    runs.sort(key=lambda r: (r.get("config", {}).get("k", 0),
                             r.get("config", {}).get("seed", 0)))
    if not runs:
        print(f"no runs found under {args.runs}")
        return
    table(runs)
    aggregate(runs)
    by_clue_table(runs)
    png = args.png or os.path.join(args.runs, "curves.png")
    try:
        plot(runs, png)
    except Exception as e:  # plotting is a convenience, never the point
        print(f"(plot skipped: {e})")



# ---------------------------------------------------------------- aggregation

def aggregate(runs, hard_max_clues=27):
    """Group finished runs by K and report mean +/- sd across seeds.

    Exact-solve accuracy saturates on easy puzzles, so the hard-bucket column
    (clue count <= hard_max_clues) and the val CE are the signals that stay
    informative once the headline number approaches 1.0.
    """
    import statistics as st

    by_k = {}
    for r in runs:
        if not r.get("final"):
            continue
        by_k.setdefault(r["config"]["k"], []).append(r)
    if not by_k:
        print("\n(no finished runs to aggregate)")
        return {}

    def hard_acc(fin):
        b = fin["test"].get("by_clues")
        if not b:
            return None
        num = den = 0
        for c, v in b.items():
            if int(c) <= hard_max_clues:
                num += v["exact_solve_acc"] * v["n"]
                den += v["n"]
        return num / den if den else None

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None, None
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

    print(f"\nAggregated across seeds (mean +/- sd); "
          f"'hard' = test puzzles with <= {hard_max_clues} clues")
    print(f"{'K':>4}  {'seeds':>5}  {'val CE':>17}  {'test exact':>17}  "
          f"{'test exact (hard)':>19}")
    out = {}
    for k in sorted(by_k):
        rs = by_k[k]
        ce_m, ce_s = ms([r["final"]["val_ce"] for r in rs])
        ex_m, ex_s = ms([r["final"]["test"]["exact_solve_acc"] for r in rs])
        hd_m, hd_s = ms([hard_acc(r["final"]) for r in rs])
        out[k] = {"n_seeds": len(rs), "val_ce": [ce_m, ce_s],
                  "test_exact": [ex_m, ex_s], "test_exact_hard": [hd_m, hd_s]}
        def c(m, s):
            return "-" if m is None else f"{m:.4f} +/- {s:.4f}"
        print(f"{k:>4}  {len(rs):>5}  {c(ce_m, ce_s):>17}  {c(ex_m, ex_s):>17}  "
              f"{c(hd_m, hd_s):>19}")

    base = out.get(0)
    if base:
        print("\nDelta vs K=0 baseline (positive test-exact delta = registers help):")
        for k in sorted(out):
            if k == 0:
                continue
            d_ce = out[k]["val_ce"][0] - base["val_ce"][0]
            d_ex = out[k]["test_exact"][0] - base["test_exact"][0]
            noise = max(base["test_exact"][1], out[k]["test_exact"][1])
            verdict = ("within seed noise" if abs(d_ex) <= noise
                       else ("BETTER" if d_ex > 0 else "WORSE"))
            print(f"  K={k:<3} val CE {d_ce:+.4f}   test exact {d_ex:+.4f}   "
                  f"(seed sd {noise:.4f}) -> {verdict}")

        # Paired comparison. Arms sharing a seed see identical data order and
        # identical masks/block plans, so the per-seed difference cancels the
        # run-to-run variation that dominates the unpaired sd above. This is
        # the sensitive test; the unpaired table is the conservative one.
        bseed = {r["config"]["seed"]: r for r in by_k[0]}
        print("\nPaired by seed (same data order and plans; K vs K=0 on that seed):")
        print(f"{'K':>4}  {'n':>2}  {'mean d(test exact)':>19}  "
              f"{'mean d(val CE)':>15}   per-seed deltas")
        for k in sorted(by_k):
            if k == 0:
                continue
            dex, dce, detail = [], [], []
            for r in sorted(by_k[k], key=lambda r: r["config"]["seed"]):
                sd_ = r["config"]["seed"]
                if sd_ not in bseed:
                    continue
                b = bseed[sd_]["final"]
                d1 = r["final"]["test"]["exact_solve_acc"] - b["test"]["exact_solve_acc"]
                dex.append(d1)
                dce.append(r["final"]["val_ce"] - b["val_ce"])
                detail.append(f"s{sd_}:{d1:+.4f}")
            if not dex:
                continue
            m_ex, s_ex = ms(dex)
            m_ce, _ = ms(dce)
            # does the paired mean clear its own standard error?
            se = (s_ex / (len(dex) ** 0.5)) if len(dex) > 1 else float("inf")
            mark = "" if se == 0 or abs(m_ex) <= 2 * se else "  <- 2se"
            print(f"{k:>4}  {len(dex):>2}  {m_ex:>+12.4f} +/-{s_ex:>6.4f}  "
                  f"{m_ce:>+15.4f}   {' '.join(detail)}{mark}")
    return out

if __name__ == "__main__":
    main()
