"""Compare runs: print a table of final metrics and plot the curves.

    python src/compare.py --runs /data/ryan.kim/registers_runs --names k0_s0 k4_s0 k16_s0
    python src/compare.py --runs /data/ryan.kim/registers_runs --glob 'k*_s0'
"""

import argparse
import glob as globmod
import json
import os


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
    ap.add_argument("--runs", default="/data/ryan.kim/registers_runs")
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
    by_clue_table(runs)
    png = args.png or os.path.join(args.runs, "curves.png")
    try:
        plot(runs, png)
    except Exception as e:  # plotting is a convenience, never the point
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
