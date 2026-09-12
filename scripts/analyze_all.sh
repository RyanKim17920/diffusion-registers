#!/bin/bash
# Run src/analyze.py over every run that has a checkpoint and K>0.
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
RUNS=${1:-/data/ryan.kim/registers_runs}
N_SOLVE=${2:-5000}
export PYTHONPATH="$REPO/src"
for d in "$RUNS"/phase1_k*_s*; do
  [ -f "$d/ckpt.pt" ] || continue
  k=$("$REPO/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['n_registers'])" "$d/model_config.json" 2>/dev/null) || continue
  [ "$k" = "0" ] && continue
  [ -f "$d/analysis_val.json" ] && { echo "skip $(basename $d) (already analysed)"; continue; }
  echo "=== analysing $(basename "$d") (K=$k) ==="
  "$REPO/.venv/bin/python" "$REPO/src/analyze.py" --run "$d" --n_solve "$N_SOLVE" --n_attn 256 || echo "ANALYSIS_FAILED $d"
done
echo ANALYZE_ALL_DONE
