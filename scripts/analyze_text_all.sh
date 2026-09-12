#!/bin/bash
set -uo pipefail
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
RUNS="${1:-${REG_RUNS:-$REPO/runs}}"
export PYTHONPATH="$REPO/src"
for d in "$RUNS"/text_k*_s*; do
  [ -f "$d/final.json" ] || { echo "skip $(basename $d) (unfinished)"; continue; }
  [ -f "$d/analysis_text.json" ] && { echo "skip $(basename $d) (done)"; continue; }
  echo "=== analysing $(basename "$d") ==="
  "$REPO/.venv/bin/python" "$REPO/src/analyze_text.py" --run "$d" \
    --n_stats 64 --val_batches 16 || echo "ANALYSIS_FAILED $d"
done
echo TEXT_ANALYZE_DONE
