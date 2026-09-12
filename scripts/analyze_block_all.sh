#!/bin/bash
# Run src/analyze_block.py over every finished block-diffusion run.
set -uo pipefail
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
RUNS="${1:-${REG_RUNS:-$REPO/runs}}"
N_SOLVE=${2:-2000}
export PYTHONPATH="$REPO/src"
for d in "$RUNS"/blk_k*_s*; do
  [ -f "$d/final.json" ] || { echo "skip $(basename $d) (unfinished)"; continue; }
  [ -f "$d/analysis_block.json" ] && { echo "skip $(basename $d) (done)"; continue; }
  echo "=== analysing $(basename "$d") ==="
  "${REG_PYTHON:-python}" "$REPO/src/analyze_block.py" --run "$d" \
    --n_solve "$N_SOLVE" || echo "ANALYSIS_FAILED $d"
done
echo BLOCK_ANALYZE_DONE
