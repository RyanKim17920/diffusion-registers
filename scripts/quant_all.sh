#!/bin/bash
# PTQ degradation across every finished run: all scaling rungs, all seeds,
# with and without the SmoothQuant repair baseline.
#
# The paper's headline number comes from here:
#   degradation(K=0) - degradation(K=16) at W4A4, paired by seed, per rung.
set -uo pipefail
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
RUNS="${1:-${REG_RUNS:-$REPO/runs}}"
export PYTHONPATH="$REPO/src"
for d in "$RUNS"/text_k*_s*; do
  [ -d "$d" ] || continue
  [ -f "$d/final.json" ] || { echo "skip $(basename $d) (unfinished)"; continue; }
  for sq in 0 0.5; do
    tag=""; [ "$sq" != "0" ] && tag="_sq$sq"
    [ -f "$d/quant_eval${tag}.json" ] && { echo "skip $(basename $d)$tag (done)"; continue; }
    echo "=== $(basename "$d")  smooth=$sq ==="
    "${REG_PYTHON:-python}" "$REPO/src/quant_eval.py" --run "$d" \
      --smooth "$sq" --val_batches 16 --bs 8 || echo "QUANT_FAILED $d sq=$sq"
  done
done
echo QUANT_ALL_DONE
