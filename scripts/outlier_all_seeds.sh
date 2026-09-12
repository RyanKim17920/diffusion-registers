#!/bin/bash
# Per-channel outlier gate across ALL seeds, not 3 -- the perplexity effect
# looked significant at n=3 and dissolved at n=8, and this measurement has to
# survive the same test before anything is claimed from it.
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export PYTHONPATH="$REPO/src"
RUNS=""
for s in 0 1 2 3 4 5 6 7; do
  for k in 0 16; do
    d=/data/ryan.kim/registers_runs/text_k${k}_s${s}
    [ -f "$d/ckpt.pt" ] && RUNS="$RUNS text_k${k}_s${s}"
  done
done
echo "runs: $RUNS"
"$REPO/.venv/bin/python" "$REPO/src/channel_outliers.py" --runs $RUNS \
  --batches 8 --bs 8 --out /data/ryan.kim/registers_runs/channel_outliers_n8.json
echo CHANNEL_OUTLIERS_ALL_DONE
