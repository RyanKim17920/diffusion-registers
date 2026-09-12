#!/bin/bash
#SBATCH --job-name=gate_cpu
#SBATCH --partition=c
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/data/ryan.kim/registers_runs/gate_cpu_%j.out
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=64
# Forward-only inference: no GPU needed, and the GPU queue is 3h deep with
# training that this measurement may make pointless.
RUNS=""
for s in 0 1 2 3 4 5 6 7; do
  for k in 0 16; do
    [ -f /data/ryan.kim/registers_runs/text_k${k}_s${s}/ckpt.pt ] && \
      RUNS="$RUNS text_k${k}_s${s}"
  done
done
echo "runs: $RUNS"
"$REPO/.venv/bin/python" "$REPO/src/channel_outliers.py" --runs $RUNS \
  --batches 4 --bs 4 --out /data/ryan.kim/registers_runs/channel_outliers_n8.json
echo CHANNEL_OUTLIERS_ALL_DONE
