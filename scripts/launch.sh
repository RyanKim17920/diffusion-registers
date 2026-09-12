#!/bin/bash
# Build a jobs file from a K list x seed list and submit it as one array job.
#
#   scripts/launch.sh --ks "0 4 16" --seeds "0 1 2" --tag phase1
#   scripts/launch.sh --ks "0 1 4 16 64" --seeds "0 1 2" --tag ksweep --steps 40000
#
# Every arm gets identical --steps/--bs/--lr/etc; only --k and --seed vary.
set -euo pipefail

KS="0 4 16"
SEEDS="0"
TAG="phase1"
EXTRA=""
RUNS=/data/ryan.kim/registers_runs

while [ $# -gt 0 ]; do
  case "$1" in
    --ks) KS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) EXTRA="$EXTRA $1"; shift ;;   # forwarded verbatim to train.py
  esac
done

REPO=/admin/home/ryan.kim/registers
mkdir -p "$RUNS/slurm" "$RUNS/jobs"
JOBS="$RUNS/jobs/${TAG}.txt"
: > "$JOBS"

n=0
for k in $KS; do
  for s in $SEEDS; do
    echo "--k $k --seed $s --name ${TAG}_k${k}_s${s} --out $RUNS$EXTRA" >> "$JOBS"
    n=$((n + 1))
  done
done

echo "jobs file: $JOBS  ($n tasks)"
cat "$JOBS"
sbatch --array=0-$((n - 1)) --job-name="reg_${TAG}" "$REPO/scripts/train.sbatch" "$JOBS"
