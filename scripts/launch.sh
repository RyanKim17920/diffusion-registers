#!/bin/bash
# Emit one training command per (K, seed) combination.
#
# Deliberately scheduler-free: it prints commands rather than submitting them,
# so it works with plain bash, xargs, GNU parallel, or any cluster scheduler.
#
#   scripts/launch.sh --ks "0 4 16" --seeds "0 1 2" --tag phase1 | bash
#   scripts/launch.sh --ks "0 16" --seeds "0 1 2" --script text_train.py \
#       --seq_len 1024 --steps 50000 | xargs -P 4 -I{} bash -c '{}'
#
# Every arm gets identical settings; only --k and --seed vary.
set -euo pipefail

REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
PY="${REG_PYTHON:-python}"
RUNS="${REG_RUNS:-$REPO/runs}"

KS="0 4 16"
SEEDS="0"
TAG="run"
SCRIPT="train.py"
EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --ks) KS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) EXTRA="$EXTRA $1"; shift ;;   # forwarded verbatim to the training script
  esac
done

for k in $KS; do
  for s in $SEEDS; do
    echo "PYTHONPATH=$REPO/src $PY $REPO/src/$SCRIPT --k $k --seed $s" \
         "--name ${TAG}_k${k}_s${s} --out $RUNS$EXTRA"
  done
done
