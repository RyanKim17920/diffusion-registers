#!/bin/bash
#SBATCH --job-name=registers
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/data/ryan.kim/registers_runs/slurm/%x_%A_%a.out
#SBATCH --error=/data/ryan.kim/registers_runs/slurm/%x_%A_%a.out
set -euo pipefail

# One array task per line of the jobs file. Each line is the argument string
# passed verbatim to src/train.py.
#   sbatch --array=0-2 scripts/train.sbatch /path/to/jobs.txt [train_block.py]
REPO=/admin/home/ryan.kim/registers
JOBS=${1:?usage: sbatch --array=0-N scripts/train.sbatch <jobs.txt> [script.py]}
SCRIPT=${2:-train.py}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")
if [ -z "$LINE" ]; then echo "no job at index $SLURM_ARRAY_TASK_ID in $JOBS"; exit 1; fi

echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "script: $SCRIPT"
echo "args: $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
# shellcheck disable=SC2086
exec "$REPO/.venv/bin/python" "$REPO/src/$SCRIPT" $LINE
