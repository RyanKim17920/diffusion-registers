#!/bin/bash
#SBATCH --job-name=gen_text
#SBATCH --partition=c
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/data/ryan.kim/registers_runs/gen_text_%j.out
set -euo pipefail
REPO=/admin/home/ryan.kim/registers
export HF_HOME=/data/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
exec "$REPO/.venv/bin/python" "$REPO/src/text_data.py" \
  --out /data/ryan.kim/registers_text_data --num_proc 32
