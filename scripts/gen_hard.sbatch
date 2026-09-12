#!/bin/bash
#SBATCH --job-name=gen_sudoku_hard
#SBATCH --partition=c
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/ryan.kim/registers_runs/gen_hard_%j.out
set -euo pipefail
REPO=/admin/home/ryan.kim/registers
exec "$REPO/.venv/bin/python" "$REPO/src/gen_sudoku.py" \
  --master-seed 4321 --workers 64 \
  --train-n 300000 --val-n 10000 --test-n 10000 \
  --min-clues 20 --max-clues 24 \
  --out-dir /data/ryan.kim/registers_data_hard
