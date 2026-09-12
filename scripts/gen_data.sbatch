#!/bin/bash
#SBATCH --job-name=gen_sudoku
#SBATCH --partition=c
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/admin/home/ryan.kim/registers/runs/gen_data_%j.out

set -euo pipefail

# No project venv / conda env found on this machine at submission time
# (checked: no registers venv, `conda` not on PATH) -- system python3 has
# numpy already available, so we use it directly per the task instructions.
PYTHON_BIN=python3

echo "host: $(hostname)"
echo "python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -c "import numpy; print('numpy', numpy.__version__)"

cd /admin/home/ryan.kim/registers

"$PYTHON_BIN" src/gen_sudoku.py \
    --master-seed 1234 \
    --workers 64 \
    --train-n 1000000 \
    --val-n 20000 \
    --test-n 20000 \
    --out-dir /data/ryan.kim/registers_data
