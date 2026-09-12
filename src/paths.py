"""Central path configuration.

Nothing in this repository hardcodes a machine-specific path. Defaults are
relative to the repository root, so a fresh clone works with no configuration;
override any of them with environment variables to put large artifacts on a
scratch filesystem:

    REG_RUNS         run outputs (checkpoints, logs, analyses)
    REG_SUDOKU_DATA  generated Sudoku dataset
    REG_TEXT_DATA    tokenized text corpus

    export REG_RUNS=/scratch/$USER/registers_runs
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _p(env, *default):
    return os.environ.get(env) or str(REPO_ROOT.joinpath(*default))


RUNS = _p("REG_RUNS", "runs")
SUDOKU_DATA = _p("REG_SUDOKU_DATA", "data", "sudoku")
TEXT_DATA = _p("REG_TEXT_DATA", "data", "text")
