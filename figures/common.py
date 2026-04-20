"""Shared utilities for thesis figure generation.

Each figure script reads eval.csv files from ``runs/`` and writes a PDF
into ``latex/original/figures/``. Scripts are idempotent; rerunning
regenerates the figures from the source data.

Run from the repository root: ``python -m figures.<figure_name>``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "latex" / "original" / "figures"


# Eval CSV column layout (from training/metrics.py)
EVAL_COLUMNS = [
    "step", "episode", "eval_reward", "eval_survival",
    "eval_food_eaten", "eval_damage_taken", "eval_death_rate",
    "epsilon", "loss", "wall_time",
]


@dataclass(frozen=True)
class Palette:
    heuristic: str = "#555555"
    e5: str = "#1f77b4"
    e5_fs: str = "#ff7f0e"
    v7_fs: str = "#2ca02c"
    v8_cycle: str = "#d62728"
    v8_reset: str = "#9467bd"
    v8_strong: str = "#8c564b"


PALETTE = Palette()


def load_eval(config: str, seed: int) -> pd.DataFrame:
    """Load eval.csv for one (config, seed). Returns empty df if missing."""
    path = RUNS_DIR / config / f"seed_{seed}" / "logs" / "eval.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVAL_COLUMNS)
    return pd.read_csv(path, names=EVAL_COLUMNS, header=0)


def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    """Rolling mean for learning-curve display. Leaves leading NaNs untouched."""
    return series.rolling(window=window, min_periods=1).mean()


def setup_style() -> None:
    """Consistent figure style across all plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save figure as PDF into the thesis figures directory."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def best_seed_for(config: str, seeds: Iterable[int] = (42, 43, 44)) -> int | None:
    """Return the seed whose best eval survival is highest, or None if no data."""
    best_val = -np.inf
    best_seed = None
    for seed in seeds:
        df = load_eval(config, seed)
        if df.empty:
            continue
        peak = df["eval_survival"].max()
        if peak > best_val:
            best_val = peak
            best_seed = seed
    return best_seed
