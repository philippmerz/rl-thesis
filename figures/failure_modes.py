"""Failure-mode quantification per ablation cell.

Reads per-episode CSVs produced by ``figures.dump_per_episode`` and
classifies each of the 100 evaluation episodes into one of four
behavioural categories. The categories are mutually exclusive and
cover all episodes; their fractions per cell are shown as a stacked
bar chart and tabulated in the thesis.

Categorisation (priority order: top match wins):
  1. ``cap_reached``      -- not terminated by death (survival reaches the 1,000-tick cap).
  2. ``enemy_suicide``    -- terminated, survival < 350, ate <= 1 food. Short, lethal episodes
                              before starvation could plausibly be the dominant cause of death.
  3. ``foraging_dying``   -- terminated, ate >= 2 food. Engaged with food but did not survive.
  4. ``camping``          -- terminated, ate <= 1 food, survival >= 350, shelter_ticks /
                              survival >= 0.5. Long-lived passive behaviour while protected by
                              a shelter -- the heuristic's signature mode.
  5. ``stasis``           -- terminated, ate <= 1 food, survival >= 350, shelter occupancy
                              < 0.5. Passive-but-exposed: low engagement with food, no shelter
                              use; consistent with the oscillation / hovering failure modes
                              described in Section~\ref{sec:failure_modes}.

Run from repository root: python -m figures.failure_modes
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures.common import REPO_ROOT, save_figure, setup_style


PER_EPISODE_DIR = REPO_ROOT / "vast_logs" / "per_episode"
CELLS = [
    ("baseline",              "Baseline (sf)"),
    ("baseline_fs",           "Baseline (fs)"),
    ("absolute_proximity",    "Abs. prox. (sf)"),
    ("absolute_proximity_fs", "Abs. prox. (fs)"),
    ("engineered_v5",         "Minimal (sf)"),
    ("engineered_v5_fs",      "Minimal (fs)"),
]
HEURISTIC_LABEL = "Heuristic"
SEEDS = [42, 43, 44, 45]
CATEGORIES = ["cap_reached", "enemy_suicide", "foraging_dying", "camping", "stasis"]
CAT_LABELS = {
    "cap_reached":     "Time-cap (1000 ticks)",
    "enemy_suicide":   "Enemy suicide",
    "foraging_dying":  "Forages, dies early",
    "camping":         "Sheltered passivity",
    "stasis":          "Exposed stasis",
}
CAT_COLORS = {
    "cap_reached":     "#117733",
    "enemy_suicide":   "#cc6677",
    "foraging_dying":  "#ddcc77",
    "camping":         "#999999",
    "stasis":          "#88ccee",
}


def classify(row: dict) -> str:
    survival = int(row["survival"])
    food = int(row["food_eaten"])
    terminated = int(row["terminated"]) == 1
    shelter_occ = int(row["shelter_ticks"]) / max(survival, 1)
    if not terminated:
        return "cap_reached"
    if survival < 350 and food <= 1:
        return "enemy_suicide"
    if food >= 2:
        return "foraging_dying"
    if shelter_occ >= 0.5:
        return "camping"
    return "stasis"


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def cell_fractions(episodes: list[dict]) -> dict[str, float]:
    counts = {cat: 0 for cat in CATEGORIES}
    for ep in episodes:
        counts[classify(ep)] += 1
    n = len(episodes)
    return {cat: counts[cat] / n for cat in CATEGORIES} if n else {cat: 0.0 for cat in CATEGORIES}


def cell_aggregate(config: str) -> tuple[dict[str, float], int]:
    """Mean per-category fraction across the three seeds."""
    fracs = []
    n_seeds = 0
    for seed in SEEDS:
        path = PER_EPISODE_DIR / f"{config}_seed_{seed}.csv"
        if not path.exists():
            continue
        episodes = load_episodes(path)
        if not episodes:
            continue
        fracs.append(cell_fractions(episodes))
        n_seeds += 1
    mean = {cat: float(np.mean([f[cat] for f in fracs])) if fracs else 0.0
            for cat in CATEGORIES}
    return mean, n_seeds


def heuristic_fractions() -> dict[str, float]:
    path = PER_EPISODE_DIR / "heuristic.csv"
    if not path.exists():
        return {cat: 0.0 for cat in CATEGORIES}
    return cell_fractions(load_episodes(path))


def shelter_use(config: str) -> float:
    """Mean shelter_ticks / survival across all episodes for the cell."""
    ratios = []
    for seed in SEEDS:
        path = PER_EPISODE_DIR / f"{config}_seed_{seed}.csv"
        if not path.exists():
            continue
        for ep in load_episodes(path):
            surv = int(ep["survival"])
            if surv > 0:
                ratios.append(int(ep["shelter_ticks"]) / surv)
    return float(np.mean(ratios)) if ratios else 0.0


def heuristic_shelter_use() -> float:
    path = PER_EPISODE_DIR / "heuristic.csv"
    if not path.exists():
        return 0.0
    ratios = []
    for ep in load_episodes(path):
        surv = int(ep["survival"])
        if surv > 0:
            ratios.append(int(ep["shelter_ticks"]) / surv)
    return float(np.mean(ratios)) if ratios else 0.0


def make_figure() -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Panel 1: stacked bar chart of failure mode fractions
    ax = axes[0]
    labels = [HEURISTIC_LABEL] + [c[1] for c in CELLS]
    rows: list[dict[str, float]] = [heuristic_fractions()]
    for cfg, _ in CELLS:
        agg, _ = cell_aggregate(cfg)
        rows.append(agg)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for cat in CATEGORIES:
        vals = np.array([r[cat] for r in rows])
        ax.bar(x, vals, bottom=bottom, label=CAT_LABELS[cat],
               color=CAT_COLORS[cat], edgecolor="black", linewidth=0.4,
               width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of evaluation episodes")
    ax.set_ylim(0, 1.0)
    ax.set_title("Failure-mode breakdown over $100$ episodes per seed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5,
              frameon=False, fontsize=9)

    # Panel 2: shelter occupancy fraction per cell
    ax = axes[1]
    sh_labels = [HEURISTIC_LABEL] + [c[1] for c in CELLS]
    sh_vals = [heuristic_shelter_use()] + [shelter_use(c[0]) for c in CELLS]
    bar_colors = ["#444444"] + ["#4477aa" if "(sf)" in c[1] else "#ee6677" for c in CELLS]
    ax.bar(np.arange(len(sh_labels)), sh_vals, color=bar_colors,
           edgecolor="black", linewidth=0.4, width=0.7)
    ax.set_xticks(np.arange(len(sh_labels)))
    ax.set_xticklabels(sh_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Shelter occupancy (fraction of survived ticks)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Shelter use")

    return fig


def write_table_csv() -> Path:
    """Side-effect: dump a CSV with the per-cell failure-mode fractions
    plus shelter occupancy, for thesis-table use."""
    out = PER_EPISODE_DIR / "failure_mode_summary.csv"
    fields = ["cell"] + CATEGORIES + ["shelter_occupancy"]
    rows = []
    h = heuristic_fractions()
    rows.append({"cell": "heuristic", **h, "shelter_occupancy": heuristic_shelter_use()})
    for cfg, label in CELLS:
        agg, _ = cell_aggregate(cfg)
        rows.append({"cell": cfg, **agg, "shelter_occupancy": shelter_use(cfg)})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "failure_modes")
    summary = write_table_csv()
    print(f"Wrote {path}")
    print(f"Wrote {summary}")
