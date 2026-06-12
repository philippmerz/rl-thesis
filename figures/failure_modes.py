"""Episode-outcome and movement-composition quantification per ablation cell.

Reads per-episode CSVs produced by ``figures.dump_per_episode`` and
separates two orthogonal views of the 100 evaluation episodes:

Outcome (one mutually exclusive label per episode, parameter-free):
  1. ``cap_reached``       -- not terminated by death.
  2. ``enemy_suicide``     -- died to enemy contact after closing distance to
                              the enemy on net over the final encounter (the
                              trailing run of enemy-visible ticks), measured
                              by the agent's own moves only.
  3. ``failure_to_escape`` -- died to enemy contact while retreating or
                              stationary on net over the final encounter.
  4. ``starvation``        -- terminal damage was starvation.

Movement composition (per-tick fractions, averaged over episodes):
  stationary / oscillating (move-and-undo) / other movement, from the
  position-trace counts in the per-episode CSVs.

The outcome fractions are shown as a stacked bar chart alongside shelter
occupancy; the movement composition is written to a summary CSV for the
thesis table.

Run from repository root: python -m figures.failure_modes
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures.common import REPO_ROOT, save_figure, setup_style


PER_EPISODE_DIR = REPO_ROOT / "eval_logs" / "per_episode"
CELLS = [
    ("baseline",              "Baseline (sf)"),
    ("baseline_fs",           "Baseline (fs)"),
    ("absolute_proximity",    "Abs. prox. (sf)"),
    ("absolute_proximity_fs", "Abs. prox. (fs)"),
    ("minimal_cap50k",              "Minimal (sf)"),
    ("minimal_fs_cap50k",           "Minimal (fs)"),
]
HEURISTIC_LABEL = "Heuristic"
SEEDS = [42, 43, 44, 45]
CATEGORIES = ["cap_reached", "enemy_suicide", "failure_to_escape", "starvation"]
CAT_LABELS = {
    "cap_reached":       "Time-cap reached",
    "enemy_suicide":     "Enemy suicide",
    "failure_to_escape": "Failure to escape",
    "starvation":        "Starvation",
}
CAT_COLORS = {
    "cap_reached":       "#117733",
    "enemy_suicide":     "#cc6677",
    "failure_to_escape": "#ddcc77",
    "starvation":        "#88ccee",
}
MOVEMENT_FIELDS = ["stationary", "oscillating", "other_movement"]


def classify(row: dict) -> str:
    if int(row["terminated"]) == 0:
        return "cap_reached"
    if row["death_cause"] == "starvation":
        return "starvation"
    return "enemy_suicide" if int(row["final_enc_net_delta"]) < 0 else "failure_to_escape"


def movement_fractions(row: dict) -> dict[str, float]:
    survival = max(int(row["survival"]), 1)
    stationary = int(row["stationary_ticks"]) / survival
    osc = int(row["osc_ticks"]) / survival
    return {
        "stationary": stationary,
        "oscillating": osc,
        "other_movement": max(1.0 - stationary - osc, 0.0),
    }


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def cell_fractions(episodes: list[dict]) -> dict[str, float]:
    counts = {cat: 0 for cat in CATEGORIES}
    for ep in episodes:
        counts[classify(ep)] += 1
    n = len(episodes)
    return {cat: counts[cat] / n for cat in CATEGORIES} if n else {}


def cell_movement(episodes: list[dict]) -> dict[str, float]:
    if not episodes:
        return {}
    per_ep = [movement_fractions(ep) for ep in episodes]
    return {f: float(np.mean([p[f] for p in per_ep])) for f in MOVEMENT_FIELDS}


def shelter_use(episodes: list[dict]) -> dict[str, float]:
    ratios = [int(ep["shelter_ticks"]) / surv
              for ep in episodes if (surv := int(ep["survival"])) > 0]
    return {"shelter_occupancy": float(np.mean(ratios))} if ratios else {}


def seed_paths(config: str) -> list[Path]:
    return [p for s in SEEDS if (p := PER_EPISODE_DIR / f"{config}_seed_{s}.csv").exists()]


def aggregate(config: str, per_file_stat) -> dict[str, float]:
    """Mean of a per-file statistic dict across the cell's seeds."""
    stats = [per_file_stat(load_episodes(p)) for p in seed_paths(config)]
    stats = [s for s in stats if s]
    if not stats:
        return {}
    return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}


def heuristic_episodes() -> list[dict]:
    path = PER_EPISODE_DIR / "heuristic.csv"
    return load_episodes(path) if path.exists() else []


def make_figure() -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Panel 1: stacked bar chart of episode outcomes
    ax = axes[0]
    labels = [HEURISTIC_LABEL] + [c[1] for c in CELLS]
    rows = [cell_fractions(heuristic_episodes())]
    rows += [aggregate(cfg, cell_fractions) for cfg, _ in CELLS]

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for cat in CATEGORIES:
        vals = np.array([r.get(cat, 0.0) for r in rows])
        ax.bar(x, vals, bottom=bottom, label=CAT_LABELS[cat],
               color=CAT_COLORS[cat], edgecolor="black", linewidth=0.4,
               width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of evaluation episodes")
    ax.set_ylim(0, 1.0)
    ax.set_title("Episode outcomes over $100$ episodes per seed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              frameon=False, fontsize=9)

    # Panel 2: shelter occupancy fraction per cell
    ax = axes[1]
    sh_vals = [shelter_use(heuristic_episodes()).get("shelter_occupancy", 0.0)]
    sh_vals += [aggregate(cfg, shelter_use).get("shelter_occupancy", 0.0) for cfg, _ in CELLS]
    bar_colors = ["#444444"] + ["#4477aa" if "(sf)" in c[1] else "#ee6677" for c in CELLS]
    ax.bar(np.arange(len(labels)), sh_vals, color=bar_colors,
           edgecolor="black", linewidth=0.4, width=0.7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Shelter occupancy (fraction of survived ticks)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Shelter use")

    return fig


def write_summaries() -> list[Path]:
    """Dump per-cell outcome fractions (plus shelter occupancy) and
    movement composition to CSVs for thesis-table use."""
    heuristic = heuristic_episodes()
    rows_outcome = [{"cell": "heuristic",
                     **cell_fractions(heuristic), **shelter_use(heuristic)}]
    rows_movement = [{"cell": "heuristic", **cell_movement(heuristic)}]
    for cfg, _ in CELLS:
        rows_outcome.append({"cell": cfg, **aggregate(cfg, cell_fractions),
                             **aggregate(cfg, shelter_use)})
        rows_movement.append({"cell": cfg, **aggregate(cfg, cell_movement)})

    outcome_path = PER_EPISODE_DIR / "outcome_summary.csv"
    movement_path = PER_EPISODE_DIR / "movement_summary.csv"
    for path, fields, rows in [
        (outcome_path, ["cell"] + CATEGORIES + ["shelter_occupancy"], rows_outcome),
        (movement_path, ["cell"] + MOVEMENT_FIELDS, rows_movement),
    ]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return [outcome_path, movement_path]


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "failure_modes")
    print(f"Wrote {path}")
    for summary in write_summaries():
        print(f"Wrote {summary}")
