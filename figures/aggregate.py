"""Aggregate per-episode benchmark results into thesis-ready summaries.

Reads ``eval_logs/per_episode/<config>_seed_<N>.csv`` (one row per
evaluation episode, four seeds per cell) and writes two roll-ups:

  eval_logs/bench_summary.csv      -- per-(config, seed) cell means.
                                      Consumed by figures.ablation_grid.
  eval_logs/cell_paired_tests.csv  -- per-cell paired t-test against the
                                      matched-cap heuristic (thesis Table 4).

Cap pairing: misspecified cells (baseline, absolute_proximity, +fs) compare
to heuristic.csv (cap=1000). Minimal cells (minimal_cap50k, +fs) compare to
heuristic_cap50k.csv (cap=50000).

Run from the repository root: python -m figures.aggregate
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


# REPO_ROOT is duplicated from figures/common.py rather than imported,
# so this script can run without the matplotlib/pandas deps that the
# plotting scripts pull in. Aggregation is stdlib-only.
REPO_ROOT = Path(__file__).resolve().parent.parent

PER_EPISODE = REPO_ROOT / "eval_logs" / "per_episode"
BENCH_SUMMARY = REPO_ROOT / "eval_logs" / "bench_summary.csv"
PAIRED_TESTS = REPO_ROOT / "eval_logs" / "cell_paired_tests.csv"

# (display_name, csv_basename, heuristic_csv_basename).
# The heuristic_csv_basename binds each cell to its matched-cap heuristic.
CELLS = [
    ("baseline",              "baseline",              "heuristic"),
    ("baseline_fs",           "baseline_fs",           "heuristic"),
    ("absolute_proximity",    "absolute_proximity",    "heuristic"),
    ("absolute_proximity_fs", "absolute_proximity_fs", "heuristic"),
    ("minimal",               "minimal_cap50k",        "heuristic_cap50k"),
    ("minimal_fs",            "minimal_fs_cap50k",     "heuristic_cap50k"),
]
SEEDS = (42, 43, 44, 45)


def load_col(path: Path, col: str) -> list[float]:
    with open(path) as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def sd(xs: list[float]) -> float:
    """Sample SD with Bessel's correction."""
    m = statistics.mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_heuristic_survivals() -> dict[str, list[float]]:
    return {
        "heuristic":        load_col(PER_EPISODE / "heuristic.csv", "survival"),
        "heuristic_cap50k": load_col(PER_EPISODE / "heuristic_cap50k.csv", "survival"),
    }


def write_bench_summary(heur_means: dict[str, float]) -> None:
    """Per-(config, seed) survival/food means; one row per training run."""
    rows = []
    for _, csv_base, heur_b in CELLS:
        for seed in SEEDS:
            path = PER_EPISODE / f"{csv_base}_seed_{seed}.csv"
            if not path.exists():
                print(f"  SKIP missing {path}")
                continue
            with open(path) as f:
                eps = list(csv.DictReader(f))
            rows.append({
                "config": csv_base,
                "seed": seed,
                "d_surv": f"{statistics.mean(int(e['survival']) for e in eps):.2f}",
                "d_food": f"{statistics.mean(float(e['food_eaten']) for e in eps):.4f}",
                "h_surv": f"{heur_means[heur_b]:.2f}",
            })
    BENCH_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCH_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "d_surv", "d_food", "h_surv"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {BENCH_SUMMARY}  ({len(rows)} rows)")


def cell_paired_test(csv_base: str, heur_surv: list[float]) -> dict | None:
    """Paired t-test of per-world-averaged DQN survival against heuristic.

    For each of the 100 worlds, average DQN survival across the 4 seeds
    to get one paired value, then run the paired test on those 100
    values. n=100 is the correct effective sample size: the 4 seeds on
    the same world are not independent draws and shouldn't be pooled
    into n=400.
    """
    surv = {}
    for s in SEEDS:
        p = PER_EPISODE / f"{csv_base}_seed_{s}.csv"
        if p.exists():
            surv[s] = load_col(p, "survival")
    if not surv:
        return None
    seeds = sorted(surv)
    n_worlds = len(heur_surv)

    surv_per_seed = [statistics.mean(surv[s]) for s in seeds]
    per_world_mean = [statistics.mean(surv[s][w] for s in seeds) for w in range(n_worlds)]
    diffs = [per_world_mean[w] - heur_surv[w] for w in range(n_worlds)]

    md = statistics.mean(diffs)
    se = sd(diffs) / math.sqrt(n_worlds)
    t = md / se if se > 0 else float("inf")
    p = math.erfc(abs(t) / math.sqrt(2))   # two-sided normal-approx p

    return {
        "cell_mean":  statistics.mean(surv_per_seed),
        "cell_sem":   sd(surv_per_seed) / math.sqrt(len(seeds)),
        "mean_diff":  md,
        "ci_lo":      md - 1.96 * se,
        "ci_hi":      md + 1.96 * se,
        "t_stat":     t,
        "p_value":    p,
        "n_worlds":   n_worlds,
        "n_seeds":    len(seeds),
    }


def write_paired_tests(heur_surv: dict[str, list[float]]) -> None:
    rows = []
    print()
    print(f"{'Cell':22s}  {'mean':>7s}  {'SEM':>5s}  {'diff':>7s}  "
          f"{'95% CI':>20s}  {'p':>8s}")
    print("-" * 78)
    for name, csv_base, heur_b in CELLS:
        r = cell_paired_test(csv_base, heur_surv[heur_b])
        if r is None:
            print(f"{name:22s}  (no data)")
            continue
        print(f"{name:22s}  {r['cell_mean']:7.1f}  {r['cell_sem']:5.1f}  "
              f"{r['mean_diff']:+7.1f}  [{r['ci_lo']:+6.1f}, {r['ci_hi']:+6.1f}]  "
              f"{r['p_value']:8.4f}")
        rows.append({"cell": name, **r})

    if not rows:
        return
    fields = list(rows[0].keys())
    with open(PAIRED_TESTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nwrote {PAIRED_TESTS}  ({len(rows)} rows)")


def main() -> None:
    heur_surv = load_heuristic_survivals()
    heur_means = {k: statistics.mean(v) for k, v in heur_surv.items()}
    write_bench_summary(heur_means)
    write_paired_tests(heur_surv)


if __name__ == "__main__":
    main()
