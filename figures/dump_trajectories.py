"""Per-tick trajectory dump for one representative episode per ablation cell.

For each main ablation cell, the episode with survival closest to the cell's
median (across all 4 training seeds x 100 evaluation worlds = 400 episodes)
is identified from the per_episode dump and replayed with full trace
logging. This yields one ``typical'' episode per cell rather than an
arbitrary draw.

Output:
  eval_logs/trajectories/<cell>.csv     trace, one row per tick
                                        columns: tick, x, y, ate_food
  eval_logs/trajectories/_summary.csv   one row per cell, columns:
                                        cell, training_seed, eval_seed,
                                        total_ticks, terminated, death_cause

Run from repository root: python -m figures.dump_trajectories
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from rl_thesis.config.config import WorldConfig
from rl_thesis.config.experiment_configs import make_world_config
from rl_thesis.environment.gym_env import SurvivalEnv

from figures.common import REPO_ROOT


CELLS_TO_TRACE = [
    "baseline",
    "baseline_fs",
    "absolute_proximity",
    "absolute_proximity_fs",
    "minimal_cap50k",
    "minimal_fs_cap50k",
]
TRAINING_SEEDS = [42, 43, 44, 45]
EVAL_SEED_OFFSET = 1000

PER_EPISODE_DIR = REPO_ROOT / "eval_logs" / "per_episode"
OUT_DIR = REPO_ROOT / "eval_logs" / "trajectories"
TRACE_COLUMNS = ["tick", "x", "y", "ate_food"]
SUMMARY_COLUMNS = [
    "cell", "training_seed", "eval_seed",
    "total_ticks", "terminated", "death_cause",
]


def find_representative_episode(cell: str) -> tuple[int, int] | None:
    """Return (training_seed, episode_index) for the episode whose survival
    is closest to the cell's median across all 400 evaluation episodes."""
    rows: list[tuple[int, int, int]] = []
    for seed in TRAINING_SEEDS:
        path = PER_EPISODE_DIR / f"{cell}_seed_{seed}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append((seed, int(r["episode"]), int(r["survival"])))
    if not rows:
        return None
    median = statistics.median(r[2] for r in rows)
    seed, ep_idx, _ = min(rows, key=lambda r: (abs(r[2] - median), r[0], r[1]))
    return seed, ep_idx


def rollout_one_episode(checkpoint: str, world_config: WorldConfig,
                        eval_seed: int) -> dict:
    from rl_thesis.agent.dqn import DQNAgent
    from rl_thesis.environment.frame_stack import FrameStackEnv

    agent = DQNAgent.from_checkpoint(checkpoint)
    base_env = SurvivalEnv(world_config)
    frame_stack = getattr(agent.config, "frame_stack", 1)
    env = FrameStackEnv(base_env, frame_stack) if frame_stack > 1 else base_env

    state, _ = env.reset(seed=eval_seed)
    pos = base_env.get_world().agent.position.as_tuple()
    trace = [{"tick": 0, "x": pos[0], "y": pos[1], "ate_food": 0}]

    last_info: dict = {}
    tick = 0
    terminated_flag = False
    truncated_flag = False
    while True:
        action = agent.select_action(state, training=False)
        state, _reward, terminated, truncated, info = env.step(action)
        tick += 1
        pos = base_env.get_world().agent.position.as_tuple()
        trace.append({
            "tick": tick,
            "x": pos[0],
            "y": pos[1],
            "ate_food": int(bool(info.get("food_eaten"))),
        })
        last_info = info
        if terminated or truncated:
            terminated_flag = bool(terminated)
            truncated_flag = bool(truncated)
            break

    if truncated_flag and not terminated_flag:
        death_cause = "cap"
    elif last_info.get("damage_taken", 0) > 0:
        death_cause = "enemy"
    else:
        death_cause = "starvation"

    return {
        "trace": trace,
        "total_ticks": tick,
        "terminated": int(terminated_flag),
        "death_cause": death_cause,
    }


def write_trace(cell: str, result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{cell}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        w.writerows(result["trace"])
    return path


def write_summary(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    summary = []
    for cell in CELLS_TO_TRACE:
        pick = find_representative_episode(cell)
        if pick is None:
            print(f"[skip] {cell}: no per_episode data")
            continue
        seed, ep_idx = pick
        eval_seed = EVAL_SEED_OFFSET + ep_idx
        ckpt = REPO_ROOT / "runs" / cell / f"seed_{seed}" / "checkpoints" / "model_best.pt"
        if not ckpt.exists():
            print(f"[skip] {cell}: no checkpoint at {ckpt}")
            continue
        try:
            world_config = make_world_config(cell)
        except Exception:
            world_config = WorldConfig()
        print(f"[{cell}] median pick: seed={seed} ep_idx={ep_idx} eval_seed={eval_seed}")
        result = rollout_one_episode(str(ckpt), world_config, eval_seed)
        path = write_trace(cell, result)
        food_count = sum(r["ate_food"] for r in result["trace"])
        summary.append({
            "cell": cell,
            "training_seed": seed,
            "eval_seed": eval_seed,
            "total_ticks": result["total_ticks"],
            "terminated": result["terminated"],
            "death_cause": result["death_cause"],
        })
        print(f"  wrote {path} ({result['total_ticks']} ticks, {food_count} food, {result['death_cause']})")
    if summary:
        path = write_summary(summary)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
