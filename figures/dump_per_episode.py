"""Per-episode benchmark dump.

For each ablation cell (config x seed) and the heuristic, replay
100 evaluation episodes and dump per-episode behavioural statistics
to ``vast_logs/per_episode/<config>_seed_<seed>.csv``.

Columns: episode, seed, survival, food_eaten, damage_taken,
final_health, final_hunger, terminated, shelter_ticks,
first_food_tick, reward. ``reward`` is the undiscounted episode
return under the cell's own reward configuration.

The data feeds the failure-mode quantification figure and the
learning-trajectory rationale: per-cell distributions are needed
to compute proxies for stasis, suicide, and active-foraging-but-
dying that aggregate statistics in the original benchmark logs
do not expose.

Run from repository root: python -m figures.dump_per_episode
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rl_thesis.config.config import WorldConfig, HumanHeuristicConfig
from rl_thesis.config.experiment_configs import make_world_config
from rl_thesis.environment.gym_env import SurvivalEnv
from rl_thesis.agent.human_heuristic import HumanHeuristicAgent

from figures.common import REPO_ROOT


CELLS = [
    ("baseline", 42), ("baseline", 43), ("baseline", 44), ("baseline", 45),
    ("baseline_fs", 42), ("baseline_fs", 43), ("baseline_fs", 44), ("baseline_fs", 45),
    ("absolute_proximity", 42), ("absolute_proximity", 43), ("absolute_proximity", 44), ("absolute_proximity", 45),
    ("absolute_proximity_fs", 42), ("absolute_proximity_fs", 43), ("absolute_proximity_fs", 44), ("absolute_proximity_fs", 45),
    ("minimal_cap50k", 42), ("minimal_cap50k", 43), ("minimal_cap50k", 44), ("minimal_cap50k", 45),
    ("minimal_fs_cap50k", 42), ("minimal_fs_cap50k", 43), ("minimal_fs_cap50k", 44), ("minimal_fs_cap50k", 45),
    ("weak_proximity", 42),
]
NUM_EPISODES = 100
START_SEED = 1000
OUT_DIR = REPO_ROOT / "vast_logs" / "per_episode"
COLUMNS = [
    "episode", "seed", "survival", "food_eaten", "damage_taken",
    "final_health", "final_hunger", "terminated",
    "shelter_ticks", "first_food_tick", "reward",
]


def write_rows(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


def rollout_heuristic(world_config: WorldConfig):
    env = SurvivalEnv(world_config)
    agent = HumanHeuristicAgent(
        hunger_threshold=HumanHeuristicConfig.hunger_threshold,
        flee_radius=HumanHeuristicConfig.flee_radius,
    )
    rows = []
    for i in range(NUM_EPISODES):
        seed = START_SEED + i
        env.reset(seed=seed)
        world = env.get_world()
        shelter_ticks = 0
        first_food_tick = -1
        tick = 0
        ep_food = 0
        ep_reward = 0.0
        while True:
            action = agent.select_action(world)
            _, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            tick += 1
            if world.agent.is_in_shelter:
                shelter_ticks += 1
            if info.get("food_eaten"):
                ep_food += 1
                if first_food_tick < 0:
                    first_food_tick = tick
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        rows.append([
            i, seed,
            int(stats["ticks_survived"]),
            int(ep_food),
            float(stats["damage_taken"]),
            float(stats["final_health"]),
            float(stats["final_hunger"]),
            int(bool(stats["terminated_by_death"])),
            shelter_ticks,
            first_food_tick,
            round(float(ep_reward), 4),
        ])
    return rows


def rollout_dqn(checkpoint: str, world_config: WorldConfig):
    from rl_thesis.agent.dqn import DQNAgent
    from rl_thesis.environment.frame_stack import FrameStackEnv

    agent = DQNAgent.from_checkpoint(checkpoint)
    base_env = SurvivalEnv(world_config)
    frame_stack = getattr(agent.config, "frame_stack", 1)
    env = FrameStackEnv(base_env, frame_stack) if frame_stack > 1 else base_env

    rows = []
    for i in range(NUM_EPISODES):
        seed = START_SEED + i
        state, _ = env.reset(seed=seed)
        shelter_ticks = 0
        first_food_tick = -1
        tick = 0
        ep_food = 0
        ep_reward = 0.0
        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            tick += 1
            if base_env.get_world().agent.is_in_shelter:
                shelter_ticks += 1
            if info.get("food_eaten"):
                ep_food += 1
                if first_food_tick < 0:
                    first_food_tick = tick
            if terminated or truncated:
                break
        stats = base_env.get_episode_stats()
        rows.append([
            i, seed,
            int(stats["ticks_survived"]),
            int(ep_food),
            float(stats["damage_taken"]),
            float(stats["final_health"]),
            float(stats["final_hunger"]),
            int(bool(stats["terminated_by_death"])),
            shelter_ticks,
            first_food_tick,
            round(float(ep_reward), 4),
        ])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Limit to a config name (run all its seeds + heuristic).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not regenerate CSVs that already exist.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override episode cap. Default keeps the WorldConfig default of 1000.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix appended to output CSV filenames (before .csv); useful for "
                         "side-by-side runs at different caps.")
    args = ap.parse_args()

    cells = [c for c in CELLS if args.only is None or c[0] == args.only]

    def maybe_lift_cap(cfg: WorldConfig) -> WorldConfig:
        if args.max_steps is None:
            return cfg
        from dataclasses import replace
        return replace(cfg, max_steps=args.max_steps)

    suffix = args.out_suffix
    heuristic_world = maybe_lift_cap(WorldConfig())
    h_path = OUT_DIR / f"heuristic{suffix}.csv"
    if not (args.skip_existing and h_path.exists()):
        print(f"[heuristic] {NUM_EPISODES} episodes (max_steps={heuristic_world.max_steps})")
        rows = rollout_heuristic(heuristic_world)
        write_rows(h_path, rows)
        print(f"  wrote {h_path}")

    for config, seed in cells:
        ckpt = REPO_ROOT / "runs" / config / f"seed_{seed}" / "checkpoints" / "model_best.pt"
        if not ckpt.exists():
            print(f"[skip] {config} seed={seed}: no checkpoint at {ckpt}")
            continue
        out_path = OUT_DIR / f"{config}_seed_{seed}{suffix}.csv"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {config} seed={seed}: csv exists")
            continue
        try:
            world_config = make_world_config(config)
        except Exception:
            world_config = WorldConfig()
        world_config = maybe_lift_cap(world_config)
        print(f"[{config} seed={seed}] {NUM_EPISODES} episodes (max_steps={world_config.max_steps})")
        rows = rollout_dqn(str(ckpt), world_config)
        write_rows(out_path, rows)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
