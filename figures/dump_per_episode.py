"""Per-episode benchmark dump.

For each ablation cell (config x seed) and the heuristic, replay
100 evaluation episodes and dump per-episode behavioural statistics
to ``eval_logs/per_episode/<config>_seed_<seed>.csv``.

Columns: episode, seed, survival, food_eaten, damage_taken,
final_health, final_hunger, terminated, shelter_ticks,
first_food_tick, reward, stationary_ticks, osc_ticks,
death_cause, final_enc_ticks, final_enc_net_delta. ``reward`` is
the undiscounted episode return under the cell's own reward
configuration. ``death_cause`` attributes the terminal damage
(``enemy`` or ``starvation``; empty when the episode reached the
cap). The ``final_enc_*`` columns describe the final encounter,
the trailing run of ticks with an enemy visible ending at episode
termination: its length and the net change in Manhattan distance
to the nearest visible enemy attributable to the agent's own moves
(enemy positions held fixed at their pre-step locations, so enemy
movement is not misattributed to the agent; negative = the agent
closed distance on net).

The data feeds the death-cause classification and the movement-
composition table: per-cell distributions are needed to separate
how episodes end (cap, suicide, failure to escape, starvation)
from what the agent does with its time (stationary, oscillating,
other movement), which aggregate statistics in the original
benchmark logs do not expose.

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
OUT_DIR = REPO_ROOT / "eval_logs" / "per_episode"
COLUMNS = [
    "episode", "seed", "survival", "food_eaten", "damage_taken",
    "final_health", "final_hunger", "terminated",
    "shelter_ticks", "first_food_tick", "reward",
    "stationary_ticks", "osc_ticks",
    "death_cause", "final_enc_ticks", "final_enc_net_delta",
]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _count_position_patterns(positions: list[tuple[int, int]]) -> tuple[int, int]:
    """Count stationary and sustained period-2 oscillation ticks from a
    position trace.

    A stationary tick is one where the agent's position did not change from
    the previous tick. An oscillation tick is one where the position equals
    the position two ticks ago and differs from the previous tick (the agent
    moved and undid the move), provided the same condition holds at an
    adjacent tick. The cycle must therefore complete at least A,B,A,B; a
    single move-and-undo (A,B,A) does not count. Together the two counts
    decompose unproductive motion within the exposed-stasis failure mode.
    """
    stationary = sum(
        1 for t in range(1, len(positions)) if positions[t] == positions[t - 1]
    )
    period2 = [
        t >= 2 and positions[t] == positions[t - 2] and positions[t] != positions[t - 1]
        for t in range(len(positions))
    ]
    osc = sum(
        1 for t in range(len(positions))
        if period2[t] and (period2[t - 1] or (t + 1 < len(positions) and period2[t + 1]))
    )
    return stationary, osc


def _enemy_step_delta(prev_pos, new_pos, enemy_positions, radius):
    """Agent-attributable distance change to the nearest visible enemy.

    Enemy positions are taken before the step and held fixed, so the
    delta reflects only the agent's own move. Returns ``None`` when no
    enemy was visible (within ``radius`` of the agent) before the step.
    """
    visible = [e for e in enemy_positions if _manhattan(prev_pos, e) <= radius]
    if not visible:
        return None
    nearest = min(visible, key=lambda e: _manhattan(prev_pos, e))
    return _manhattan(new_pos, nearest) - _manhattan(prev_pos, nearest)


def _final_encounter(deltas: list[int | None]) -> tuple[int, int]:
    """Length and net distance change of the trailing enemy-visible run.

    Walks backward from the final tick while an enemy was visible
    (delta is not ``None``). The net sum is negative when the agent's
    own moves closed distance to the enemy over the encounter.
    """
    length = 0
    net = 0
    for delta in reversed(deltas):
        if delta is None:
            break
        length += 1
        net += delta
    return length, net


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
        positions = [world.agent.position.as_tuple()]
        enemy_deltas: list[int | None] = []
        last_tick_enemy_hit = False
        radius = world.config.observation_radius
        while True:
            prev_pos = positions[-1]
            enemies_prev = [e.position.as_tuple() for e in world.enemies]
            action = agent.select_action(world)
            _, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            tick += 1
            new_pos = world.agent.position.as_tuple()
            positions.append(new_pos)
            enemy_deltas.append(_enemy_step_delta(prev_pos, new_pos, enemies_prev, radius))
            last_tick_enemy_hit = info.get("damage_taken", 0) > 0
            if world.agent.is_in_shelter:
                shelter_ticks += 1
            if info.get("food_eaten"):
                ep_food += 1
                if first_food_tick < 0:
                    first_food_tick = tick
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        stationary, osc = _count_position_patterns(positions)
        death_cause = ""
        if stats["terminated_by_death"]:
            death_cause = "enemy" if last_tick_enemy_hit else "starvation"
        enc_ticks, enc_net = _final_encounter(enemy_deltas)
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
            stationary,
            osc,
            death_cause,
            enc_ticks,
            enc_net,
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
        world = base_env.get_world()
        positions = [world.agent.position.as_tuple()]
        enemy_deltas: list[int | None] = []
        last_tick_enemy_hit = False
        radius = world.config.observation_radius
        while True:
            prev_pos = positions[-1]
            enemies_prev = [e.position.as_tuple() for e in world.enemies]
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            tick += 1
            world = base_env.get_world()
            new_pos = world.agent.position.as_tuple()
            positions.append(new_pos)
            enemy_deltas.append(_enemy_step_delta(prev_pos, new_pos, enemies_prev, radius))
            last_tick_enemy_hit = info.get("damage_taken", 0) > 0
            if world.agent.is_in_shelter:
                shelter_ticks += 1
            if info.get("food_eaten"):
                ep_food += 1
                if first_food_tick < 0:
                    first_food_tick = tick
            if terminated or truncated:
                break
        stats = base_env.get_episode_stats()
        stationary, osc = _count_position_patterns(positions)
        death_cause = ""
        if stats["terminated_by_death"]:
            death_cause = "enemy" if last_tick_enemy_hit else "starvation"
        enc_ticks, enc_net = _final_encounter(enemy_deltas)
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
            stationary,
            osc,
            death_cause,
            enc_ticks,
            enc_net,
        ])
    return rows


def main():
    from dataclasses import replace

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Limit to a config name (still runs both heuristics).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not regenerate CSVs that already exist.")
    args = ap.parse_args()

    cells = [c for c in CELLS if args.only is None or c[0] == args.only]

    # Heuristic at both episode caps: cap=1000 pairs with the misspecified
    # cells, cap=50000 pairs with the minimal cells.
    for cap, suffix in [(1000, ""), (50000, "_cap50k")]:
        h_path = OUT_DIR / f"heuristic{suffix}.csv"
        if args.skip_existing and h_path.exists():
            print(f"[skip] heuristic cap={cap}: csv exists")
            continue
        h_world = replace(WorldConfig(), max_steps=cap)
        print(f"[heuristic cap={cap}] {NUM_EPISODES} episodes")
        rows = rollout_heuristic(h_world)
        write_rows(h_path, rows)
        print(f"  wrote {h_path}")

    # Each cell runs at its own natural cap, taken from its WorldConfig.
    for config, seed in cells:
        ckpt = REPO_ROOT / "runs" / config / f"seed_{seed}" / "checkpoints" / "model_best.pt"
        if not ckpt.exists():
            print(f"[skip] {config} seed={seed}: no checkpoint at {ckpt}")
            continue
        out_path = OUT_DIR / f"{config}_seed_{seed}.csv"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {config} seed={seed}: csv exists")
            continue
        try:
            world_config = make_world_config(config)
        except Exception:
            world_config = WorldConfig()
        print(f"[{config} seed={seed}] {NUM_EPISODES} episodes (max_steps={world_config.max_steps})")
        rows = rollout_dqn(str(ckpt), world_config)
        write_rows(out_path, rows)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
