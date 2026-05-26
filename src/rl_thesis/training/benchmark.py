"""Benchmark evaluation: compare any agent against the heuristic baseline.

Usage:
    python -m rl_thesis.training.benchmark                              # heuristic only
    python -m rl_thesis.training.benchmark --checkpoint path/to.pt      # DQN vs heuristic
    python -m rl_thesis.training.benchmark --config minimal_fs_cap50k   # custom world config
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np

from rl_thesis.config.config import WorldConfig, HumanHeuristicConfig
from rl_thesis.config.experiment_configs import make_world_config
from rl_thesis.environment.gym_env import SurvivalEnv
from rl_thesis.agent.human_heuristic import HumanHeuristicAgent


def evaluate_heuristic(
    world_config: WorldConfig,
    num_episodes: int = 100,
    start_seed: int = 1000,
) -> Dict[str, List[float]]:
    """Run the heuristic agent and collect per-episode statistics."""
    env = SurvivalEnv(world_config)
    agent = HumanHeuristicAgent(
        hunger_threshold=HumanHeuristicConfig.hunger_threshold,
        flee_radius=HumanHeuristicConfig.flee_radius,
    )

    results: Dict[str, List[float]] = {
        "survival": [], "food_eaten": [], "damage_taken": [],
        "reward": [], "death": [],
    }

    for i in range(num_episodes):
        env.reset(seed=start_seed + i)
        world = env.get_world()
        ep_reward = 0.0
        ep_food = 0

        while True:
            action = agent.select_action(world)
            _, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info.get("food_eaten"):
                ep_food += 1
            if terminated or truncated:
                break

        stats = env.get_episode_stats()
        results["survival"].append(float(stats["ticks_survived"]))
        results["food_eaten"].append(float(ep_food))
        results["damage_taken"].append(float(stats["damage_taken"]))
        results["reward"].append(ep_reward)
        results["death"].append(1.0 if terminated else 0.0)

    return results


def evaluate_dqn(
    checkpoint_path: str,
    world_config: WorldConfig,
    num_episodes: int = 100,
    start_seed: int = 1000,
) -> Dict[str, List[float]]:
    """Run a DQN checkpoint and collect per-episode statistics."""
    from rl_thesis.agent.dqn import DQNAgent
    from rl_thesis.environment.frame_stack import FrameStackEnv

    agent = DQNAgent.from_checkpoint(checkpoint_path)
    env = SurvivalEnv(world_config)

    frame_stack = getattr(agent.config, 'frame_stack', 1)
    if frame_stack > 1:
        env = FrameStackEnv(env, frame_stack)

    results: Dict[str, List[float]] = {
        "survival": [], "food_eaten": [], "damage_taken": [],
        "reward": [], "death": [],
    }

    for i in range(num_episodes):
        state, _ = env.reset(seed=start_seed + i)
        ep_reward = 0.0
        ep_food = 0

        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info.get("food_eaten"):
                ep_food += 1
            if terminated or truncated:
                break

        stats = env.get_episode_stats()
        results["survival"].append(float(stats["ticks_survived"]))
        results["food_eaten"].append(float(ep_food))
        results["damage_taken"].append(float(stats["damage_taken"]))
        results["reward"].append(ep_reward)
        results["death"].append(1.0 if terminated else 0.0)

    return results


def summarize(results: Dict[str, List[float]], label: str) -> Dict[str, float]:
    """Print and return summary statistics."""
    n = len(results["survival"])
    survival = np.array(results["survival"])
    food = np.array(results["food_eaten"])
    damage = np.array(results["damage_taken"])
    death = np.array(results["death"])

    mean_surv = survival.mean()
    se_surv = survival.std(ddof=1) / np.sqrt(n)
    mean_food = food.mean()
    death_rate = death.mean()
    time_limit_rate = 1.0 - death_rate
    mean_damage = damage.mean()

    print(f"\n{'=' * 50}")
    print(f"  {label}  ({n} episodes)")
    print(f"{'=' * 50}")
    print(f"  Survival:     {mean_surv:.1f} +/- {1.96 * se_surv:.1f} (95% CI)")
    print(f"  Food eaten:   {mean_food:.2f}")
    print(f"  Damage taken: {mean_damage:.1f}")
    print(f"  Death rate:   {death_rate:.0%}")
    print(f"  Time limit:   {time_limit_rate:.0%}")
    print(f"  Median surv:  {np.median(survival):.0f}")
    print(f"  Min/Max surv: {survival.min():.0f} / {survival.max():.0f}")

    return {
        "mean_survival": float(mean_surv),
        "se_survival": float(se_surv),
        "mean_food": float(mean_food),
        "death_rate": float(death_rate),
        "mean_damage": float(mean_damage),
    }


def compare(h_results: Dict[str, List[float]], d_results: Dict[str, List[float]]) -> None:
    """Paired t-test on per-world survival differences.

    Both evaluations use matched seeds, so episode i is the same world for
    both agents. The paired analysis honors that pairing and is strictly
    more powerful than an unpaired test when world difficulty drives
    correlated outcomes across the two agents. This matches the analysis
    used for the thesis cell results.
    """
    h_surv = np.array(h_results["survival"])
    d_surv = np.array(d_results["survival"])

    if len(h_surv) != len(d_surv):
        raise ValueError(
            f"Paired test requires equal-length samples: got "
            f"{len(h_surv)} heuristic, {len(d_surv)} DQN"
        )

    diffs = d_surv - h_surv
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0
    df = n - 1

    ci_half = 1.96 * se
    ci_lo, ci_hi = mean_diff - ci_half, mean_diff + ci_half

    # Two-tailed p-value via normal approximation; at n>=30 the t(n-1)
    # distribution is indistinguishable from N(0,1) at the 5% threshold.
    import math
    z = abs(t_stat)
    # use stdlib so we don't need scipy just for this
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - cdf_z)

    print(f"\n{'=' * 50}")
    print(f"  Comparison (DQN - Heuristic, paired)")
    print(f"{'=' * 50}")
    print(f"  Per-world mean diff: {mean_diff:+.1f} ticks")
    print(f"  95% CI:              [{ci_lo:+.1f}, {ci_hi:+.1f}]")
    print(f"  Paired t-test:       t={t_stat:.2f}, df={df}, p~{p_value:.4f}")
    if p_value < 0.05 and mean_diff > 0:
        print(f"  ** DQN significantly outperforms heuristic (p<0.05) **")
    elif p_value < 0.05 and mean_diff < 0:
        print(f"  Heuristic significantly outperforms DQN (p<0.05)")
    else:
        print(f"  No significant difference (p~{p_value:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to DQN checkpoint to evaluate")
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment config name for world config")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--start-seed", type=int, default=1000,
                        help="First episode seed")
    args = parser.parse_args()

    if args.config:
        world_config = make_world_config(args.config)
    else:
        world_config = WorldConfig()

    print(f"Evaluating over {args.episodes} episodes (seeds {args.start_seed}-{args.start_seed + args.episodes - 1})")

    h_results = evaluate_heuristic(world_config, args.episodes, args.start_seed)

    if args.checkpoint:
        d_results = evaluate_dqn(args.checkpoint, world_config, args.episodes, args.start_seed)
        compare(h_results, d_results)


if __name__ == "__main__":
    main()
