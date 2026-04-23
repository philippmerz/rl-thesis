from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rl_thesis.config.experiment_configs import (
    make_world_config, make_dqn_config, get_config_names, describe_config,
)
from rl_thesis.training.trainer import Trainer

if TYPE_CHECKING:
    from rl_thesis.config.config import DQNConfig


def _format_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h {m}m {s}s"


def run_single(config_name: str, seed: int, dqn_config: DQNConfig,
               checkpoint: str | None = None,
               warm_start: str | None = None,
               demo_episodes: int = 0) -> None:
    """Train a single (config, seed) combination.

    If *checkpoint* is provided, training resumes from that file.
    If *warm_start* is provided, only network weights are loaded (fresh
    optimizer, epsilon, and counters).
    """
    run_dir = Path("runs") / config_name / f"seed_{seed}"
    dqn = replace(
        dqn_config,
        checkpoint_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
    )
    world_config = make_world_config(config_name, seed=seed)
    trainer = Trainer(world_config, dqn, checkpoint_path=checkpoint,
                      warm_start_path=warm_start)

    weights = describe_config(config_name)
    print("=" * 60)
    print(f"Config: {config_name}  |  Seed: {seed}")
    print(f"Output: {run_dir}")
    if checkpoint:
        print(f"Resume: {checkpoint}")
    if warm_start:
        print(f"Warm start: {warm_start}")
    print("-" * 60)
    for k, v in weights.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"Device: {trainer.agent.device}  |  Steps: {dqn.total_timesteps:,}")
    print("=" * 60)

    if not checkpoint:
        trainer.metrics.save_run_config(config_name, seed, world_config, dqn)

    if demo_episodes > 0 and not checkpoint:
        n = trainer.load_demonstrations(num_episodes=demo_episodes)
        print(f"Loaded {n:,} demonstration transitions from {demo_episodes} heuristic episodes")

    try:
        metrics = trainer.train(total_steps=dqn.total_timesteps)

        summary = metrics.get_summary()
        print("\n" + "=" * 60)
        print(f"Done: {config_name} / seed {seed}")
        print(f"  Episodes: {summary['episodes']}")
        print(f"  Best eval reward: {summary['best_reward']:.1f}")
        print(f"  Best survival: {summary['best_survival']:.0f} ticks")
        print(f"  Wall time: {_format_time(summary['training_time'])}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving checkpoint...")
        trainer.save_manual_checkpoint(trainer.agent.steps_done)
        print("Saved.")

    finally:
        trainer.metrics.close()


def run_grid(seeds: list[int], configs: list[str] | None = None) -> None:
    """Run the full experiment grid: all configs x all seeds.

    DQN hyperparameters are resolved per config via _dqn overrides
    in REWARD_CONFIGS.
    """
    config_names = configs or get_config_names()
    total_runs = len(config_names) * len(seeds)

    print("=" * 60)
    print(f"Experiment grid: {len(config_names)} configs x {len(seeds)} seeds = {total_runs} runs")
    print(f"Configs: {', '.join(config_names)}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    completed = 0
    for config_name in config_names:
        dqn = make_dqn_config(config_name)
        for seed in seeds:
            completed += 1
            print(f"\n{'#' * 60}")
            print(f"# Run {completed}/{total_runs}")
            print(f"{'#' * 60}")
            run_single(config_name, seed, dqn)

    print(f"\nAll {total_runs} runs complete.")
