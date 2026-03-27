from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rl_thesis.config.reward_configs import make_world_config, get_config_names, describe_config
from rl_thesis.training.trainer import Trainer

if TYPE_CHECKING:
    from rl_thesis.config.config import DQNConfig


def _format_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h {m}m {s}s"


def run_single(config_name: str, seed: int, dqn_config: DQNConfig,
               checkpoint: str | None = None) -> None:
    """Train a single (config, seed) combination.

    If *checkpoint* is provided, training resumes from that file.
    """
    run_dir = Path("runs") / config_name / f"seed_{seed}"
    dqn = replace(
        dqn_config,
        checkpoint_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
    )
    world_config = make_world_config(config_name, seed=seed)

    weights = describe_config(config_name)
    print("=" * 60)
    print(f"Config: {config_name}  |  Seed: {seed}")
    print(f"Output: {run_dir}")
    if checkpoint:
        print(f"Resume: {checkpoint}")
    print("-" * 60)
    for k, v in weights.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"Device: {dqn.device}  |  Steps: {dqn.total_timesteps:,}")
    print("=" * 60)

    trainer = Trainer(world_config, dqn, checkpoint_path=checkpoint)
    if not checkpoint:
        trainer.metrics.save_run_config(config_name, seed, world_config, dqn)

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
        trainer.metrics.close()
        print("Saved.")

    finally:
        trainer.metrics.close()


def run_grid(seeds: list[int], dqn_config: DQNConfig,
             configs: list[str] | None = None) -> None:
    """Run the full experiment grid: all configs x all seeds."""
    config_names = configs or get_config_names()
    total_runs = len(config_names) * len(seeds)

    print("=" * 60)
    print(f"Experiment grid: {len(config_names)} configs x {len(seeds)} seeds = {total_runs} runs")
    print(f"Configs: {', '.join(config_names)}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    completed = 0
    for config_name in config_names:
        for seed in seeds:
            completed += 1
            print(f"\n{'#' * 60}")
            print(f"# Run {completed}/{total_runs}")
            print(f"{'#' * 60}")
            run_single(config_name, seed, dqn_config)

    print(f"\nAll {total_runs} runs complete.")
