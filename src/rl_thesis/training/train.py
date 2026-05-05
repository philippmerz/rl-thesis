from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rl_thesis.config.experiment_configs import (
    make_world_config, describe_config,
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
               warm_start: str | None = None) -> None:
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
