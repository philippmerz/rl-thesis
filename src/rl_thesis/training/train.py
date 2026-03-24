from __future__ import annotations

from typing import TYPE_CHECKING

from rl_thesis.training.trainer import Trainer

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig, DQNConfig


def run_training(world_config: WorldConfig, dqn_config: DQNConfig):
    print("=" * 60)
    print("Survival RL - DQN Training")
    print("=" * 60)
    print(f"Device: {dqn_config.device}")
    print(f"World size: {world_config.width}x{world_config.height}")
    print(f"Total steps: {dqn_config.total_timesteps:,}")
    print(f"Learning rate: {dqn_config.learning_rate}")
    print(f"Batch size: {dqn_config.batch_size}")
    print(f"Buffer size: {dqn_config.buffer_size:,}")
    print(f"Epsilon: {dqn_config.epsilon_start} -> {dqn_config.epsilon_end} over {dqn_config.epsilon_decay_steps:,} steps")
    print("=" * 60)

    trainer = Trainer(world_config, dqn_config)

    try:
        print("\nStarting training...")
        metrics = trainer.train(total_steps=dqn_config.total_timesteps)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        summary = metrics.get_summary()
        print(f"Total episodes: {summary['episodes']}")
        print(f"Best episode reward: {summary['best_reward']:.1f}")
        print(f"Best average reward: {summary['best_avg_reward']:.1f}")
        print(f"Best survival time: {summary['best_survival']} ticks")
        print(f"Final average reward: {summary['avg_reward']:.1f}")
        print(f"Final average survival: {summary['avg_survival']:.0f} ticks")
        print(f"Training time: {metrics._format_time(summary['training_time'])}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer._save_checkpoint(trainer.agent.steps_done, final=False)
        trainer.metrics.save()
        print("Checkpoint saved.")
