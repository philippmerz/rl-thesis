"""
Main training loop: episode collection, evaluation, checkpointing, metrics logging.
"""
from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any

import numpy as np
from tqdm import tqdm

from rl_thesis.environment.gym_env import SurvivalEnv
from rl_thesis.agent.dqn import DQNAgent
from rl_thesis.training.metrics import MetricsLogger

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig, DQNConfig


class Trainer:
    def __init__(self, world_config: WorldConfig, dqn_config: DQNConfig,
                 checkpoint_path: Optional[str] = None):
        self.world_config = world_config
        self.dqn_config = dqn_config

        self.checkpoint_dir = dqn_config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = dqn_config.log_dir
        self.metrics = MetricsLogger(self.log_dir)

        self.env = SurvivalEnv(world_config)

        self.agent = DQNAgent(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            config=dqn_config,
            grid_h=world_config.observation_grid_size,
            grid_w=world_config.observation_grid_size,
            spatial_channels=world_config.num_spatial_channels,
            scalar_dim=world_config.num_scalars,
        )

        if checkpoint_path:
            self.agent.load(checkpoint_path)
            print(f"Resumed from {checkpoint_path} (step {self.agent.steps_done:,})")

        self.on_episode_end: Optional[Callable[[int, Dict], None]] = None
        self.on_checkpoint: Optional[Callable[[int, str], None]] = None

    def train(
        self,
        total_steps: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ) -> MetricsLogger:
        total_steps = total_steps or self.dqn_config.total_timesteps
        start_step = self.agent.steps_done

        episode_reward = 0.0
        episode_length = 0
        episode_count = self.metrics.episode_count
        recent_losses: deque[float] = deque(maxlen=1000)
        global_step = start_step

        # Warmup: fill replay buffer before training begins.
        # On fresh start, use random actions for exploration diversity.
        # On resume, use the loaded policy to collect on-policy transitions.
        resuming = start_step > 0
        curr_buffer_size = len(self.agent.replay_buffer)
        min_buffer_size = self.dqn_config.min_buffer_size

        if curr_buffer_size < min_buffer_size:
            warmup_steps = min_buffer_size - curr_buffer_size
            label = "policy" if resuming else "random"
            print(f"Warming up replay buffer with {warmup_steps} {label} actions...")

            with tqdm(total=warmup_steps, desc="Warmup") as pbar:
                w_state, _ = self.env.reset()
                for _ in range(warmup_steps):
                    if resuming:
                        action = self.agent.select_action(w_state, training=False)
                    else:
                        action = np.random.randint(self.env.action_size)
                    w_next_state, reward, terminated, truncated, _ = self.env.step(action)

                    self.agent.store_transition(w_state, action, reward, w_next_state, terminated)

                    w_state = w_next_state
                    if terminated or truncated:
                        if not terminated:
                            self.agent.discard_pending()
                        w_state, _ = self.env.reset()

                    pbar.update(1)

        self.agent.discard_pending()
        state, _ = self.env.reset()
        pbar = tqdm(total=total_steps, initial=start_step, desc="Training")

        for step in range(start_step, total_steps):
            global_step = step
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.agent.store_transition(state, action, reward, next_state, terminated)

            loss = self.agent.train_step()
            if loss is not None:
                recent_losses.append(loss)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                episode_count += 1
                episode_stats = self.env.get_episode_stats()
                self.metrics.log_episode(step, episode_stats)

                if self.on_episode_end:
                    self.on_episode_end(episode_count, episode_stats)

                if not terminated:
                    self.agent.discard_pending()

                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Periodic evaluation
            if step > 0 and step % self.dqn_config.eval_freq == 0:
                eval_results = self.evaluate(
                    num_episodes=self.dqn_config.eval_episodes,
                    callback=eval_callback,
                )
                avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                self.metrics.log_eval(
                    step=step,
                    episode=episode_count,
                    eval_results=eval_results,
                    epsilon=self.agent.epsilon,
                    loss=avg_loss,
                )
                pbar.set_postfix({
                    "eval": f"{eval_results['reward']:.1f}",
                    "surv": f"{eval_results['survival']:.0f}",
                    "eps": f"{self.agent.epsilon:.3f}",
                })

            # Periodic checkpointing
            if step > 0 and step % self.dqn_config.checkpoint_freq == 0:
                checkpoint_path = self._save_checkpoint(step)
                if self.on_checkpoint:
                    self.on_checkpoint(step, checkpoint_path)

            pbar.update(1)
            if step % 1000 == 0:
                pbar.set_description(f"Ep {episode_count} | eps {self.agent.epsilon:.2f}")

        pbar.close()
        self._save_checkpoint(total_steps, final=True)

        return self.metrics

    def evaluate(
        self,
        num_episodes: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run greedy evaluation episodes and return aggregate metrics."""
        eval_env = SurvivalEnv(self.world_config)

        rewards = []
        survivals = []
        food_counts = []
        damage_totals = []
        deaths = 0

        for ep in range(num_episodes):
            state, _ = eval_env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_food = 0
            ep_damage = 0.0

            while True:
                action = self.agent.select_action(state, training=False)
                if callback:
                    callback(eval_env, state, action, ep_steps)

                next_state, reward, terminated, truncated, info = eval_env.step(action)

                ep_reward += reward
                ep_steps += 1
                if info.get("food_eaten"):
                    ep_food += 1
                ep_damage += info.get("damage_taken", 0.0)

                state = next_state
                if terminated or truncated:
                    if terminated:
                        deaths += 1
                    break

            rewards.append(ep_reward)
            survivals.append(ep_steps)
            food_counts.append(ep_food)
            damage_totals.append(ep_damage)

        return {
            "reward": float(np.mean(rewards)),
            "survival": float(np.mean(survivals)),
            "food_eaten": float(np.mean(food_counts)),
            "damage_taken": float(np.mean(damage_totals)),
            "death_rate": deaths / num_episodes,
        }

    def _save_checkpoint(self, step: int, final: bool = False) -> str:
        filename = "model_final.pt" if final else f"model_step_{step}.pt"
        path = str(self.checkpoint_dir / filename)
        self.agent.save(path)

        latest_path = str(self.checkpoint_dir / "model_latest.pt")
        shutil.copy(path, latest_path)
        return path

