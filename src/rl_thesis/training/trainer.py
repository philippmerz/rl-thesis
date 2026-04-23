"""
Main training loop: episode collection, evaluation, checkpointing, metrics logging.
"""
from __future__ import annotations

import shutil
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict

import numpy as np
from tqdm import tqdm

from rl_thesis.environment.gym_env import SurvivalEnv
from rl_thesis.environment.frame_stack import FrameStackEnv
from rl_thesis.agent.dqn import DQNAgent
from rl_thesis.agent.human_heuristic import HumanHeuristicAgent
from rl_thesis.config.config import HumanHeuristicConfig
from rl_thesis.training.metrics import MetricsLogger

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig, DQNConfig


class Trainer:
    def __init__(self, world_config: WorldConfig, dqn_config: DQNConfig,
                 checkpoint_path: Optional[str] = None,
                 warm_start_path: Optional[str] = None):
        assert not (checkpoint_path and warm_start_path), \
            "checkpoint_path and warm_start_path are mutually exclusive"

        self.world_config = world_config
        self.dqn_config = dqn_config

        self.checkpoint_dir = dqn_config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = dqn_config.log_dir
        self.metrics = MetricsLogger(self.log_dir)

        self.env = self._make_env(world_config, dqn_config)

        spatial_channels = world_config.num_spatial_channels * dqn_config.frame_stack
        self.agent = DQNAgent(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            config=dqn_config,
            grid_h=world_config.observation_grid_size,
            grid_w=world_config.observation_grid_size,
            spatial_channels=spatial_channels,
            scalar_dim=world_config.num_scalars,
        )

        if checkpoint_path:
            self.agent.load(checkpoint_path)
            print(f"Resumed from {checkpoint_path} (step {self.agent.steps_done:,})")

        if warm_start_path:
            self.agent.load_weights(warm_start_path)
            print(f"Warm-started weights from {warm_start_path}")

        self._frame_stack = dqn_config.frame_stack

        self._latest_checkpoint_step = self._discover_latest_checkpoint_step()
        self._best_eval_survival = -float('inf')
        self._show_progress = sys.stdout.isatty()

    @staticmethod
    def _make_env(world_config: WorldConfig, dqn_config: DQNConfig):
        env = SurvivalEnv(world_config)
        if dqn_config.frame_stack > 1:
            env = FrameStackEnv(env, dqn_config.frame_stack)
        return env

    def load_demonstrations(self, num_episodes: int = 100, start_seed: int = 5000) -> int:
        """Pre-fill the replay buffer with heuristic agent demonstrations.

        Returns the number of transitions added.
        """
        demo_env = self._make_env(self.world_config, self.dqn_config)
        heuristic = HumanHeuristicAgent(
            hunger_threshold=HumanHeuristicConfig.hunger_threshold,
            flee_radius=HumanHeuristicConfig.flee_radius,
        )
        total_transitions = 0

        for ep in range(num_episodes):
            state, _ = demo_env.reset(seed=start_seed + ep)
            world = demo_env.get_world()

            while True:
                action = heuristic.select_action(world)
                next_state, reward, terminated, truncated, info = demo_env.step(action)
                self.agent.store_transition(state, action, reward, next_state, terminated)
                total_transitions += 1
                state = next_state
                if terminated or truncated:
                    if not terminated:
                        self.agent.discard_pending()
                    break

        return total_transitions

    def train(
        self,
        total_steps: Optional[int] = None,
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

            with tqdm(total=warmup_steps, desc="Warmup", disable=not self._show_progress) as pbar:
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
        pbar = tqdm(
            total=total_steps,
            initial=start_step,
            desc="Training",
            disable=not self._show_progress,
        )

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

                if not terminated:
                    self.agent.discard_pending()

                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Periodic evaluation
            if step > 0 and step % self.dqn_config.eval_freq == 0:
                eval_results = self.evaluate(
                    num_episodes=self.dqn_config.eval_episodes,
                )
                avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                self.metrics.log_eval(
                    step=step,
                    episode=episode_count,
                    eval_results=eval_results,
                    epsilon=self.agent.epsilon,
                    loss=avg_loss,
                )
                if eval_results['survival'] > self._best_eval_survival:
                    self._best_eval_survival = eval_results['survival']
                    self.agent.save(str(self.checkpoint_dir / "model_best.pt"))

                pbar.set_postfix({
                    "eval": f"{eval_results['reward']:.1f}",
                    "surv": f"{eval_results['survival']:.0f}",
                    "eps": f"{self.agent.epsilon:.3f}",
                })

            # Periodic checkpointing
            if step > 0 and step % self.dqn_config.checkpoint_freq == 0:
                self._save_periodic_checkpoint(step)

            # Periodic head reset (Nikishin 2022) to counter primacy bias
            reset_freq = self.dqn_config.head_reset_freq
            if reset_freq > 0 and step > 0 and step % reset_freq == 0:
                self.agent.reset_head()
                pbar.write(f"[step {step}] Reset Dueling head weights")

            if step > 0 and step % self.dqn_config.eval_freq == 0:
                self.metrics.log_system(
                    step=step,
                    episode=episode_count,
                    checkpoint_dir=self.checkpoint_dir,
                )

            pbar.update(1)
            if step % 1000 == 0:
                pbar.set_description(f"Ep {episode_count} | eps {self.agent.epsilon:.2f}")

        pbar.close()
        self._save_final_checkpoint(total_steps)

        return self.metrics

    def evaluate(
        self,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """Run greedy evaluation episodes and return aggregate metrics."""
        eval_env = self._make_env(self.world_config, self.dqn_config)

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

    def save_manual_checkpoint(self, step: int) -> str:
        manual_path = self.checkpoint_dir / f"model_manual_step_{step}.pt"
        self.agent.save(str(manual_path))
        self._refresh_latest_checkpoint(
            source_path=manual_path,
            progress_step=max(step, self.agent.steps_done),
        )
        return str(manual_path)

    def _save_periodic_checkpoint(self, step: int) -> str:
        progress_step = max(step, self.agent.steps_done)
        if self._should_keep_periodic_checkpoint(step):
            saved_path = self.checkpoint_dir / f"model_step_{step}.pt"
            self.agent.save(str(saved_path))
            self._refresh_latest_checkpoint(saved_path, progress_step)
        elif progress_step >= self._latest_checkpoint_step:
            saved_path = self.checkpoint_dir / "model_latest.pt"
            self.agent.save(str(saved_path))
            self._latest_checkpoint_step = progress_step
        else:
            saved_path = self.checkpoint_dir / "model_latest.pt"

        self._prune_old_checkpoints()
        return str(saved_path)

    def _save_final_checkpoint(self, step: int) -> str:
        final_path = self.checkpoint_dir / "model_final.pt"
        self.agent.save(str(final_path))
        self._refresh_latest_checkpoint(
            source_path=final_path,
            progress_step=max(step, self.agent.steps_done),
        )
        self._prune_old_checkpoints()
        return str(final_path)

    def _should_keep_periodic_checkpoint(self, step: int) -> bool:
        stride = max(self.dqn_config.checkpoint_keep_stride, 1)
        checkpoint_idx = step // self.dqn_config.checkpoint_freq
        return stride == 1 or (checkpoint_idx - 1) % stride == 0

    def _prune_old_checkpoints(self) -> None:
        for path in self.checkpoint_dir.glob("model_step_*.pt"):
            try:
                step = int(path.stem.rsplit("_", maxsplit=1)[-1])
            except ValueError:
                continue

            # Keep ad-hoc/manual checkpoints that are not on the periodic schedule.
            if step % self.dqn_config.checkpoint_freq != 0:
                continue

            if not self._should_keep_periodic_checkpoint(step):
                path.unlink(missing_ok=True)

    def _refresh_latest_checkpoint(self, source_path: Path, progress_step: int) -> None:
        if progress_step < self._latest_checkpoint_step:
            return

        latest_path = self.checkpoint_dir / "model_latest.pt"
        if source_path != latest_path:
            shutil.copy(source_path, latest_path)
        self._latest_checkpoint_step = progress_step

    def _discover_latest_checkpoint_step(self) -> int:
        latest_step = -1

        for path in self.checkpoint_dir.glob("model_*.pt"):
            parsed_step = self._parse_checkpoint_step_from_name(path)
            if parsed_step is not None:
                latest_step = max(latest_step, parsed_step)

        for name in ("model_latest.pt", "model_final.pt"):
            checkpoint_path = self.checkpoint_dir / name
            if not checkpoint_path.exists():
                continue

            try:
                import torch

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                latest_step = max(latest_step, int(checkpoint.get("steps_done", -1)))
            except Exception:
                continue

        return latest_step

    @staticmethod
    def _parse_checkpoint_step_from_name(path: Path) -> Optional[int]:
        stem = path.stem
        for prefix in ("model_step_", "model_manual_step_"):
            if stem.startswith(prefix):
                suffix = stem[len(prefix):]
                try:
                    return int(suffix)
                except ValueError:
                    return None
        return None
