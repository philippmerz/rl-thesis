"""
main training loop with:
- episode collection
- periodic evaluation
- checkpointing
- progress logging
"""
from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any
import numpy as np
from tqdm import tqdm

from rl_thesis.environment.gym_env import SurvivalEnv
from rl_thesis.agent.dqn import DQNAgent
from rl_thesis.training.metrics import MetricsTracker

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig, DQNConfig


class Trainer:
    def __init__(self, world_config: WorldConfig, dqn_config: DQNConfig):
        self.world_config = world_config
        self.dqn_config = dqn_config

        self.checkpoint_dir = dqn_config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = dqn_config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        self.env = SurvivalEnv(world_config)
 
        # Create agent
        self.agent = DQNAgent(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            use_prioritized_replay=True,
            use_dueling=True,
            grid_h=world_config.observation_grid_size,
            grid_w=world_config.observation_grid_size,
            spatial_channels=world_config.num_spatial_channels,
            scalar_dim=world_config.num_scalars,
        )
        
        # Callbacks
        self.on_episode_end: Optional[Callable[[int, Dict], None]] = None
        self.on_checkpoint: Optional[Callable[[int, str], None]] = None
    
    def train(
        self,
        total_steps: Optional[int] = None,
        progress_bar: bool = True,
        eval_callback: Optional[Callable] = None,
    ) -> MetricsTracker:
        """
        Run the training loop.
        
        Args:
            total_steps: Total training steps (uses config if None)
            progress_bar: Whether to show tqdm progress bar
            eval_callback: Optional callback for evaluation visualization
            
        Returns:
            MetricsTracker with all training metrics
        """
        total_steps = total_steps or self.dqn_config.total_timesteps
        
        # Initialize
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        
        # Warmup phase: 
        # This ensures stable gradients and diversifies initial experience
        curr_buffer_size = len(self.agent.replay_buffer)
        min_buffer_size = self.dqn_config.min_buffer_size
        
        if curr_buffer_size < min_buffer_size:
            warmup_steps = min_buffer_size - curr_buffer_size
            print(f"Warming up replay buffer with {warmup_steps} random actions...")
            
            with tqdm(total=warmup_steps, disable=not progress_bar, desc="Warmup") as pbar_warmup:
                # Use a specific warmup state loop to avoid mixing with main training vars
                w_state, _ = self.env.reset()
                for _ in range(warmup_steps):
                    # Random action for exploration
                    action = np.random.randint(self.env.action_size)
                    w_next_state, reward, terminated, truncated, _ = self.env.step(action)
                    w_done = terminated or truncated
                    
                    # Store transition
                    self.agent.store_transition(w_state, action, reward, w_next_state, terminated)
                    
                    w_state = w_next_state
                    if w_done:
                        w_state, _ = self.env.reset()
                    
                    pbar_warmup.update(1)
            
            # Reset environment for main training
            state, _ = self.env.reset()
        
        # Progress bar
        pbar = tqdm(total=total_steps, disable=not progress_bar, desc="Training")
        
        for step in range(total_steps):
            # Select and execute action
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # CRITICAL: For replay buffer, only use `terminated` (death), not `truncated` (time limit)
            # Truncation means the episode was cut short but the agent was still alive,
            # so future rewards should still be bootstrapped (done=False for Q-learning)
            self.agent.store_transition(state, action, reward, next_state, terminated)
            
            # Train
            loss = self.agent.train_step()

            # Periodic plasticity diagnostics (gate on training steps, not env steps)
            plasticity_interval = getattr(
                self.config.plasticity, 'log_interval', 1000
            )
            train_steps = self.agent.updates_done
            if (
                loss is not None
                and self.config.plasticity.enabled
                and train_steps > 0
                and train_steps % plasticity_interval == 0
            ):
                snap = self.agent.plasticity_tracker.snapshot(train_steps)
                self.agent.plasticity_tracker.history.append(snap)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Episode end
            if done:
                episode_count += 1
                episode_stats = self.env.get_episode_stats()
                
                # Callback
                if self.on_episode_end:
                    self.on_episode_end(episode_count, episode_stats)
                
                # Reset
                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            # Periodic evaluation
            if step > 0 and step % self.dqn_config.eval_freq == 0:
                eval_reward = self.evaluate(
                    num_episodes=self.dqn_config.eval_episodes,
                    callback=eval_callback
                )
                pbar.set_postfix({
                    'eval_reward': f"{eval_reward:.1f}",
                    'epsilon': f"{self.agent.epsilon:.3f}",
                })
            
            # Periodic checkpointing
            if step > 0 and step % self.dqn_config.checkpoint_freq == 0:
                checkpoint_path = self._save_checkpoint(step)
                if self.on_checkpoint:
                    self.on_checkpoint(step, checkpoint_path)
            
            # Update progress bar
            pbar.update(1)
            if step % 1000 == 0:
                pbar.set_description(
                    f"Ep {episode_count} | "
                    f"Avg R: {self.metrics.get_avg_reward():.1f} | "
                    f"ε: {self.agent.epsilon:.2f}"
                )
        
        pbar.close()
        
        # Final checkpoint
        self._save_checkpoint(total_steps, final=True)
        
        # Save plasticity diagnostics
        if self.config.plasticity.enabled:
            self._save_plasticity_diagnostics()
        
        return self.metrics
    
    def evaluate(
        self,
        num_episodes: int = 10,
        callback: Optional[Callable] = None,
    ) -> float:
        """
        Run evaluation episodes with greedy policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            callback: Optional callback receiving (env, state, action, step)
            
        Returns:
            Average episode reward
        """
        eval_env = SurvivalEnv(self.world_config)
        
        total_rewards = []
        
        for ep in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0.0
            step = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                
                if callback:
                    callback(eval_env, state, action, step)
                
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                step += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return float(np.mean(total_rewards))
    
    def _save_checkpoint(self, step: int, final: bool = False) -> str:
        """Save a training checkpoint."""
        if final:
            filename = "model_final.pt"
        else:
            filename = f"model_step_{step}.pt"
        
        path = str(self.checkpoint_dir / filename)
        self.agent.save(path)
        
        # Copy to 'latest' for easy loading (avoids redundant serialization)
        latest_path = str(self.checkpoint_dir / "model_latest.pt")
        shutil.copy(path, latest_path)
        
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint to resume training."""
        self.agent.load(path)
    
    def get_agent(self) -> DQNAgent:
        """Get the trained agent."""
        return self.agent

    def _save_plasticity_diagnostics(self) -> None:
        """Save plasticity tracker history to a JSON file."""
        tracker = self.agent.plasticity_tracker
        if not tracker.history:
            return
        data = tracker.history_to_dict()
        path = self.log_dir / "plasticity_diagnostics.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def train_agent(
    world_config: WorldConfig,
    dqn_config: DQNConfig,
    total_steps: int | None = None,
    progress_bar: bool = True,
) -> DQNAgent:
    trainer = Trainer(world_config, dqn_config)
    trainer.train(total_steps=total_steps, progress_bar=progress_bar)
    return trainer.get_agent()
