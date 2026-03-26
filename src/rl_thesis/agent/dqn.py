"""
Deep Q-Network Agent implementation.

Combines the neural network, replay buffer, and exploration strategy
into a complete DQN agent with training and inference capabilities.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_thesis.config.config import DQNConfig
from rl_thesis.agent.network import create_network, DQNetwork
from rl_thesis.agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:

    
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DQNConfig,
        grid_h: int,
        grid_w: int,
        spatial_channels: int,
        scalar_dim: int,
    ):
        """
        Initialize the DQN agent.
        
        Args:
            observation_size: Size of observation vector
            action_size: Number of possible actions
            config: DQN configuration
            grid_h: Height of spatial observation grid
            grid_w: Width of spatial observation grid
            spatial_channels: Number of spatial observation channels
            scalar_dim: Number of non-spatial scalar features
        """
        self.config = config
        self.observation_size = observation_size
        self.action_size = action_size
        
        # Setup device (prefer MPS for Apple Silicon)
        self.device = self._setup_device()
        
        # Target & policy networks
        net_kwargs = dict(
            merge_hidden=self.config.merge_hidden,
            head_hidden=self.config.head_hidden,
            cnn_channels=self.config.cnn_channels,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_channels=spatial_channels,
            scalar_dim=scalar_dim,
        )
        self.policy_net = create_network(
            action_size=action_size, **net_kwargs,
        ).to(self.device)

        self.target_net = create_network(
            action_size=action_size, **net_kwargs,
        ).to(self.device)
        
        # Copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        wd = self.config.weight_decay
        decay_params = [
            p for n, p in self.policy_net.named_parameters()
            if p.requires_grad and "bias" not in n and "norm" not in n.lower()
        ]
        no_decay_params = [
            p for n, p in self.policy_net.named_parameters()
            if p.requires_grad and ("bias" in n or "norm" in n.lower())
        ]
        self.optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            eps=1e-5,
        )

        
        effective_training_steps = max(
            self.config.total_timesteps - self.config.min_buffer_size + 1, 1
        )
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=effective_training_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=10,
        )
        
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            beta_frames=self.config.epsilon_decay_steps
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.epsilon_decay = (
            (self.config.epsilon_start - self.config.epsilon_end) / 
            self.config.epsilon_decay_steps
        )
        
        # Training state
        self.steps_done = 0
        self.updates_done = 0
        
        # Metrics tracking (bounded to prevent memory growth in long runs)
        self.loss_history: deque[float] = deque(maxlen=10000)
        self.q_value_history: deque[float] = deque(maxlen=10000)

    def _setup_device(self) -> torch.device:
        """Setup the compute device (MPS, CUDA, or CPU)."""
        choice = self.config.device
        if choice == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if choice == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            training: If True, use epsilon-greedy; if False, greedy
            
        Returns:
            Selected action index
        """
        if training:
            # Decay epsilon
            if self.steps_done > 0:
                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon - self.epsilon_decay
                )
            self.steps_done += 1
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Greedy action from policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.config.min_buffer_size):
            return None
        
        loss = self._train_step_prioritized()

        self.updates_done += 1
        
        # Step the learning rate scheduler (guard against overshoot)
        if self.lr_scheduler._step_count <= self.lr_scheduler.total_steps:
            self.lr_scheduler.step()
        
        # Update target network periodically
        if self.updates_done % self.config.target_update_freq == 0:
            self._update_target_network()

        self.loss_history.append(loss)
        
        return loss
    
    def _compute_td(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute current Q-values and Double-DQN targets.

        Returns:
            (current_q, target_q) — both shape (batch,).
        """
        rewards = torch.clamp(rewards, -100.0, 100.0)

        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            # Double DQN: select actions with policy net, evaluate with target net
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            next_q = torch.clamp(next_q, -1000.0, 1000.0)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        return current_q, target_q

    def _optimize(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient clipping."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def _to_device(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.to(self.device) for t in tensors)

    def _train_step_prioritized(self) -> float:
        """Training step with prioritized experience replay."""
        (states, actions, rewards, next_states, dones,
         indices, weights) = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones, weights = self._to_device(
            states, actions, rewards, next_states, dones, weights
        )

        current_q, target_q = self._compute_td(
            states, actions, rewards, next_states, dones
        )

        # Importance-weighted loss
        td_errors = (current_q - target_q).detach().cpu().numpy()
        element_wise_loss = nn.functional.smooth_l1_loss(
            current_q, target_q, reduction='none'
        )
        loss = (element_wise_loss * weights).mean()

        self.q_value_history.append(current_q.mean().item())
        self._optimize(loss)

        self.replay_buffer.update_priorities(indices, td_errors)

        return loss.item()
    
    def _update_target_network(self) -> None:
        """Hard update of target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_current_lr(self) -> float:
        """Get the current learning rate from scheduler."""
        return self.lr_scheduler.get_last_lr()[0]
    
    def save(self, path: str, max_retries: int = 3) -> None:
        """
        Save agent state to disk with retry logic and atomic writes.
        
        Args:
            path: Path to save checkpoint
            max_retries: Number of retry attempts on failure
        """
        encoder = self.policy_net.encoder
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'epsilon': self.epsilon,
            'config': self.config,
            'observation_size': self.observation_size,
            'action_size': self.action_size,
            'grid_h': encoder.grid_h,
            'grid_w': encoder.grid_w,
            'spatial_channels': encoder.spatial_channels,
            'scalar_dim': encoder.scalar_dim,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                # Write to temp file first, then atomic rename
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix='.pt.tmp', 
                    dir=path.parent
                )
                os.close(temp_fd)
                
                torch.save(checkpoint, temp_path)
                shutil.move(temp_path, path)
                return  # Success
                
            except (RuntimeError, OSError) as e:
                # Clean up temp file if it exists
                if 'temp_path' in locals() and Path(temp_path).exists():
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 sec
                    print(f"\nCheckpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to save checkpoint after {max_retries} attempts: {e}"
                    ) from e
    
    def load(self, path: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler if present (for backward compatibility)
        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        self.steps_done = checkpoint['steps_done']
        self.updates_done = checkpoint['updates_done']
        self.epsilon = checkpoint['epsilon']
    
    @classmethod
    def from_checkpoint(cls, path: str) -> 'DQNAgent':
        """
        Create an agent from a saved checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded DQNAgent instance
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        agent = cls(
            observation_size=checkpoint['observation_size'],
            action_size=checkpoint['action_size'],
            config=checkpoint['config'],
            grid_h=checkpoint.get('grid_h', 15),
            grid_w=checkpoint.get('grid_w', 15),
            spatial_channels=checkpoint.get('spatial_channels', 3),
            scalar_dim=checkpoint.get('scalar_dim', 3),
        )
        
        agent.load(path)
        return agent
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        recent_losses = self.loss_history[-100:] if self.loss_history else [0]
        recent_q = self.q_value_history[-100:] if self.q_value_history else [0]
        
        return {
            'steps': self.steps_done,
            'updates': self.updates_done,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(recent_losses),
            'avg_q_value': np.mean(recent_q),
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device),
        }
