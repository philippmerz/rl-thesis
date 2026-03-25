from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import random
import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Stores transitions in a circular buffer and samples
    uniformly at random for training.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Whether episode ended
        """
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        uniform random sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Separate and stack into tensors
        states = torch.FloatTensor(np.array([t.state for t in transitions]))
        actions = torch.LongTensor([t.action for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones = torch.FloatTensor([float(t.done) for t in transitions])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size


class SumTree:
    """
    Naive sampling is too slow for many samples.
    
    Sum tree supports O(log n) priority updates and O(log n) sampling
    proportional to priorities.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        # root to leaf, left to right: [parent, left_child, right_child, ...]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf index for a given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get the total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Transition) -> None:
        """Add a new sample with given priority."""
        idx = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float) -> None:
        """Update the priority of a sample."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Transition]:
        """Sample based on cumulative priority."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        
        # Small constant to avoid zero priorities
        self.epsilon = 1e-6
        
        # Track max priority for new samples
        self.max_priority = 1.0
    
    def _get_beta(self) -> float:
        """Get current beta value (annealed)."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + (1.0 - self.beta_start) * progress
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition with max priority."""
        transition = Transition(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Sample a prioritized batch.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, 
                     tree_indices, importance_weights)
        """
        batch = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)
        
        # Divide priority space into segments
        segment = self.tree.total() / batch_size
        
        beta = self._get_beta()
        
        for i in range(batch_size):
            # Sample from each segment
            low = segment * i
            high = segment * (i + 1)
            s = random.uniform(low, high)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices[i] = idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor([float(t.done) for t in batch])
        weights = torch.FloatTensor(weights)
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Tree indices of sampled transitions
            td_errors: Absolute TD-errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Current number of stored transitions."""
        return self.tree.n_entries
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.tree.n_entries >= min_size
