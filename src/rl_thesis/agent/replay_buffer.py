"""Prioritized experience replay with n-step returns.

Observations are stored in pre-allocated numpy arrays for contiguous memory
and fast vectorized sampling. The SumTree enables O(log n) prioritized
sampling with importance-sampling weight correction.
"""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch


class SumTree:
    """Binary tree where each parent is the sum of its children.

    Supports O(log n) priority updates and proportional sampling.
    All operations are iterative to avoid stack depth issues at scale.
    Position management is the caller's responsibility — this class
    is a pure priority structure.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    @property
    def total(self) -> float:
        return self.tree[0]

    def update(self, leaf_idx: int, priority: float) -> None:
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def sample(self, s: float) -> Tuple[int, float]:
        """Find the leaf for cumulative sum *s*. Returns (leaf_idx, priority)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx]


class NStepPrioritizedBuffer:
    """Memory-efficient prioritized replay buffer with n-step returns.

    Stores transitions in pre-allocated numpy arrays (no Python-object-per-
    transition overhead). Computes multi-step discounted returns on the fly
    and stores the bootstrapping discount factor per transition to handle
    variable-length returns at episode boundaries.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_step: int,
        gamma: float,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_step = n_step
        self.gamma = gamma

        # --- storage (pre-allocated) ---
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.gamma_ns = np.zeros(capacity, dtype=np.float32)

        # --- priority tree ---
        self.tree = SumTree(capacity)

        # --- PER hyper-parameters ---
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6
        self.max_priority = 1.0

        # --- write bookkeeping ---
        self._write_pos = 0
        self.size = 0

        # --- n-step accumulator ---
        # Each entry: (obs, action, reward, next_obs, done)
        self._pending: deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()


    # Public API
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._pending.append((state, action, reward, next_state, done))

        if len(self._pending) >= self.n_step:
            self._commit_oldest()

        if done:
            # Flush remaining pending transitions with truncated n-step returns
            while self._pending:
                self._commit_oldest()

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        torch.Tensor,
    ]:
        """Sample a prioritized batch.

        Returns:
            (states, actions, returns, next_states, dones, gamma_ns,
             tree_indices, importance_weights)
        """
        indices = np.empty(batch_size, dtype=np.int64)
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            leaf_idx, priority = self.tree.sample(s)
            data_idx = leaf_idx - self.capacity + 1
            tree_indices[i] = leaf_idx
            indices[i] = data_idx
            priorities[i] = priority

        beta = self._current_beta()
        sampling_probs = priorities / self.tree.total
        weights = (self.size * sampling_probs) ** (-beta)
        weights /= weights.max()

        self.frame += 1

        return (
            torch.as_tensor(self.states[indices]),
            torch.as_tensor(self.actions[indices]),
            torch.as_tensor(self.returns[indices]),
            torch.as_tensor(self.next_states[indices]),
            torch.as_tensor(self.dones[indices].astype(np.float32)),
            torch.as_tensor(self.gamma_ns[indices]),
            tree_indices,
            torch.as_tensor(weights.astype(np.float32)),
        )

    def update_priorities(
        self, tree_indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(int(idx), float(p))
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def discard_pending(self) -> None:
        """Drop uncommitted n-step entries (e.g. after a forced env reset)."""
        self._pending.clear()


    # Internals
    def _current_beta(self) -> float:
        progress = min(1.0, self.frame / max(self.beta_frames, 1))
        return self.beta_start + (1.0 - self.beta_start) * progress

    def _commit_oldest(self) -> None:
        """Compute the n-step return for the oldest pending transition and store it."""
        entries = list(self._pending)
        state, action, _, _, _ = entries[0]

        n_step_return = 0.0
        gamma_power = 1.0
        terminal = False

        for _, _, r, _, d in entries:
            n_step_return += gamma_power * r
            gamma_power *= self.gamma
            if d:
                terminal = True
                break

        # The bootstrap state is the *next_state* of the last contributing step
        last_contributing = entries[min(len(entries), self.n_step) - 1]
        if terminal:
            # Find the terminal step
            for entry in entries:
                if entry[4]:  # done flag
                    last_contributing = entry
                    break
        next_state = last_contributing[3]

        pos = self._write_pos % self.capacity
        self.states[pos] = state
        self.actions[pos] = action
        self.returns[pos] = n_step_return
        self.next_states[pos] = next_state
        self.dones[pos] = terminal
        self.gamma_ns[pos] = gamma_power

        leaf_idx = pos + self.capacity - 1
        priority = self.max_priority ** self.alpha
        self.tree.update(leaf_idx, priority)

        self._write_pos += 1
        self.size = min(self.size + 1, self.capacity)
        self._pending.popleft()
