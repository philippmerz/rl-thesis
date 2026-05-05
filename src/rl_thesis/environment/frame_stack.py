"""Frame-stacking observation wrapper.

Stacks the last N spatial observation grids as additional channels,
giving the agent temporal context (movement direction, approach/retreat).
Scalar features (health, hunger, in_shelter) come from the latest frame only.

Observation layout (flat):
    [frame_{t-N+1}_spatial, ..., frame_t_spatial, scalars_t]

where each frame's spatial block has shape (C * H * W,) and scalars_t has
shape (num_scalars,). The CNN encoder receives spatial_channels = C * N.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from rl_thesis.environment.gym_env import SurvivalEnv
    from rl_thesis.environment.world import World, WorldState


class FrameStackEnv:
    """Wraps a SurvivalEnv to stack the last *n_frames* spatial observations.

    Implements the same interface as SurvivalEnv so it can be used as a
    drop-in replacement in the training loop, evaluation, and demo code.
    """

    def __init__(self, env: SurvivalEnv, n_frames: int):
        assert n_frames >= 2, f"frame stacking requires n_frames >= 2, got {n_frames}"
        self._env = env
        self._n_frames = n_frames

        cfg = env.config
        self._spatial_dim = cfg.num_spatial_channels * cfg.observation_grid_size ** 2
        self._scalar_dim = cfg.num_scalars

        self._frames: deque[np.ndarray] = deque(maxlen=n_frames)

    @property
    def observation_size(self) -> int:
        return self._spatial_dim * self._n_frames + self._scalar_dim

    @property
    def action_size(self) -> int:
        return self._env.action_size

    @property
    def config(self):
        return self._env.config

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self._env.reset(seed=seed)
        spatial = obs[:self._spatial_dim]
        for _ in range(self._n_frames):
            self._frames.append(spatial.copy())
        return self._build_obs(obs), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._frames.append(obs[:self._spatial_dim].copy())
        return self._build_obs(obs), reward, terminated, truncated, info

    def get_state(self) -> WorldState:
        return self._env.get_state()

    def get_world(self) -> World:
        return self._env.get_world()

    def get_episode_stats(self) -> Dict[str, Any]:
        return self._env.get_episode_stats()

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Concatenate stacked spatial frames with current scalars."""
        scalars = raw_obs[self._spatial_dim:]
        parts = list(self._frames)
        parts.append(scalars)
        return np.concatenate(parts)
