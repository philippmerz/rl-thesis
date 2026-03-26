"""
Gym-style environment wrapper for the survival world.

Follows standard step/reset API pattern.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Dict, Optional, Any
import numpy as np

from rl_thesis.environment.world import World, WorldState

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig


class SurvivalEnv:
    """
    Standardized interface:
    - reset() -> observation
    - step(action) -> (observation, reward, done, truncated, info)
    
    Fully self-contained, can run headlessly.
    """
    
    def __init__(self, config: WorldConfig):
        self.config = config
        self.initial_seed = self.config.initial_seed
        self._episode_seed = self.initial_seed
        self.max_steps = self.config.max_steps

        # Create internal world
        self._world = World(config)
        
        # Track episode state
        self._current_step = 0
        self._episode_count = 0
        
    @property
    def seed(self) -> int:
        """Current episode seed (backward-compat alias)."""
        return self._episode_seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._episode_seed = value

    @property
    def observation_size(self) -> int:
        """Size of the observation vector."""
        return self._world.observation_size
    
    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return self._world.action_size
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional new random seed
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            self._episode_seed = seed
        else:
            # Increment seed for variety across episodes
            self._episode_seed += 1
        
        self._world.reset(self._episode_seed)
        self._current_step = 0
        self._episode_count += 1
        
        observation = self._world.get_observation()
        info = {
            'episode': self._episode_count,
            'seed': self._episode_seed,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        action = max(0, min(self.action_size - 1, action))
        
        # Step the world
        _, reward, terminated, info = self._world.step(action)
        
        self._current_step += 1
        
        # Check for truncation (max steps reached)
        truncated = self._current_step >= self.max_steps
        
        # Get new observation
        observation = self._world.get_observation()
        
        # Add extra info
        info['step'] = self._current_step
        info['truncated'] = truncated
        
        return observation, reward, terminated, truncated, info
    
    def get_state(self) -> WorldState:
        """Get the current world state (for visualization)."""
        return self._world.get_state()
    
    def get_world(self) -> World:
        """Get direct access to the world (for advanced use)."""
        return self._world
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current/last episode."""
        agent = self._world.agent
        return {
            'ticks_survived': agent.ticks_survived,
            'food_eaten': agent.food_eaten,
            'damage_taken': agent.damage_taken,
            'final_health': agent.health,
            'final_hunger': agent.hunger,
            'episode_reward': self._world.episode_reward,
            'terminated_by_death': not agent.is_alive,
        }


