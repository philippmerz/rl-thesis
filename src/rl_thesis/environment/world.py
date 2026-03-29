from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Dict, Set
import numpy as np

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig

from rl_thesis.environment.entities import (
    Agent, Enemy, Food, Shelter, Position, Direction
)


@dataclass
class WorldState:
    """
    Immutable snapshot of the world state.
    
    Decouple observation from internal, mutable world state.
    """
    agent_position: Tuple[int, int]
    agent_health: float
    agent_hunger: float
    agent_max_health: float
    agent_max_hunger: float
    agent_in_shelter: bool
    agent_alive: bool
    
    enemies: List[Tuple[int, int]]
    food: List[Tuple[int, int]]
    shelters: List[Tuple[int, int]]
    
    ticks: int
    

class World:
    """
    Core simulation logic for the survival environment.

    Manages:
    - Agent state (health, hunger, position)
    - Entity spawning and removal (enemies, food)
    - Entity updates (enemy movement, attacking)
    - Collision detection and resolution
    - Reward calculation

    All logic is self-contained with no visualization dependencies.
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.seed = self.config.initial_seed

        # Instance-level RNGs — avoid mutating global state (B5 fix).
        self._rng = random.Random(self.seed)
        
        # Initialize entities
        self.agent: Agent = self._create_agent()
        self.enemies: List[Enemy] = []
        self.food_items: List[Food] = []
        self.shelters: List[Shelter] = []
        
        # Precompute sets/dicts for O(1) presence checks
        self._shelter_positions: Set[Tuple[int, int]] = set()
        self._food_positions: Dict[Tuple[int, int], Food] = {}
        self._enemy_positions: Dict[Tuple[int, int], List[Enemy]] = {}
        
        # World state
        self.ticks = 0
        self.episode_reward = 0.0

        # Generate static elements
        self._generate_shelters()
        
        # Spawn initial entities
        self._spawn_initial_entities()
    
    def _create_agent(self) -> Agent:
        pos = (self.config.width // 2, self.config.height // 2)
        
        return Agent.create(
            x=pos[0],
            y=pos[1],
            max_health=self.config.max_health,
            max_hunger=self.config.max_hunger,
        )
    
    def _generate_shelters(self) -> None:
        """Generate static shelters based on shelter density."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self._rng.random() < self.config.shelter_density:
                    shelter = Shelter.create(
                        x=x, 
                        y=y, 
                        protection=self.config.shelter_protection
                    )
                    self.shelters.append(shelter)
                    self._shelter_positions.add((x, y))
    
    def _spawn_initial_entities(self) -> None:
        """Spawn initial food and enemies."""
        for _ in range(int(self.config.max_food * self.config.initial_food_fraction)):
            self._try_spawn_food()
        
        for _ in range(int(self.config.max_enemies * self.config.initial_enemy_fraction)):
            self._try_spawn_enemy()
    
    def _try_spawn_food(self) -> bool:
        """Attempt to spawn food at a random valid location."""
        if len(self.food_items) >= self.config.max_food:
            return False
        
        # Try random positions until we find a valid one
        # Guarantees less than 0.1% failure at 50% occupied map
        SPAWN_ATTEMPTS = 10
        for _ in range(SPAWN_ATTEMPTS):
            x = self._rng.randint(0, self.config.width - 1)
            y = self._rng.randint(0, self.config.height - 1)
            
            # Don't spawn on shelter or existing food
            pos = (x, y)
            if pos in self._shelter_positions or pos in self._food_positions:
                continue
            
            # Check spawn probability
            if self._rng.random() > self.config.food_density:
                continue
            
            food = Food.create(x=x, y=y, nutrition_value=self.config.food_value)
            self.food_items.append(food)
            self._food_positions[pos] = food
            return True
        
        return False
    
    def _try_spawn_enemy(self) -> bool:
        """Attempt to spawn enemy at a random valid location."""

        if len(self.enemies) >= self.config.max_enemies:
            return False
        
        # Try random positions
        # Guarantees less than 0.1% failure at 50% occupied map
        SPAWN_ATTEMPTS = 10
        for _ in range(SPAWN_ATTEMPTS):
            x = self._rng.randint(0, self.config.width - 1)
            y = self._rng.randint(0, self.config.height - 1)
            
            # Don't spawn on shelter or near agent
            pos = (x, y)
            if pos in self._shelter_positions:
                continue
            
            agent_pos = self.agent.position.as_tuple()
            if abs(x - agent_pos[0]) < 5 and abs(y - agent_pos[1]) < 5:
                continue  # Too close to agent
            
            # Check spawn probability
            if self._rng.random() > self.config.enemy_density * 10:  # Scale up for spawning check
                continue
            
            enemy = Enemy.create(
                x=x, 
                y=y, 
                damage=self.config.enemy_damage,
                speed=self.config.enemy_speed,
                vision_range=self.config.enemy_vision_range,
            )
            self.enemies.append(enemy)
            
            # Update spatial lookup
            if pos not in self._enemy_positions:
                self._enemy_positions[pos] = []
            self._enemy_positions[pos].append(enemy)
            
            return True
        
        return False
    
    def reset(self, seed: int) -> None:
        """Reset the world to initial state."""
        self.seed = seed
        
        self._rng = random.Random(self.seed)
        
        # Reset entities
        self.agent = self._create_agent()
        self.enemies = []
        self.food_items = []
        self.shelters = []
        
        # Reset spatial lookups
        self._shelter_positions = set()
        self._food_positions = {}
        self._enemy_positions = {}
        
        # Reset state
        self.ticks = 0
        self.episode_reward = 0.0
        
        # Regenerate world
        self._generate_shelters()
        self._spawn_initial_entities()
    
    def step(self, action: int) -> Tuple[float, bool, Dict]:
        """
        Execute one simulation step.
        
        Args:
            action: Movement action (0=none, 1=up, 2=down, 3=left, 4=right)
            
        Returns:
            Tuple of (reward, done, info)
        """
        reward = 0.0
        info = {}
        
        # 1. Move agent
        direction = Direction.from_action(action)
        old_position = self.agent.position.as_tuple()
        self.agent.move(direction, self.config.width, self.config.height)
        new_position = self.agent.position.as_tuple()
        
        # Apply movement cost
        if old_position != new_position:
            self.agent.deplete_hunger(self.config.movement_cost)
        
        # 2. Check if agent is in shelter
        self.agent.is_in_shelter = new_position in self._shelter_positions
        
        # 3. Check for food consumption
        if new_position in self._food_positions:
            food = self._food_positions[new_position]
            if food.is_active:
                nutrition = food.consume()
                self.agent.eat(nutrition)
                reward += self.config.reward_food_eaten
                del self._food_positions[new_position]
                info['food_eaten'] = True
        
        # 4. Update enemies and check attacks
        self._enemy_positions = {}
        total_damage = 0.0
        occupied: set = set()
        
        for enemy in self.enemies:
            enemy.update(
                self.agent.position, 
                self.config.width, 
                self.config.height,
                shelter_positions=self._shelter_positions,
                occupied_positions=occupied,
                rng=self._rng,
            )
            
            # Update spatial lookup
            pos = enemy.position.as_tuple()
            occupied.add(pos)
            if pos not in self._enemy_positions:
                self._enemy_positions[pos] = []
            self._enemy_positions[pos].append(enemy)
            
            # Check for attack
            if enemy.can_attack(self.agent.position):
                total_damage += enemy.damage
        
        # Apply damage with shelter protection
        if total_damage > 0:
            protection = self.config.shelter_protection if self.agent.is_in_shelter else 0.0
            self.agent.take_damage(total_damage, protection)
            reward += self.config.reward_enemy_damage_taken * (total_damage * (1 - protection))
            info['damage_taken'] = total_damage * (1 - protection)
        
        # 5. Deplete hunger
        self.agent.deplete_hunger(self.config.hunger_depletion_rate)
        
        # 6. Handle starvation or regeneration
        if self.agent.is_starving:
            self.agent.take_damage(self.config.starvation_damage)
            reward += self.config.reward_starvation_damage * self.config.starvation_damage
        elif self.agent.hunger_ratio > 0.3:
            self.agent.regenerate(self.config.health_regen_rate)
        
        # 7. Reward shaping
        if self.agent.hunger_ratio < 0.3:
            reward += self.config.reward_low_hunger

        reward += self.config.reward_hunger_proportional * (1.0 - self.agent.hunger_ratio)

        if self.config.reward_food_visible_proximity != 0.0:
            nearest_food_distance = self._nearest_visible_food_distance()
            if nearest_food_distance is not None:
                max_visible_distance = 2 * self.config.observation_radius
                closeness = (max_visible_distance - nearest_food_distance + 1) / (max_visible_distance + 1)
                reward += self.config.reward_food_visible_proximity * closeness
        
        if self.agent.is_in_shelter and self._has_nearby_enemy(3):
            reward += self.config.reward_shelter_safety

        reward += self.config.reward_survival_tick
        
        # 8. Check death
        done = not self.agent.is_alive
        if done:
            reward += self.config.reward_death
            info['death'] = True
        
        # 9. Spawn new entities
        if self._rng.random() < self.config.food_spawn_rate:
            self._try_spawn_food()
        if self._rng.random() < self.config.enemy_spawn_rate:
            self._try_spawn_enemy()
        
        # 10. Remove inactive food
        self.food_items = [f for f in self.food_items if f.is_active]
        
        # 11. Update tick counter
        self.ticks += 1
        self.agent.tick()
        self.episode_reward += reward
        
        info['ticks'] = self.ticks
        info['health'] = self.agent.health
        info['hunger'] = self.agent.hunger
        info['episode_reward'] = self.episode_reward
        
        return reward, done, info
    
    def _has_nearby_enemy(self, radius: int) -> bool:
        """Return whether any enemy is within radius of the agent."""
        agent_pos = self.agent.position
        for enemy in self.enemies:
            if enemy.position.distance_to(agent_pos) <= radius:
                return True
        return False

    def _nearest_visible_food_distance(self) -> int | None:
        """Return the Manhattan distance to the nearest visible food tile."""
        agent_x, agent_y = self.agent.position.as_tuple()
        radius = self.config.observation_radius
        nearest_distance = None

        for food_x, food_y in self._food_positions:
            if abs(food_x - agent_x) > radius or abs(food_y - agent_y) > radius:
                continue

            distance = abs(food_x - agent_x) + abs(food_y - agent_y)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance

        return nearest_distance
    
    def get_state(self) -> WorldState:
        """Get an immutable snapshot of the current world state."""
        return WorldState(
            agent_position=self.agent.position.as_tuple(),
            agent_health=self.agent.health,
            agent_hunger=self.agent.hunger,
            agent_max_health=self.agent.max_health,
            agent_max_hunger=self.agent.max_hunger,
            agent_in_shelter=self.agent.is_in_shelter,
            agent_alive=self.agent.is_alive,
            enemies=[e.position.as_tuple() for e in self.enemies],
            food=[f.position.as_tuple() for f in self.food_items if f.is_active],
            shelters=[s.position.as_tuple() for s in self.shelters],
            ticks=self.ticks,
        )
    
    def get_observation(self) -> np.ndarray:
        """
        Generate the observation array for the RL agent.

        The observation is a multi-channel grid centered on the agent:
        - Channel 0: Enemy presence (binary)
        - Channel 1: Food presence (binary)
        - Channel 2: Shelter presence (binary)

        Plus scalar agent stats (health, hunger, in_shelter).

        Returns:
            Flattened observation array
        """
        radius = self.config.observation_radius
        obs_size = 2 * radius + 1
        num_channels = self.config.num_spatial_channels
        spatial_dim = num_channels * obs_size * obs_size

        agent_x, agent_y = self.agent.position.as_tuple()
        observation = np.zeros(spatial_dim + self.config.num_scalars, dtype=np.float32)
        spatial = observation[:spatial_dim].reshape(num_channels, obs_size, obs_size)
        
        # Fill observation grid
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                world_x = agent_x + dx
                world_y = agent_y + dy
                
                obs_x = dx + radius
                obs_y = dy + radius
                
                # Check bounds
                if not (0 <= world_x < self.config.width and 
                        0 <= world_y < self.config.height):
                    continue  # Out of bounds - leave as zeros
                
                pos = (world_x, world_y)
                
                # Channel 0: Enemies (normalized by max expected)
                if pos in self._enemy_positions:
                    spatial[0, obs_y, obs_x] = 1.0
                
                # Channel 1: Food
                if pos in self._food_positions:
                    spatial[1, obs_y, obs_x] = 1.0
                
                # Channel 2: Shelters
                if pos in self._shelter_positions:
                    spatial[2, obs_y, obs_x] = 1.0

        observation[spatial_dim:] = (
            self.agent.health / self.agent.max_health,
            self.agent.hunger / self.agent.max_hunger,
            1.0 if self.agent.is_in_shelter else 0.0,
        )
        return observation

    @property
    def observation_size(self) -> int:
        """Calculate the size of the observation vector."""
        return self.config.observation_size
    
    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return len(Direction)
