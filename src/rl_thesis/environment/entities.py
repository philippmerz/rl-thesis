"""
Entity definitions for the survival environment.

All entities in the world (Agent, Enemies, Food, Shelters) are defined here
with their properties and behaviors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum, auto
import random


class Direction(Enum):
    """Movement directions for entities."""
    NONE = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    
    def to_delta(self) -> Tuple[int, int]:
        """Convert direction to (dx, dy) offset."""
        deltas = {
            Direction.NONE: (0, 0),
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }
        return deltas[self]
    
    @classmethod
    def from_action(cls, action: int) -> Direction:
        """Convert action index to direction by enum order (0=none, 1=up, 2=down, 3=left, 4=right)."""
        return list(cls)[action]


@dataclass
class Position:
    """2D position with boundary checking."""
    x: int
    y: int
    
    def move(self, direction: Direction, width: int, height: int) -> Position:
        """Return new position after moving, clamped to world bounds."""
        dx, dy = direction.to_delta()
        new_x = max(0, min(width - 1, self.x + dx))
        new_y = max(0, min(height - 1, self.y + dy))
        return Position(new_x, new_y)
    
    def distance_to(self, other: Position) -> float:
        """Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def as_tuple(self) -> Tuple[int, int]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class Agent:
    """
    The player-controlled agent trying to survive.
    
    The agent must eat food to maintain hunger, avoid enemies,
    and use shelters for protection. Health regenerates when fed,
    but depletes from starvation or enemy attacks.
    """
    position: Position
    health: float
    hunger: float
    max_health: float
    max_hunger: float
    is_alive: bool = True
    is_in_shelter: bool = False
    ticks_survived: int = 0
    food_eaten: int = 0
    damage_taken: float = 0.0
    
    @classmethod
    def create(cls, x: int, y: int, max_health: float, max_hunger: float) -> Agent:
        """Factory method to create a new agent at full health/hunger."""
        return cls(
            position=Position(x, y),
            health=max_health,
            hunger=max_hunger,
            max_health=max_health,
            max_hunger=max_hunger,
        )
    
    def move(self, direction: Direction, width: int, height: int) -> None:
        """Move the agent in the specified direction."""
        self.position = self.position.move(direction, width, height)
    
    def eat(self, food_value: float) -> None:
        """Consume food and restore hunger."""
        self.hunger = min(self.max_hunger, self.hunger + food_value)
        self.food_eaten += 1
    
    def take_damage(self, damage: float, reduction: float = 0.0) -> None:
        """Take damage, potentially reduced by shelter protection."""
        actual_damage = damage * (1.0 - reduction)
        self.health -= actual_damage
        self.damage_taken += actual_damage
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
    
    def regenerate(self, amount: float) -> None:
        """Regenerate health when well-fed."""
        self.health = min(self.max_health, self.health + amount)
    
    def deplete_hunger(self, amount: float) -> None:
        """Reduce hunger over time."""
        self.hunger = max(0, self.hunger - amount)
    
    def tick(self) -> None:
        """Increment survival counter."""
        self.ticks_survived += 1
    
    @property
    def health_ratio(self) -> float:
        """Current health as fraction of maximum."""
        return self.health / self.max_health
    
    @property
    def hunger_ratio(self) -> float:
        """Current hunger as fraction of maximum."""
        return self.hunger / self.max_hunger
    
    @property
    def is_starving(self) -> bool:
        """True if hunger is depleted."""
        return self.hunger <= 0


@dataclass
class Enemy:
    """
    A hostile entity that moves toward and attacks the agent.
    
    Enemies spawn in the world based on enemy density and 
    actively hunt the agent when within vision range.
    """
    position: Position
    damage: float
    speed: float  # Probability of moving each tick
    vision_range: int
    
    @classmethod
    def create(cls, x: int, y: int, damage: float, speed: float, vision_range: int) -> Enemy:
        """Factory method to create an enemy."""
        return cls(
            position=Position(x, y),
            damage=damage,
            speed=speed,
            vision_range=vision_range,
        )
    
    def update(self, agent_position: Position, width: int, height: int,
               shelter_positions: set = None,
               occupied_positions: set = None,
               rng: random.Random | None = None) -> None:
        """Move toward agent if in range, otherwise random movement.
        
        Enemies cannot enter shelter tiles or tiles already occupied by
        another enemy.
        """
        _rng = rng or random  # fall back to global if no instance RNG
        if _rng.random() > self.speed:
            return  # Skip movement this tick
        
        distance = self.position.distance_to(agent_position)
        
        if distance <= self.vision_range:
            # Move toward agent
            direction = self._get_direction_toward(agent_position)
        else:
            # Random movement
            direction = _rng.choice(list(Direction))
        
        new_pos = self.position.move(direction, width, height)
        new_tuple = new_pos.as_tuple()
        # Enemies cannot enter shelters or already-occupied tiles
        if shelter_positions and new_tuple in shelter_positions:
            return
        if occupied_positions and new_tuple in occupied_positions:
            return
        self.position = new_pos
    
    def _get_direction_toward(self, target: Position) -> Direction:
        """Get the best direction to move toward target."""
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        
        # Prefer the axis with greater distance
        if abs(dx) >= abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            return Direction.DOWN if dy > 0 else Direction.UP
    
    def can_attack(self, agent_position: Position) -> bool:
        """Check if enemy is adjacent to agent."""
        return self.position.distance_to(agent_position) <= 1


@dataclass
class Food:
    """
    A consumable item that restores the agent's hunger.
    
    Food spawns based on food density and remains
    until consumed by the agent.
    """
    position: Position
    nutrition_value: float
    is_active: bool = True
    
    @classmethod
    def create(cls, x: int, y: int, nutrition_value: float) -> Food:
        """Factory method to create food."""
        return cls(
            position=Position(x, y),
            nutrition_value=nutrition_value,
        )
    
    def consume(self) -> float:
        """Mark as consumed and return nutrition value."""
        self.is_active = False
        return self.nutrition_value


@dataclass 
class Shelter:
    """
    When the agent occupies a shelter tile, incoming damage
    is reduced by the shelter's protection factor.
    """
    position: Position
    protection: float  # Damage reduction (0-1)
    
    @classmethod
    def create(cls, x: int, y: int, protection: float) -> Shelter:
        """Factory method to create a shelter."""
        return cls(
            position=Position(x, y),
            protection=protection,
        )
