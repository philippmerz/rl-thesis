"""Configuration module for the survival environment and DQN training."""
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


@dataclass
class WorldConfig:
    """Configuration for the world/environment."""

    initial_seed: int = 42                # For reproducibility
    max_steps: int = 1000                 # Max steps before truncation

    # World dimensions
    width: int = 64
    height: int = 64

    @property
    def num_cells(self) -> int:
        """Total number of cells in the world."""
        return self.width * self.height
    
    # Terrain parameters (uniform across the map)
    food_density: float = 0.15            # Probability of food spawning per tick per cell
    shelter_density: float = 0.01         # Probability of shelter per cell during generation
    enemy_density: float = 0.01           # Base enemy density for spawn checks
    food_value: float = 20.0              # Nutrition value of all food items
    enemy_damage: float = 15.0            # Damage dealt by all enemies
    movement_cost: float = 0.5            # Hunger cost for moving
    
    # Agent settings
    max_health: float = 100.0
    max_hunger: float = 100.0
    hunger_depletion_rate: float = 0.2    # Hunger lost per tick
    health_regen_rate: float = 0.5        # Health gained per tick when healthy
    starvation_damage: float = 0.5        # Health lost per tick when starving
    
    # Entity settings
    max_enemy_density: float = 0.002        # Max enemies per cell (~8 enemies on 64x64 grid)
    max_food_density: float = 0.009         # Max food per cell (~30 food items on 64x64 grid)

    @property
    def max_enemies(self) -> int:
        """Maximum number of enemies allowed in the world."""
        return int(self.num_cells * self.max_enemy_density)
    
    @property
    def max_food(self) -> int:
        """Maximum number of food items allowed in the world."""
        return int(self.num_cells * self.max_food_density)

    enemy_speed: float = 0.5              # Probability of enemies moving each tick
    enemy_vision_range: int = 3           # Manhattan distance within which enemies can see the agent
    shelter_protection: float = 1.0       # Damage reduction in shelter (0-1); 1.0 = full protection
    
    # Initial spawn fractions (fraction of max capacity present at episode start)
    initial_food_fraction: float = 0.5    # Start with half the max food
    initial_enemy_fraction: float = 0.33  # Start with a third of max enemies

    # Spawn rates (per tick)
    # Effective spawn probability per tick: spawn_rate*density.
    food_spawn_rate: float = 0.05
    enemy_spawn_rate: float = 0.01
    
    # Observation settings
    observation_radius: int = 7           # Agent's vision radius
    num_spatial_channels: int = 3         # enemy, food, shelter channels
    num_agent_stats: int = 3              # health, hunger, in_shelter

    @property
    def observation_grid_size(self) -> int:
        """Side length of the spatial observation grid."""
        return 2 * self.observation_radius + 1

    @property
    def num_scalars(self) -> int:
        """Total number of scalar (non-spatial) observation features."""
        return self.num_agent_stats

    @property
    def observation_size(self) -> int:
        """Total flat observation vector length."""
        g = self.observation_grid_size
        return self.num_spatial_channels * g * g + self.num_scalars

    # Reward shaping (satisfies constraints C1-C7, see reward_configs.py)
    reward_food_eaten: float = 5.0
    reward_survival_tick: float = 0.0
    reward_death: float = -10.0
    reward_enemy_damage_taken: float = -0.1
    reward_starvation_damage: float = -1.0
    reward_low_hunger: float = 0.0
    reward_hunger_proportional: float = -0.3
    reward_food_visible_proximity: float = 0.1
    reward_shelter_safety: float = 0.1

@dataclass
class HumanHeuristicConfig:
    """Configuration for the scripted heuristic agent."""
    hunger_threshold: float = 0.5         # Hunger ratio below which to forage
    flee_radius: int = 5                 # Manhattan distance to trigger fleeing from enemies

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Window settings
    cell_size: int = 10                   # Pixels per grid cell
    fps: int = 30                         # Target frames per second
    
    # Colors (RGB)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    agent_color: Tuple[int, int, int] = (0, 191, 255)       # Deep sky blue
    enemy_color: Tuple[int, int, int] = (255, 69, 0)        # Red-orange
    food_color: Tuple[int, int, int] = (50, 205, 50)        # Lime green
    shelter_color: Tuple[int, int, int] = (169, 169, 169)   # Dark gray
    
    # HUD settings
    hud_height: int = 60                  # Pixels for the status bar
    health_bar_color: Tuple[int, int, int] = (220, 20, 60)  # Crimson
    hunger_bar_color: Tuple[int, int, int] = (255, 165, 0)  # Orange
    
    tick_duration_ms: int = 10           # For human observer

@dataclass
class DQNConfig:
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    
    merge_hidden: int = 256
    head_hidden: int = 128
    
    cnn_channels: Tuple[int, ...] = (32, 64, 64)
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    weight_decay: float = 1e-4
    n_step: int = 3                       # Multi-step returns horizon

    # Replay buffer
    buffer_size: int = 500_000
    min_buffer_size: int = 10_000         # Wait for samples before training

    # Target network (Polyak averaging per training step)
    tau: float = 0.005

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000

    # Training duration
    total_timesteps: int = 1_000_000

    # Checkpointing
    checkpoint_freq: int = 10_000
    checkpoint_keep_stride: int = 10      # 10 keeps every 10th periodic checkpoint
    eval_freq: int = 5_000
    eval_episodes: int = 10
    
    # Device
    device: str = "auto"  # "auto" picks cuda > mps > cpu; override with "cuda"/"mps"/"cpu"
