"""Named reward configurations for the experiment grid.

Each config is a dict of WorldConfig field overrides.
Only reward-related fields are changed; world mechanics stay constant.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Any

from rl_thesis.config.config import WorldConfig


REWARD_CONFIGS: Dict[str, Dict[str, float]] = {
    # Reference configuration: current hand-tuned defaults.
    "baseline": {},

    # Pure survival signal: constant tick reward + terminal death penalty.
    # The agent receives no information about food, damage, or shelters.
    # It must discover that eating prevents starvation purely from the
    # tick reward disappearing on death.
    "survival_only": {
        "reward_food_eaten": 0.0,
        "reward_survival_tick": 1.0,
        "reward_death": -100.0,
        "reward_damage_taken": 0.0,
        "reward_low_hunger": 0.0,
        "reward_shelter_safety": 0.0,
    },

    # Aggressive foraging: large food bonus, weak penalties.
    # Expected behavior: risk-taking food collection, high damage taken.
    "foraging": {
        "reward_food_eaten": 30.0,
        "reward_survival_tick": 0.0,
        "reward_death": -20.0,
        "reward_damage_taken": -1.0,
        "reward_low_hunger": -1.0,
        "reward_shelter_safety": 0.0,
    },

    # Risk-averse: heavy penalties for damage and death.
    # Expected behavior: shelter-seeking, cautious movement.
    "cautious": {
        "reward_food_eaten": 10.0,
        "reward_survival_tick": 0.0,
        "reward_death": -100.0,
        "reward_damage_taken": -5.0,
        "reward_low_hunger": -0.5,
        "reward_shelter_safety": 1.0,
    },

    # Shelter camping: strong shelter incentive + tick reward.
    # Expected behavior: stay in shelter, minimal foraging.
    "shelter": {
        "reward_food_eaten": 5.0,
        "reward_survival_tick": 0.5,
        "reward_death": -50.0,
        "reward_damage_taken": -3.0,
        "reward_low_hunger": -0.3,
        "reward_shelter_safety": 3.0,
    },

    # All signals active at moderate weights.
    # Expected behavior: mixed strategy.
    "balanced": {
        "reward_food_eaten": 10.0,
        "reward_survival_tick": 0.2,
        "reward_death": -50.0,
        "reward_damage_taken": -2.0,
        "reward_low_hunger": -0.5,
        "reward_shelter_safety": 0.5,
    },

    # Minimal shaping: only food and death.
    # No intermediate feedback about damage, hunger, or shelters.
    "sparse": {
        "reward_food_eaten": 15.0,
        "reward_survival_tick": 0.0,
        "reward_death": -50.0,
        "reward_damage_taken": 0.0,
        "reward_low_hunger": 0.0,
        "reward_shelter_safety": 0.0,
    },
}


def make_world_config(config_name: str, seed: int = 42) -> WorldConfig:
    """Create a WorldConfig with the named reward configuration applied."""
    if config_name not in REWARD_CONFIGS:
        available = ", ".join(sorted(REWARD_CONFIGS))
        raise ValueError(f"Unknown reward config '{config_name}'. Available: {available}")

    overrides = REWARD_CONFIGS[config_name]
    return replace(WorldConfig(initial_seed=seed), **overrides)


def get_config_names() -> list[str]:
    return list(REWARD_CONFIGS.keys())


def describe_config(config_name: str) -> Dict[str, Any]:
    """Return the full reward weights for a named config (with defaults filled in)."""
    wc = make_world_config(config_name)
    return {
        "reward_food_eaten": wc.reward_food_eaten,
        "reward_survival_tick": wc.reward_survival_tick,
        "reward_death": wc.reward_death,
        "reward_damage_taken": wc.reward_damage_taken,
        "reward_low_hunger": wc.reward_low_hunger,
        "reward_shelter_safety": wc.reward_shelter_safety,
    }
