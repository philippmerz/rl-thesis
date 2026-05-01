"""Named experiment configurations for the Option A ablation.

Each entry defines one reproducible experiment. A config dict contains
:class:`WorldConfig` reward-field overrides (``reward_*``,
``proximity_delta``, ``movement_cost``, etc.) plus an optional top-level
``frame_stack`` entry that overrides :class:`DQNConfig.frame_stack`.

Nothing else is overridable from a config. All other training
hyperparameters (learning rate, buffer size, tau, epsilon schedule,
n-step horizon, total timesteps) are fixed at the :class:`DQNConfig`
defaults across every experiment. The ablation has a single axis of
variation in observation space (``frame_stack`` in {1, 4}) crossed
with a reward-shape axis defined by the three reward configurations
(``baseline``, ``absolute_proximity``, ``engineered_v5``).

The reward-design heuristics H1-H4 and the derivations behind the
configs below are given in the thesis (Section "Reward Configurations").
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Any

from rl_thesis.config.config import WorldConfig, DQNConfig


EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Main ablation matrix (reward shape x observation)
    # Each reward row has a single-frame and a 4-frame variant.
    # ------------------------------------------------------------------

    # Baseline: default reward weights. Produces all three failure modes
    # (oscillation, hovering, suicidal enemy-seeking) in a single run.
    "baseline": {},
    "baseline_fs": {
        "frame_stack": 4,
    },

    # Absolute-closeness proximity (instead of per-step change form).
    # Produces food hovering: the agent camps next to food without
    # eating because the accumulated proximity stream dominates the
    # one-time food reward.
    "absolute_proximity": {
        "proximity_delta": False,
    },
    "absolute_proximity_fs": {
        "proximity_delta": False,
        "frame_stack": 4,
    },

    # Minimal (E5): three closeness-change proximity rewards gated on
    # hunger, plus the terminal death penalty. Every other reward
    # component is zeroed. Gating:
    #   hungry   (u < 0.5): food proximity on,    shelter off
    #   well-fed (u >= 0.5): shelter proximity on, food off
    #   enemy proximity: always on.
    "engineered_v5": {
        "reward_food_eaten": 0.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
    },
    "engineered_v5_fs": {
        "reward_food_eaten": 0.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "frame_stack": 4,
    },

    # Episode-cap robustness check: identical to engineered_v5_fs but
    # with the per-episode tick cap raised from 1,000 to 50,000 both
    # during training and at benchmark. Used to verify that the
    # parity-not-superiority verdict for the headline cell is not an
    # artefact of training under truncation.
    "engineered_v5_fs_cap50k": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "frame_stack": 4,
    },

    # ------------------------------------------------------------------
    # Illustrative diagnostic (single-seed qualitative demo, not part of
    # the main ablation). Kept to show one heuristic violation has the
    # empirical consequence derived analytically.
    # ------------------------------------------------------------------

    # H1 violation: food-proximity weight below the movement-cost
    # break-even (0.02 < 0.0225). The analytical prediction is the
    # suicide failure mode: the agent cannot profitably move toward
    # food, so the cheapest escape from the accumulating hunger
    # penalty is to walk into an enemy and end the episode.
    "weak_proximity": {
        "reward_food_visible_proximity": 0.02,
    },
}


def _get_config(config_name: str) -> Dict[str, Any]:
    if config_name not in EXPERIMENT_CONFIGS:
        available = ", ".join(sorted(EXPERIMENT_CONFIGS))
        raise ValueError(f"Unknown experiment config '{config_name}'. Available: {available}")
    return EXPERIMENT_CONFIGS[config_name]


def make_world_config(config_name: str, seed: int = 42) -> WorldConfig:
    """Create a WorldConfig with the named experiment's reward overrides applied."""
    raw = _get_config(config_name)
    world_overrides = {k: v for k, v in raw.items() if k != "frame_stack"}
    return replace(WorldConfig(initial_seed=seed), **world_overrides)


def make_dqn_config(config_name: str, **cli_overrides: Any) -> DQNConfig:
    """Create a DQNConfig using the named experiment's ``frame_stack`` and defaults.

    CLI overrides (``total_timesteps``, ``eval_episodes``) take precedence.
    """
    raw = _get_config(config_name)
    frame_stack = raw.get("frame_stack", 1)
    return replace(DQNConfig(), frame_stack=frame_stack, **cli_overrides)


def describe_config(config_name: str) -> Dict[str, Any]:
    """Return the explicit overrides defined for this config.

    Shows only fields this config actually changes from the defaults,
    split into ``world`` (WorldConfig overrides) and ``dqn``
    (``frame_stack`` override, if any).
    """
    raw = _get_config(config_name)
    world = {k: v for k, v in raw.items() if k != "frame_stack"}
    dqn = {"frame_stack": raw["frame_stack"]} if "frame_stack" in raw else {}
    return {"world": world, "dqn": dqn}
