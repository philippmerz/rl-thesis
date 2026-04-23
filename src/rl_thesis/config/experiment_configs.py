"""Named experiment configurations.

Each entry defines one reproducible experiment. A config dict contains
WorldConfig reward-field overrides (``reward_*``, ``proximity_delta``,
``movement_cost``, etc.) plus an optional top-level ``frame_stack``
entry that overrides :class:`DQNConfig.frame_stack`.

Nothing else is overridable from a config. All other training
hyperparameters (learning rate, buffer size, tau, epsilon schedule,
n-step horizon, total timesteps) are fixed at the :class:`DQNConfig`
defaults across every experiment so that the ablation has a single
axis of variation in DQN space (``frame_stack``) and an axis of
variation in reward space.

Reward feasibility constraints
==============================

Environment constants (from WorldConfig defaults):
    u_max = 100       max hunger
    h_max = 100       max health
    c     = 0.5       movement hunger cost
    u_dot = 0.2       hunger depletion per tick
    h_s   = 0.5       starvation damage per tick
    d_e   = 15        enemy damage per hit
    f     = 20        food nutrition value
    D     = 14        max visible distance (2 * obs_radius)

Derived:
    T_starve_still = u_max / u_dot                  = 500 ticks
    T_starve_move  = u_max / (u_dot + c)            ~ 143 ticks
    T_death        = h_max / h_s                     = 200 ticks
    T_passive      = T_starve_still + T_death        = 700 ticks

Notation: |w| denotes absolute value of a (negative) weight.

C1  Anti-stasis: one step toward food must yield more proximity
    reward than the hunger penalty from the movement cost.

        w_prox > |w_hprop| * c * (D + 1) / u_max

    baseline:  0.1  > 0.3 * 0.5 * 15 / 100 = 0.0225  OK

C2  Anti-hovering: max proximity reward (at distance 0) must not
    exceed the average hunger penalty per tick.

        w_prox < |w_hprop| / 2

    baseline:  0.1  < 0.3 / 2 = 0.15  OK

C3  Anti-passivity: the total tick reward over a passive episode
    must not exceed the accumulated penalties.

        w_tick < (|w_hprop| * Sigma_h + |w_sdm| * h_s * T_death + |w_death|)
                 / T_passive

    baseline:  0.0  < (0.3 * 449.5 + 1.0 * 100 + 10) / 700 = 0.35  OK

C4  Enemy flee: the damage penalty from one hit must exceed the
    hunger penalty from one evasive step.

        |w_edm| * d_e > |w_hprop| * c / u_max

    baseline:  0.1 * 15 = 1.5 > 0.3 * 0.5 / 100 = 0.0015  OK

C5  Starvation escalation: per-tick penalty when starving must
    exceed the max hunger-proportional penalty, creating urgency.

        |w_sdm| * h_s > |w_hprop|

    baseline:  1.0 * 0.5 = 0.5 > 0.3  OK

C6  Terminal signal: the death penalty must be the worst single
    event, exceeding the best single event (eating).

        |w_death| > w_food

    baseline:  10 > 5  OK

C7  Food trip: reaching food at expected distance d_bar must be
    worth the movement cost. Always satisfied when C1 holds and
    food restores more hunger than the trip costs:

        f > d_bar * c

    With ~36 food on 64x64, d_bar ~ 5:  20 > 5 * 0.5 = 2.5  OK
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

    # ------------------------------------------------------------------
    # Diagnostic configs (each isolates one failure mode)
    # Used for the Failure Modes section; not part of the main matrix.
    # ------------------------------------------------------------------

    # C1 violation: proximity below movement break-even. Predicts
    # suicide-by-enemy: the agent cannot profitably move toward food,
    # so the cheapest escape from the accumulating hunger penalty is
    # to walk into an enemy and end the episode.
    "weak_proximity": {
        "reward_food_visible_proximity": 0.02,
    },

    # Zero movement cost. If the agent still oscillates, oscillation is
    # a training-dynamics artifact rather than a reward-rational strategy.
    "free_movement": {
        "movement_cost": 0.0,
    },

    # C6 violation: no death penalty. Tests whether the per-step
    # signals alone suffice to learn long-horizon survival.
    "no_death": {
        "reward_death": 0.0,
    },

    # Sparse: only the terminal death penalty is non-zero. Tests the
    # credit-assignment limit of single-signal RL over a 1000-step
    # episode horizon.
    "death_only": {
        "reward_food_eaten": 0.0,
        "reward_enemy_damage_taken": 0.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_food_visible_proximity": 0.0,
        "reward_shelter_safety": 0.0,
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
