"""Named experiment configurations.

Each entry defines one reproducible experiment. A config may override:

- WorldConfig fields: reward weights, world mechanics (movement cost,
  enemy density, spawn rates), observation parameters.
- DQNConfig fields: hyperparameters such as ``frame_stack``, ``gamma``,
  ``lr_schedule``, ``head_reset_freq``, ``epsilon_cycle_steps``, etc.

The two override sets are separated by an optional ``_dqn`` key: any
top-level key applies to WorldConfig, and entries under ``_dqn`` apply
to DQNConfig. Use :func:`make_world_config` and :func:`make_dqn_config`
to materialize both.

Reward Feasibility Constraints
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

    forage:  0.1  > 0.3 * 0.5 * 15 / 100 = 0.0225  OK

C2  Anti-hovering: max proximity reward (at distance 0) must not
    exceed the average hunger penalty per tick.

        w_prox < |w_hprop| / 2

    forage:  0.1  < 0.3 / 2 = 0.15  OK

C3  Anti-passivity: the total tick reward over a passive episode
    must not exceed the accumulated penalties.

        w_tick < (|w_hprop| * Sigma_h + |w_sdm| * h_s * T_death + |w_death|)
                 / T_passive

    where Sigma_h ~ 449.5 for stationary depletion over 700 ticks.

    forage:  0.0  < (0.3 * 449.5 + 1.0 * 100 + 10) / 700 = 0.35  OK

C4  Enemy flee: the damage penalty from one hit must exceed the
    hunger penalty from one evasive step.

        |w_edm| * d_e > |w_hprop| * c / u_max

    forage:  0.1 * 15 = 1.5 > 0.3 * 0.5 / 100 = 0.0015  OK

C5  Starvation escalation: per-tick penalty when starving must
    exceed the max hunger-proportional penalty, creating urgency.

        |w_sdm| * h_s > |w_hprop|

    forage:  1.0 * 0.5 = 0.5 > 0.3  OK

C6  Terminal signal: the death penalty must be the worst single
    event, exceeding the best single event (eating).

        |w_death| > w_food

    forage:  10 > 5  OK

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

# Top-level keys are WorldConfig field overrides. An optional "_dqn" key
# holds DQNConfig overrides for the experiment, keeping reward shaping,
# world mechanics, and agent hyperparameters in one place per experiment.
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Feasible baseline: all constraints C1-C7 satisfied.
    # Proximity rewards use the per-step closeness-change form.
    "baseline": {},

    # Original baseline with absolute proximity reward.
    # Produces food hovering: the accumulated proximity stream
    # (~36.6 over an episode) dominates the one-time food reward (+5).
    "absolute_proximity": {
        "proximity_delta": False,
    },

    # Deliberate C1 violation (anti-stasis). Proximity below the
    # movement break-even threshold: w_prox=0.02 < 0.0225.
    # Predicted failure: agent learns to stay still because moving
    # toward food costs more in hunger penalty than it gains in
    # proximity reward.
    "weak_proximity": {
        "reward_food_visible_proximity": 0.02,
    },

    # Diagnostic: baseline rewards with zero movement cost.
    # If the agent still oscillates, oscillation is a training
    # artifact, not a reward-rational strategy.
    "free_movement": {
        "movement_cost": 0.0,
    },

    # C6 violation (terminal signal). No death penalty.
    # Tests whether per-step consequences alone (hunger, damage,
    # starvation) are sufficient to learn survival, or whether
    # the terminal signal is necessary for TD bootstrapping to
    # propagate long-horizon danger.
    "no_death": {
        "reward_death": 0.0,
    },

    # Death-only: the sole reward signal is the terminal death
    # penalty. All per-step shaping removed. Tests whether a
    # single sparse signal can drive learning over a 1000-step
    # horizon, or whether the credit assignment gap is too wide.
    "death_only": {
        "reward_food_eaten": 0.0,
        "reward_enemy_damage_taken": 0.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_food_visible_proximity": 0.0,
        "reward_shelter_safety": 0.0,
    },

    # Engineered: analytically designed to produce all three
    # heuristic behaviors (flee, forage, shelter) as emergent
    # Q-value optima.
    #
    # Key changes:
    # 1. Remove hunger_proportional: eliminates the structural C1
    #    incompatibility (PV of movement cost at gamma=0.99 exceeds
    #    any feasible proximity reward by factor 22).
    # 2. Enable low_hunger threshold penalty (-0.5 at <30% hunger):
    #    creates urgency without the proportional-to-movement-cost
    #    coupling. Flat penalty means movement doesn't amplify cost.
    # 3. Stronger food reward (8.0) and proximity (0.15): compensate
    #    for removed hunger gradient; PV of eating at distance 5 is
    #    ~+22 from delayed threshold penalty alone.
    # 4. Enemy proximity (closeness-change form): flee gradient before contact,
    #    5x food proximity gradient. Only when not in shelter.
    # 5. Stronger damage penalty: -7.5 per hit vs +8.0 per food.
    "engineered": {
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": -0.5,
        "reward_food_eaten": 8.0,
        "reward_food_visible_proximity": 0.15,
        "reward_enemy_damage_taken": -0.5,
        "reward_enemy_proximity": -0.5,
    },

    # Engineered v4: symmetric proximity gradients.
    #
    # V1-V3 all plateau at ~620-680 survival because the agent never
    # learns to navigate TO shelter when well-fed. The shelter_safety
    # reward is binary (in/out), giving no gradient for approach.
    #
    # Fix: add shelter proximity (closeness-change form) that mirrors food proximity.
    # When hungry (< 50%): food proximity active, shelter proximity off.
    # When well-fed (>= 50%): shelter proximity active, food proximity off.
    # This creates symmetric approach gradients for both behavioral modes.
    #
    # Also adds small survival_tick (0.05) to directly reward each tick
    # alive, making conservation strategies Q-value positive.

    # Engineered v5: minimal reward set.
    #
    # Only three proximity gradients + death. No food_eaten, no
    # starvation_damage, no hunger penalties, no shelter_safety.
    # The hypothesis: fewer signals reduce Q-network confusion.
    # The three closeness-change proximity rewards directly encode the
    # behavioral switch (hungry->food, well-fed->shelter, enemy->flee).
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

    # V5 with 4-frame stacking for temporal context.
    #
    # The agent sees a single snapshot and cannot observe movement
    # direction, enemy approach/retreat, or its own trajectory.
    # Stacking the last 4 spatial grids (12 channels instead of 3)
    # gives the CNN access to motion patterns.
    #
    # Buffer reduced to 250K (stacked obs is 4x larger in memory).
    # 5M steps with tuned hyperparameters as in v5_long.
    "engineered_v5_fs": {
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
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 5_000_000,
            "epsilon_decay_steps": 1_000_000,
            "buffer_size": 250_000,
            "tau": 0.002,
        },
    },

    # V5_fs + small food_eaten: backup experiment.
    #
    # Prior food_eaten experiments (2.0, 5.0) destabilized V5's flee
    # behavior because the food reward competed with the enemy proximity
    # gradient (-0.5). At 0.3, a successful foraging trip nets ~+0.35
    # total (+0.05 proximity + 0.30 food eaten), while enemy proximity
    # cost per approach step is ~0.033. Ratio preserves flee priority.
    # epsilon_end=0.05 for policy-collapse prevention.
    "engineered_v6_fs_food": {
        "reward_food_eaten": 0.3,
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
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 5_000_000,
            "epsilon_decay_steps": 1_000_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
        },
    },

    # V6_fs_food with constant LR to test plasticity hypothesis.
    #
    # V6_fs_food peaked at 860 survival (step 940K) then collapsed by
    # 3.4M. The collapse correlates with OneCycle LR decay: by step 2M
    # the LR has dropped to ~1e-5, too small to correct the replay buffer
    # distribution shift as foraging transitions age out.
    #
    # Fix: constant LR at 1e-4 maintains plasticity throughout training.
    # Shorter training (2M steps) captures the peak window without the
    # harmful late-training phase. Epsilon decay over 500K gives more
    # greedy data earlier to test whether the policy holds.
    "engineered_v7_fs": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.3,
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
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 2_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
        },
    },

    # V8_fs_cycle: V7_fs + cyclical epsilon.
    #
    # V7_fs peaks early then drifts because the greedy policy biases
    # the replay buffer toward its own behavior. Cyclical epsilon
    # resets exploration to 0.5 every 500K steps, refilling the buffer
    # with diverse transitions and stress-testing the current policy.
    "engineered_v8_fs_cycle": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.3,
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
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 2_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "epsilon_cycle_steps": 500_000,
            "epsilon_cycle_peak": 0.5,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
        },
    },

    # V8_fs_reset: V7_fs + periodic Dueling head resets (Nikishin 2022).
    #
    # Directly addresses primacy bias and plasticity loss by reinitializing
    # the last FC layers every 500K steps while keeping the CNN encoder
    # and replay buffer. The network re-fits from accumulated experience
    # rather than drifting with buffer distribution shift.
    "engineered_v8_fs_reset": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.3,
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
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 2_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
            "head_reset_freq": 500_000,
        },
    },

    # V8_fs_strong: V7_fs + stronger foraging signals.
    #
    # V7_fs peaks at 3.85 food/episode; heuristic baseline 1000-tick
    # survival would require ~9. Doubling food signals (food_eaten 0.3->0.5,
    # food_proximity 0.15->0.3) pushes foraging harder while keeping the
    # enemy proximity ratio favorable for flee (enemy 0.5 vs food 0.5
    # per trip: 1.0x ratio, was 3.3x in V7_fs). Risk: destabilization
    # like V6 (food=5.0) but at 10x smaller magnitude.
    "engineered_v8_fs_strong": {
        "reward_food_eaten": 0.5,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.3,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 2_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
        },
    },

    # V9_fs_strong_reset: stronger foraging signals (V8_fs_strong) + head resets.
    # V8_fs_strong alone underperformed (2/3 seeds below heuristic), likely
    # because the stronger signals amplified the collapse trajectory. Head
    # resets may rescue the strong-signal regime by periodically restoring
    # plasticity before the policy over-commits to foraging into danger.
    "engineered_v9_fs_strong_reset": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.5,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.3,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 2_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
            "head_reset_freq": 500_000,
        },
    },

    # V9_fs_strong_reset extended to 10M steps. Tests whether the
    # best-performing config keeps improving past 2M when plasticity
    # is actively preserved (constant LR) and actively restored
    # (periodic head resets). Previous long runs (V5_fs at 5M,
    # V6_fs_food at 5M) showed no late improvement, but they used
    # OneCycle LR which decays to ~1e-5 by 2M, confounding "no late
    # improvement" with "LR too low to update." This run controls
    # for that.
    "engineered_v9_fs_strong_reset_long": {
        "max_steps": 50_000,
        "reward_food_eaten": 0.5,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.3,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": 0.0,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "_dqn": {
            "frame_stack": 4,
            "total_timesteps": 10_000_000,
            "epsilon_decay_steps": 500_000,
            "epsilon_end": 0.05,
            "buffer_size": 250_000,
            "tau": 0.002,
            "lr_schedule": "constant",
            "head_reset_freq": 500_000,
        },
    },

    # V5 + moderate food_eaten reward (0.3).
    #
    # V5 shelter-camps because foraging is reward-invisible: the
    # proximity deltas yield +0.05 per trip, 20-200x below TD noise.
    # food_eaten at 2.0-5.0 overwhelmed the flee gradient (-0.5).
    # At 0.3, a clean foraging trip nets +0.35 total, detectable
    # above noise but still dominated locally by enemy proximity.
    "engineered_v6": {
        "reward_food_eaten": 0.3,
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
        "_dqn": {
            "total_timesteps": 5_000_000,
            "epsilon_decay_steps": 1_000_000,
            "buffer_size": 1_000_000,
            "tau": 0.002,
        },
    },

    "engineered_v4": {
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": -0.5,
        "low_hunger_threshold": 0.5,
        "reward_food_eaten": 8.0,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": -0.5,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.2,
        "reward_survival_tick": 0.05,
    },
}


def _split_overrides(config_name: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a config entry into WorldConfig overrides and DQNConfig overrides."""
    if config_name not in EXPERIMENT_CONFIGS:
        available = ", ".join(sorted(EXPERIMENT_CONFIGS))
        raise ValueError(f"Unknown experiment config '{config_name}'. Available: {available}")
    raw = EXPERIMENT_CONFIGS[config_name]
    dqn_overrides = raw.get("_dqn", {})
    world_overrides = {k: v for k, v in raw.items() if k != "_dqn"}
    return world_overrides, dqn_overrides


def make_world_config(config_name: str, seed: int = 42) -> WorldConfig:
    """Create a WorldConfig with the named experiment's world overrides applied."""
    world_overrides, _ = _split_overrides(config_name)
    return replace(WorldConfig(initial_seed=seed), **world_overrides)


def make_dqn_config(config_name: str, **cli_overrides: Any) -> DQNConfig:
    """Create a DQNConfig with the named experiment's DQN overrides and CLI overrides.

    CLI overrides take precedence over config-level ``_dqn`` overrides.
    """
    _, dqn_overrides = _split_overrides(config_name)
    merged = {**dqn_overrides, **cli_overrides}
    return replace(DQNConfig(), **merged) if merged else DQNConfig()


def describe_config(config_name: str) -> Dict[str, Any]:
    """Return the explicit overrides defined for this config.

    Shows only the fields this config actually changes from the defaults,
    split into ``world`` (WorldConfig overrides) and ``dqn`` (DQNConfig
    overrides). Unmodified fields follow the defaults in
    :class:`WorldConfig` and :class:`DQNConfig`.
    """
    world_overrides, dqn_overrides = _split_overrides(config_name)
    return {"world": dict(world_overrides), "dqn": dict(dqn_overrides)}
