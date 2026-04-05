"""Named reward configurations for the experiment grid.

Each config is a dict of WorldConfig field overrides.
Only reward-related fields are changed; world mechanics stay constant.

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

# Each config is a dict of WorldConfig field overrides.
# An optional "_dqn" key holds DQNConfig overrides for the experiment,
# keeping reward shaping and hyperparameter choices in one place.
REWARD_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Feasible baseline: all constraints C1-C7 satisfied.
    # Uses delta-based (PBRS) proximity reward.
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
    # 4. Enemy proximity (PBRS delta): flee gradient before contact,
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

    # Engineered v2: addresses hunger conservation plateau.
    #
    # The v1 agent forages well and avoids enemies but moves too much,
    # burning hunger ~3.5x faster than the heuristic (which idles near
    # shelter). Changes from v1:
    #
    # 1. Raise low_hunger threshold from 30% to 50%: creates urgency
    #    earlier, incentivizing food-seeking before critical levels.
    # 2. Add small hunger_proportional (-0.02): provides continuous
    #    gradient for hunger conservation without violating C1.
    #    PV of movement cost: 0.02*0.5/100/0.01 = 0.01/step
    #    Proximity reward: 0.15/15 = 0.01/step (break-even)
    # 3. Stronger shelter_safety (0.3): rewards idle shelter behavior.
    # 4. Stronger food_eaten (10.0): compensates for hunger_proportional
    #    drag, ensures foraging trips remain strongly Q-positive.
    "engineered_v2": {
        "reward_hunger_proportional": -0.02,
        "reward_low_hunger": -0.5,
        "low_hunger_threshold": 0.5,
        "reward_food_eaten": 10.0,
        "reward_food_visible_proximity": 0.15,
        "reward_enemy_damage_taken": -0.5,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_safety": 0.3,
    },

    # Engineered v3: conditional proximity + hunger conservation.
    #
    # Key insight from v1 analysis: the agent forages well (2x heuristic
    # food) and avoids enemies (4x less damage/tick) but moves every tick,
    # burning hunger 3.5x faster. The heuristic idles near shelter when
    # well-fed.
    #
    # The fix: gate food proximity reward behind the hunger threshold.
    # When well-fed (hunger > 50%), no proximity pull toward food.
    # The shelter reward becomes the dominant signal, teaching "stay put."
    # When hungry (hunger < 50%), full proximity gradient activates.
    #
    # This mirrors the heuristic's conditional logic through reward design:
    #   hungry -> forage (proximity + food_eaten)
    #   well-fed -> shelter (shelter_safety, no competing proximity)
    #   enemy nearby -> flee (enemy_proximity, always active)
    "engineered_v3": {
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": -0.5,
        "low_hunger_threshold": 0.5,
        "reward_food_eaten": 8.0,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": -0.5,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_safety": 0.2,
    },

    # Engineered v4: symmetric proximity gradients.
    #
    # V1-V3 all plateau at ~620-680 survival because the agent never
    # learns to navigate TO shelter when well-fed. The shelter_safety
    # reward is binary (in/out), giving no gradient for approach.
    #
    # Fix: add shelter proximity (PBRS delta) that mirrors food proximity.
    # When hungry (< 50%): food proximity active, shelter proximity off.
    # When well-fed (>= 50%): shelter proximity active, food proximity off.
    # This creates symmetric approach gradients for both behavioral modes.
    #
    # Also adds small survival_tick (0.05) to directly reward each tick
    # alive, making conservation strategies Q-value positive.

    # Engineered v6: V5 structure + food_eaten reward.
    #
    # V5 learned excellent shelter/flee behavior (710 survival, only
    # 10 damage/episode) but barely forages (0.58 food vs heuristic 1.19).
    # The survival gap (~60 ticks) is exactly explained by the food gap:
    # 0.6 food * 100 ticks/food = 60 ticks.
    #
    # Root cause: V5 has no food_eaten reward. The food proximity
    # gradient guides toward food, but eating triggers a state
    # transition past the hunger threshold, turning off the proximity
    # signal. From the Q-network's perspective, eating is a transition
    # to a lower-reward state.
    #
    # Fix: add food_eaten=5.0 to make the foraging trip terminal
    # action explicitly rewarding. Five signals total: three PBRS
    # delta proximities, one discrete food event, one terminal death.
    "engineered_v6": {
        "reward_food_eaten": 5.0,
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

    # V6 with n_step=10: longer credit assignment horizon.
    # The food_eaten reward at the end of a foraging trip needs to
    # propagate back through ~5-10 approach steps. With n_step=5,
    # only the last 5 steps see the reward directly; earlier steps
    # rely on slow TD bootstrapping. n_step=10 covers the full
    # typical foraging trajectory.
    "engineered_v6_n10": {
        "reward_food_eaten": 5.0,
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
        "_dqn": {"n_step": 10},
    },

    # Engineered v7: calibrated food + damage on V5 structure.
    #
    # V6 showed that food_eaten=5.0 destabilized V5's shelter/flee
    # behavior (464 vs 710 survival). The large food reward overwhelmed
    # the enemy proximity gradient, causing the agent to trade safety
    # for food. V7 uses a smaller food_eaten (2.0) and reintroduces a
    # moderate damage penalty (-0.3) to maintain the flee incentive.
    "engineered_v7": {
        "reward_food_eaten": 2.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": -0.3,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
    },

    # V7 curriculum phase 1: no enemies.
    # Trains foraging and shelter cycling in isolation. Enemy channels
    # in the observation are present but empty (all zeros).
    "engineered_v7_cur_p1": {
        "reward_food_eaten": 2.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": -0.3,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
        "max_enemy_density": 0.0,
        "initial_enemy_fraction": 0.0,
        "enemy_spawn_rate": 0.0,
    },

    # V7 curriculum phase 2: full environment with warm-started weights.
    # Same rewards as V7; separate config name for distinct output directory.
    "engineered_v7_cur": {
        "reward_food_eaten": 2.0,
        "reward_starvation_damage": 0.0,
        "reward_hunger_proportional": 0.0,
        "reward_low_hunger": 0.0,
        "low_hunger_threshold": 0.5,
        "reward_food_visible_proximity": 0.15,
        "proximity_only_when_hungry": True,
        "reward_enemy_damage_taken": -0.3,
        "reward_enemy_proximity": -0.5,
        "reward_shelter_proximity": 0.15,
        "reward_shelter_safety": 0.0,
        "reward_survival_tick": 0.0,
    },

    # Engineered v5: minimal reward set.
    #
    # Only three proximity gradients + death. No food_eaten, no
    # starvation_damage, no hunger penalties, no shelter_safety.
    # The hypothesis: fewer signals reduce Q-network confusion.
    # The three PBRS delta proximities directly encode the behavioral
    # switch (hungry->food, well-fed->shelter, enemy->flee).
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

    # V5 with extended training and tuned DQN hyperparameters.
    #
    # V5 seed 42 peaked at 766.9 survival (step 1.56M) before
    # oscillating, suggesting the policy hadn't fully stabilized.
    # Changes from default DQN config:
    #   - 5M steps: 2.5x more training to let the policy converge
    #   - epsilon_decay 1M: proportionally longer exploration phase
    #   - buffer 1M: retain more diverse experience over 5M steps
    #   - tau 0.002: slower target updates for stability
    "engineered_v5_long": {
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
    if config_name not in REWARD_CONFIGS:
        available = ", ".join(sorted(REWARD_CONFIGS))
        raise ValueError(f"Unknown reward config '{config_name}'. Available: {available}")
    raw = REWARD_CONFIGS[config_name]
    dqn_overrides = raw.get("_dqn", {})
    world_overrides = {k: v for k, v in raw.items() if k != "_dqn"}
    return world_overrides, dqn_overrides


def make_world_config(config_name: str, seed: int = 42) -> WorldConfig:
    """Create a WorldConfig with the named reward configuration applied."""
    world_overrides, _ = _split_overrides(config_name)
    return replace(WorldConfig(initial_seed=seed), **world_overrides)


def make_dqn_config(config_name: str, **cli_overrides: Any) -> DQNConfig:
    """Create a DQNConfig with config-level and CLI overrides applied.

    CLI overrides take precedence over config-level _dqn overrides.
    """
    _, dqn_overrides = _split_overrides(config_name)
    merged = {**dqn_overrides, **cli_overrides}
    return replace(DQNConfig(), **merged) if merged else DQNConfig()


def get_config_names() -> list[str]:
    return list(REWARD_CONFIGS.keys())


def describe_config(config_name: str) -> Dict[str, Any]:
    """Return the full reward weights for a named config (with defaults filled in)."""
    wc = make_world_config(config_name)
    return {
        "reward_food_eaten": wc.reward_food_eaten,
        "reward_survival_tick": wc.reward_survival_tick,
        "reward_death": wc.reward_death,
        "reward_enemy_damage_taken": wc.reward_enemy_damage_taken,
        "reward_starvation_damage": wc.reward_starvation_damage,
        "reward_low_hunger": wc.reward_low_hunger,
        "reward_hunger_proportional": wc.reward_hunger_proportional,
        "reward_food_visible_proximity": wc.reward_food_visible_proximity,
        "reward_enemy_proximity": wc.reward_enemy_proximity,
        "reward_shelter_proximity": wc.reward_shelter_proximity,
        "reward_shelter_safety": wc.reward_shelter_safety,
        "low_hunger_threshold": wc.low_hunger_threshold,
    }


def validate_config(config_name: str) -> list[str]:
    """Check reward feasibility constraints C1-C7. Returns list of violations."""
    wc = make_world_config(config_name)

    w_food = wc.reward_food_eaten
    w_tick = wc.reward_survival_tick
    w_death = abs(wc.reward_death)
    w_edm = abs(wc.reward_enemy_damage_taken)
    w_sdm = abs(wc.reward_starvation_damage)
    w_hprop = abs(wc.reward_hunger_proportional)
    w_prox = wc.reward_food_visible_proximity

    c = wc.movement_cost
    u_max = wc.max_hunger
    h_max = wc.max_health
    h_s = wc.starvation_damage
    d_e = wc.enemy_damage
    f = wc.food_value
    D = 2 * wc.observation_radius

    T_starve = u_max / wc.hunger_depletion_rate
    T_death = h_max / h_s
    T_passive = T_starve + T_death

    # Sigma_h: cumulative (1 - ratio) over a passive episode.
    u_dot = wc.hunger_depletion_rate
    sigma_h = sum(
        min(1.0, u_dot * t / u_max)
        for t in range(int(T_passive))
    )

    violations = []

    # C1: anti-stasis
    if w_hprop > 0:
        c1_bound = w_hprop * c * (D + 1) / u_max
        if w_prox <= c1_bound:
            violations.append(
                f"C1 anti-stasis: w_prox={w_prox} <= {c1_bound:.4f}"
            )

    # C2: anti-hovering
    if w_hprop > 0:
        c2_bound = w_hprop / 2
        if w_prox >= c2_bound:
            violations.append(
                f"C2 anti-hovering: w_prox={w_prox} >= {c2_bound:.4f}"
            )

    # C3: anti-passivity
    c3_bound = (w_hprop * sigma_h + w_sdm * h_s * T_death + w_death) / T_passive
    if w_tick >= c3_bound:
        violations.append(
            f"C3 anti-passivity: w_tick={w_tick} >= {c3_bound:.4f}"
        )

    # C4: enemy flee
    if w_hprop > 0:
        c4_bound = w_hprop * c / (u_max * d_e)
        if w_edm <= c4_bound:
            violations.append(
                f"C4 enemy flee: |w_edm|={w_edm} <= {c4_bound:.6f}"
            )

    # C5: starvation escalation
    if w_hprop > 0 and w_sdm * h_s <= w_hprop:
        violations.append(
            f"C5 starvation escalation: |w_sdm|*h_s={w_sdm * h_s} <= |w_hprop|={w_hprop}"
        )

    # C6: terminal signal
    if w_death <= w_food:
        violations.append(
            f"C6 terminal signal: |w_death|={w_death} <= w_food={w_food}"
        )

    # C7: food trip
    d_bar = 5  # approximate expected distance to nearest food
    if f <= d_bar * c:
        violations.append(
            f"C7 food trip: f={f} <= d_bar*c={d_bar * c}"
        )

    return violations
