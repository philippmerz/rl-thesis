# Experiments Log

Chronological record of all reward shaping and architecture experiments. Each entry states motivation, key changes, result, and takeaway.

## Reference baselines

**Heuristic baseline** (scripted agent)
- 768 mean survival, 1.19 food, 95% death rate, 5% time-limit (1000 tick)
- Camps in nearest shelter, forages when hunger < 0.5, flees enemies within distance 5

## Phase 1: Baseline + failure mode discovery

### `baseline`
- Default weights: food_eaten=+5, death=-10, enemy_damage=-0.1, starvation_damage=-1, hunger_proportional=-0.3, food_proximity=+0.1, shelter_safety=+0.1
- All three failure modes observed in a single training run: oscillation, food hovering, suicidal enemy-seeking
- Motivated the failure mode analysis and design heuristics

### `absolute_proximity`
- `proximity_delta: False` (use raw closeness instead of delta)
- Produced persistent food hovering: agent camps next to food without eating. The accumulated proximity stream (~36/episode at +0.1) dominated the one-time +5 food reward
- Takeaway: PBRS delta form eliminates hovering incentive

### `weak_proximity`
- `food_proximity=0.02` (violates H1: 0.02 < 0.0225 break-even)
- Agent learns suicide mode to escape accumulating hunger penalty
- Validates H1 as necessary for foraging to be Q-value positive

### `no_death`
- `death=0` (removed terminal signal)
- Tested whether per-step penalties alone drive survival
- Ineffective: no clear episode-level gradient

### `death_only`
- All per-step rewards zero, only terminal death penalty
- Sparse reward credit assignment failed as expected at this horizon

### `free_movement`
- `movement_cost=0` (diagnostic)
- Agent still oscillates, suggesting oscillation is partly a training dynamics artifact, not pure reward rationality

## Phase 2: Engineered progression

### `engineered` (E1)
- Removed `hunger_proportional`, added flat `low_hunger=-0.5` threshold penalty, food=8.0, food_proximity=0.15, enemy_proximity=-0.5
- Key finding: `hunger_proportional` + `movement_cost` + gamma=0.99 produces stasis. PV of per-step movement penalty (0.15) exceeds max proximity reward (0.007) by factor 22. No proximity weight can overcome this.
- E1 agent forages well (~2 food/ep) but moves every tick, burning hunger 3.5x faster than heuristic

### `engineered_v2`
- Attempted hunger conservation: threshold raised to 0.5, added small `hunger_proportional=-0.02`, stronger shelter_safety, food=10.0
- Movement cost reintroduced, agent regressed
- Takeaway: any nonzero hunger_proportional causes the stasis problem

### `engineered_v3`
- Added `proximity_only_when_hungry` gating
- Food proximity fires when u<0.5, shelter proximity when u>=0.5
- Introduced conditional mode switch through reward gating

### `engineered_v4` (E4, 9 components)
- E3 + symmetric proximity gradients (food + shelter + enemy) + survival_tick=0.05
- Benchmark: 695 survival, 2.19 food, 99 damage
- Over-forages: eats 2x heuristic but takes full damage (no shelter behavior)

### `engineered_v5` (E5, 4 components) — STRONG BASELINE
- Minimal set: 3 PBRS delta proximities (food when hungry, shelter when well-fed, enemy always) + death penalty. All event-driven and threshold rewards zeroed.
- Benchmark: 706 survival, 0.58 food, 10.4 damage, 100% death
- Episode-level: 50% of episodes eat 0 food (654 survival), 50% eat 1+ food (767 survival, matches heuristic)
- Peak eval: 766.9 survival at step 1.56M (seed 42)
- **Takeaway: fewer signals outperform more (E5 > E4 despite strictly less information)**

## Phase 3: Food reward ablations

### `engineered_v6` (V5 + food_eaten=5.0)
- Added `food_eaten=5.0` to E5
- Benchmark: 467 survival, 0.81 food, 100 damage (10x more than E5)
- Catastrophic regression. Food reward overwhelmed enemy proximity gradient, agent pursues food into danger.

### `engineered_v6_n10`
- V6 + n_step=10 (longer credit assignment)
- Same collapse. The issue was not n-step horizon.

### `engineered_v7`
- food_eaten=2.0 with compensating enemy_damage=-0.3
- Benchmark: 497 survival, 0.70 food, 126 damage
- Still too large. Magnitude to affect foraging exceeds threshold that preserves flee.

### `engineered_v7_cur` (curriculum learning)
- Phase 1: 1M steps with no enemies (learn foraging first)
- Phase 2: warm-start weights, full environment
- Benchmark: 685 survival (worse than E5)
- Foraging learned without enemies did not transfer. Flee behavior must be learned jointly.

## Phase 4: Extended training

### `engineered_v5_long`
- E5 + 5M steps, 1M epsilon decay, 1M buffer, tau=0.002
- No improvement. Same peak (~715) as 2M run, then slow decline.
- Takeaway: extended training with standard LR schedule doesn't help E5.

## Phase 5: Frame stacking (new observation)

### `engineered_v5_fs`
- E5 reward + 4-frame stacking, 5M steps, buffer=250K, tau=0.002
- Peak eval: seed 44 hit 867.5 survival with 5.70 food at step 380K (epsilon=0.62)
- Then collapsed back to 645-665 survival with 0-0.35 food by step 2.9M
- Benchmark of best checkpoint: **811 survival, 4.12 food, 88% death (p=0.007 vs heuristic)** — first agent to significantly beat heuristic
- Takeaway: frame stacking gives representational capacity for conditional foraging, but the policy collapsed as epsilon decayed

### `engineered_v6_fs_g1` (FS + gamma=1 + survival_tick)
- Hypothesis: collapse caused by discount blindness (death penalty invisible at gamma=0.99)
- Added gamma=1.0, survival_tick=0.005, epsilon_end=0.05
- Peak: seed 44 hit 781 survival with 5.25 food at step 520K
- End state (5M steps): 608-632 survival, **0.00 food**, 25-29 damage — full collapse
- Benchmark: heuristic significantly outperforms
- Takeaway: gamma=1 worsened late-training dynamics. Agent learned neither shelter nor forage. Rejected hypothesis.

### `engineered_v6_fs_food` (FS + food_eaten=0.3)
- Hypothesis: small food reward preserves flee gradient while rewarding foraging terminal action. Trip nets +0.35 total vs enemy proximity penalty of 0.033/step.
- Peak: seed 43 hit 860 survival with 7.10 food, 65% death at step 940K (epsilon=0.11)
- Still strong at step 1.52M: 821 survival, 10.35 food, 65% death — broadest peak observed
- By 4M steps: collapsed to 630-660 survival with 0-1 food
- Takeaway: small food reward works (vs V6/V7's 2-5 which destabilized). Peak was broader than V5_fs. But still collapsed late.

## Phase 6: Plasticity investigation

### Collapse mechanism hypothesis
- All FS experiments peak during moderate epsilon then collapse as epsilon decays AND LR decays together
- Replay buffer distribution shift: as policy becomes greedy, non-foraging transitions dominate
- OneCycle LR at step 2M is ~5e-5; by step 3M it's ~2e-5. Too small to correct the buffer shift once foraging transitions age out
- Self-reinforcing: less foraging in buffer → worse foraging Q-values → less foraging behavior

### `engineered_v7_fs` (FS + food_eaten=0.3 + constant LR + 2M steps)
- Hypothesis: constant LR (1e-4) maintains plasticity to resist buffer distribution shift. Shorter training (2M) captures peak window.
- Benchmarks (100 episodes each):
  - Seed 42: 797.7 survival, 4.55 food, p=0.060
  - Seed 43: **826.1 survival, 3.91 food, p=0.0001**
  - Seed 44: **812.3 survival, 3.85 food, 12% time-limit, p=0.007**
- Takeaway: constant LR helps. 2/3 seeds significantly beat heuristic. Still regression in late training but not catastrophic collapse.

## Phase 7: Targeted collapse interventions (V8)

Three interventions tested in parallel, each attacking a different candidate root cause. All use V7_fs base (FS + food_eaten=0.3 + constant LR + 2M steps).

### `engineered_v8_fs_cycle` — cyclical epsilon
- Hypothesis: epsilon resets to 0.5 every 500K steps refill the replay buffer with diverse transitions, breaking the self-reinforcing distribution shift.
- Benchmarks:
  - Seed 42: 780.0 survival, 3.75 food, 10% time-limit, p=0.38
  - Seed 43: **858.7 survival, 5.36 food, 28% time-limit, p<0.05** (NEW RECORD at time)
  - Seed 44: 742.2 survival, 1.42 food, p=0.05 (below heuristic)
- Takeaway: best seed broke 850 for the first time. Inconsistent across seeds (28% vs 3% time-limit).

### `engineered_v8_fs_reset` — Nikishin head resets
- Hypothesis: resetting the Dueling head weights every 500K steps (keeping CNN encoder and replay buffer) directly counters plasticity loss and primacy bias (Nikishin et al. 2022).
- Benchmarks:
  - Seed 42: 788.0 survival, 6.22 food, 25% time-limit, p=0.24
  - Seed 43: **840.5 survival, 7.63 food, 35% time-limit, p<0.001**
  - Seed 44: **871.7 survival, 6.25 food, 34% time-limit, p<0.0001** (overall best)
- Median episode for seed 44: 902 ticks. Min/max: 398/1000.
- Takeaway: head resets produced the highest individual seed and the best food-consumption numbers. 2/3 seeds significantly beat heuristic. Plasticity diagnosis confirmed as a useful fix.

### `engineered_v8_fs_strong` — stronger foraging signals
- Hypothesis: doubling food_eaten (0.3 → 0.5) and food_proximity (0.15 → 0.3) pushes foraging harder. Enemy proximity stays at -0.5; ratio now 1.0x instead of 3.3x.
- Benchmarks:
  - Seed 42: 788.4 survival, 4.61 food, 12% time-limit, p=0.21
  - Seed 43: 637.9 survival, 2.92 food, p<0.001 (**heuristic wins**)
  - Seed 44: 646.9 survival, 3.60 food, p<0.001 (**heuristic wins**)
- Takeaway: stronger signals hurt more than helped. 2/3 seeds significantly underperformed heuristic. Rejected.

## Key findings

1. **Fewer reward components outperform more** (E5 > E4 with strictly less information)
2. **Delta proximity eliminates hovering** (vs absolute proximity)
3. **Hunger-proportional penalty is structurally incompatible** with movement cost at gamma=0.99
4. **Food reward magnitude matters**: 0.3 works with FS, 2-5 destabilizes flee, 0.5 also destabilizes
5. **Frame stacking enables conditional foraging** — 811 benchmark (p=0.007) is the first significant improvement
6. **Policy collapse is not a reward problem** — gamma=1 made it worse
7. **Policy collapse correlates with LR decay** — constant LR (V7_fs) shows more durable peaks
8. **Plasticity diagnosis is correct** — Nikishin head resets (V8_fs_reset) produced the best overall result: 871 survival, 34% time-limit rate, 6.25 food/episode (p<0.0001)

## Current best

V8_fs_reset seed 44: 871 survival ± 27.7 (95% CI), 6.25 food/episode, 34% of episodes reach 1000-tick time limit, 66% death rate. Significantly beats heuristic (768 survival, 5% time-limit) with p<0.0001, t=6.05.

Not "consistently 1000 ticks" but a ~13% improvement in mean survival and ~7x improvement in time-limit rate over the heuristic baseline.

## Remaining questions

- Why does V8_fs_reset show inconsistency across seeds (seed 42 at 788 vs seed 44 at 871)?
- Would combining cyclical epsilon + head resets work better than either alone?
- Would longer resets (every 250K) or fewer (every 1M) work better?
- Can we push past 35% time-limit without architectural changes beyond DQN?

## Appendix: Historical configurations

The configs in this appendix were run during the investigation but are not part of the final thesis narrative: each either failed to clear a bar that a later config cleared, or tested a hypothesis that was ruled out. The dicts below are the exact overrides they applied to `WorldConfig` defaults (plus `_dqn` overrides on `DQNConfig` where shown). They have been removed from `src/rl_thesis/config/experiment_configs.py`; this appendix preserves them for reproducibility.

### Reward-shaping dead ends

**`engineered_v2`** — Hunger conservation attempt. Raise `low_hunger_threshold` to 50%, add `hunger_proportional=-0.02`, stronger `shelter_safety` (0.3), stronger `food_eaten` (10.0). Outcome: hunger-proportional penalty at `-0.02` fought the movement cost and the agent regressed below v1.

```python
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
```

**`engineered_v3`** — Conditional proximity gating: food proximity only when hungry; shelter behaviour dominant when well-fed. Outcome: 620--680 survival plateau, same as v1/v2. Ruled in as a useful idea (kept in later configs via `proximity_only_when_hungry`), but this specific tuning did not beat v1.

```python
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
```

**`engineered_v6_n10`** — Extend n-step return horizon from 5 to 10, holding `food_eaten=5.0`. Hypothesis: longer credit assignment propagates the food reward further back through an approach trajectory. Outcome: same collapse as `food_eaten=5.0` at n=5. The issue was reward magnitude, not credit-assignment horizon.

```python
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
```

**`engineered_v7`** — Calibrated `food_eaten=2.0` plus `damage_taken=-0.3`, on the V5 reward scaffold. Outcome: still too large; foraging trips dominated the flee gradient and survival regressed.

```python
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
```

**`engineered_v7_cur_p1` / `engineered_v7_cur`** — Two-phase curriculum: train foraging with enemies suppressed (phase 1), then warm-start into the full environment (phase 2). Outcome: the foraging policy learned without enemies did not transfer; phase 2 benchmarked at 685 survival, below E5. Curriculum abandoned.

```python
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
```

### Frame-stacking / dynamics dead ends

**`engineered_v6_fs_g1`** — V5_fs with `gamma=1.0` plus `survival_tick=0.005` and `epsilon_end=0.05`. Hypothesis: V5_fs's collapse was discount-blindness to long-horizon death; removing the discount should make survival explicitly valuable. Outcome: `gamma=1` worsened late-training dynamics; the collapse was a plasticity/buffer problem, not a discount problem. This run is what motivated the shift from reward interventions to training-dynamics interventions.

```python
"engineered_v6_fs_g1": {
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
    "reward_survival_tick": 0.005,
    "_dqn": {
        "frame_stack": 4,
        "total_timesteps": 5_000_000,
        "epsilon_decay_steps": 1_000_000,
        "epsilon_end": 0.05,
        "buffer_size": 250_000,
        "tau": 0.002,
        "gamma": 1.0,
    },
},
```

**`engineered_v5_long`** — E5 with 5M training steps (2.5x default), buffer raised to 1M, slower target updates. Hypothesis: the policy simply hadn't converged. Outcome: no improvement; peak survival stayed at ~715 and then slowly declined. Extended training by itself does not fix the collapse.

```python
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
```

**`engineered_v9_fs_cycle_reset`** — Cyclical epsilon plus head resets on top of V7_fs. Hypothesis: the two V8 interventions should compose (buffer diversity + network plasticity). Outcome: did not compose cleanly; the combination was no better than head resets alone and was superseded by `engineered_v9_fs_strong_reset` (stronger signals + resets), which did compose well.

```python
"engineered_v9_fs_cycle_reset": {
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
        "head_reset_freq": 500_000,
    },
},
```
