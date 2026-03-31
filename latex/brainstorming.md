# Engineering a DQN Agent to Beat the Heuristic

## Target: Heuristic Agent Performance (100 episodes, seeds 1000-1099)
- Survival: 768.8 +/- 19.1 (95% CI)
- Food eaten: 1.19
- Damage taken: 99.9
- Death rate: 94%, Time limit: 6%
- Median: 754, Min/Max: 454/1000

## Root Cause Analysis

### The structural C1 incompatibility (gamma=0.99)
With movement_cost=0.5, hunger_proportional=-0.3, gamma=0.99:
- Each movement step costs 0.5 hunger
- Through w_hprop, this permanently shifts per-tick penalty by 0.3 * 0.5/100 = 0.0015
- Present value of permanent shift: 0.0015/(1-0.99) = **0.15 per step**
- Delta proximity reward per step: 0.1/15 = **0.007 per step**
- Ratio: **movement costs 22x more than proximity reward**
- No feasible proximity weight can overcome this at gamma=0.99

### Enemy avoidance failure (all existing configs)
- Only ~3 enemies on 64x64 grid; encounters are rare
- Damage penalty -0.1 per unit = -1.5 per hit; too weak
- No gradient for proactive fleeing before contact
- n_step=3 insufficient to propagate approaching-enemy signal

### Free movement diagnostic (movement_cost=0)
NOTE: movement_cost=0 is experimental, not a valid reward config.
But its 100-episode benchmark reveals the DQN CAN learn effectively:
- Survival: 809.6 +/- 35.8 (vs heuristic 792.0 on same config)
- Food: 2.33, Death: 76%, TimeLimit: 24%
- But high variance (min=125) from catastrophic early enemy deaths
- The 10-episode training eval (649 survival) was MISLEADING - high variance

## Solution: The "Engineered" Config

### Core fix: replace hunger_proportional with threshold penalty
Set reward_hunger_proportional = 0.0 (removes structural conflict).
Set reward_low_hunger = -0.5 (flat penalty when hunger < 30%).

Why this works (key mathematical property):
- Flat penalty doesn't scale with hunger level, unlike proportional
- Movement depletes hunger faster but incurs the SAME -0.5/tick penalty once below threshold
- The PV of earlier threshold crossing from one movement step:
  approximately 0.22 (near threshold), negligible (far from threshold)
- The PV of eating food at distance 5: **+29.44** (7.61 food + 21.83 delayed penalty)
- Foraging is strongly Q-value positive even with movement_cost=0.5

The critical insight: hunger_proportional creates a DIFFERENTIAL penalty for
movement (each step permanently increases the penalty via hunger depletion).
The threshold penalty creates a FLAT penalty that doesn't amplify with movement.

### Enemy avoidance: PBRS delta proximity + stronger damage
- reward_enemy_proximity = -0.5 (delta-based, only when not in shelter)
- reward_enemy_damage_taken = -0.5 (-7.5 per hit vs +8.0 per food)
- Flee gradient: 0.5/15 = 0.033 per step (5x food proximity gradient of 0.007)
- Shelter blocks enemy proximity penalty, teaching agent that shelter = safe

### Complete "engineered" config
```
movement_cost:                0.5   (default, unchanged)
reward_hunger_proportional:   0.0   (removed -- the structural fix)
reward_low_hunger:           -0.5   (threshold at 30% hunger)
reward_food_eaten:            8.0   (increased from 5.0)
reward_food_visible_proximity:0.15  (increased from 0.1)
reward_enemy_damage_taken:   -0.5   (increased from -0.1)
reward_enemy_proximity:      -0.5   (NEW, PBRS delta)
reward_death:               -10.0   (default)
reward_starvation_damage:    -1.0   (default)
reward_shelter_safety:        0.1   (default)
proximity_delta:              True  (default)
```

### Constraint check: all satisfied (w_hprop=0 trivializes C1,C2,C4,C5)

### Demonstration pre-loading (DQfD-lite)
Pre-fill replay buffer with 100 heuristic episodes (~77k transitions).
This bootstraps all three behaviors (flee, forage, shelter) before training
begins, bypassing the cold-start exploration problem.

## Experiment Log

### 50k step diagnostic (no demos, movement_cost=0.5)
Survival plateaued around 480-548, food 0.55-1.1, death rate 100%.
Enemy damage low (4.5-21.75) indicating proximity penalty works for avoidance.
But epsilon still > 0.91 at end - WAY too early to judge foraging.

### 50k step diagnostic (with 100 demo episodes, movement_cost=0.5)
Survival: 524-635, food 0.20-0.55, damage 13.5-45.75, death rate 100%.
Demos provide slightly higher initial survival (635 vs 540 at step 5k)
but both converge to similar performance by step 45k.
Still too early (epsilon > 0.91) for meaningful comparison.

### Key observation from 50k diagnostics
Both configs are still in the random-exploration phase (epsilon > 0.91).
The baseline config took 500k+ steps to show meaningful behavior.
Need to run for 1-2M steps for a proper evaluation.

## Next Steps (for full training run)

### Recommended training command
```bash
rl_thesis train --config engineered --seed 42 --steps 2000000 --eval-episodes 20 --demos 100
```

### What to expect
- Steps 0-100k: mostly random exploration, epsilon 1.0 -> 0.8
- Steps 100k-500k: epsilon drops to 0.05, main learning phase
- Steps 500k-2M: refinement phase with stable epsilon
- Best checkpoint saving captures peak performance

### If training stalls at ~500-600 survival
Consider:
1. Increasing demo episodes to 200 (more behavioral coverage)
2. Reducing low_hunger penalty to -0.3 (less overwhelming signal)
3. Increasing n_step from 3 to 5 (better credit assignment)
4. Using constant LR instead of OneCycleLR (--lr-schedule constant)

### If enemy avoidance is poor (high damage, early deaths)
Consider:
1. Increasing reward_enemy_proximity to -1.0
2. Adding an explicit shelter bonus (reward for being in shelter)
3. Frame stacking (4 frames) to give agent velocity information

### If foraging is poor (low food count)
Consider:
1. Increasing food_visible_proximity to 0.25
2. Increasing food_eaten to 12.0
3. Lowering the low_hunger threshold from 30% to 50% for earlier urgency

## Code Changes Summary (all implemented)

### config.py
- Added `reward_enemy_proximity: float = 0.0` to WorldConfig
- Added `lr_schedule: str = "onecycle"` to DQNConfig

### world.py
- Added `_prev_enemy_closeness` tracking (init + reset)
- Added `_nearest_visible_enemy_distance()` method
- Added enemy proximity delta reward computation in step()
  (only applied when not in shelter)
- Fixed missing `_prev_closeness` initialization in __init__

### reward_configs.py
- Added "engineered" config with all new weights
- Updated describe_config() to include reward_enemy_proximity

### dqn.py
- Added constant LR schedule option (lr_schedule="constant")
- Fixed scheduler step guard for both scheduler types

### trainer.py
- Added `load_demonstrations()` method (runs heuristic, fills buffer)
- Added `_best_eval_survival` tracking and model_best.pt saving

### cli.py
- Added --steps, --lr-schedule, --eval-episodes, --demos options to train
- Added benchmark command

### New: training/benchmark.py
- Evaluates heuristic and DQN agents over N episodes
- Reports survival, food, damage, death rate with 95% CI
- Welch's t-test comparison (pure numpy, no scipy)
