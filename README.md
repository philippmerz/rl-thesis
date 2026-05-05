# rl-thesis

Reward shaping for multi-objective survival in grid-based deep reinforcement learning.

A Rainbow-lite DQN agent learns to survive on a 64x64 grid by foraging food, avoiding enemies, and using shelters. The codebase studies how reward function design and observation context (frame stacking) affect learned behavior.

## Install

```
pip install -e .
```

## Demo

Watch a trained agent or the scripted heuristic play in a pygame window.

```
# Heuristic baseline (default)
python -m rl_thesis.cli demo

# Trained DQN checkpoint (frame stacking auto-detected from checkpoint)
python -m rl_thesis.cli demo --checkpoint runs/engineered_v5_fs/seed_42/checkpoints/model_best.pt

# Override frame stacking (useful when checkpoint config is missing or wrong)
python -m rl_thesis.cli demo --checkpoint path/to/model.pt --frame-stack 4
```

Press ESC or close the window to exit. The demo runs 10 episodes.

## Train

```
python -m rl_thesis.cli train --config engineered_v5 --seed 42
```

Available configs live in `src/rl_thesis/config/experiment_configs.py`. `engineered_v5` is the minimal 4-component reward; `engineered_v5_fs` adds 4-frame stacking and 5M-step training.

CLI options:

- `--steps N` override total timesteps
- `--resume PATH` resume from checkpoint (full optimizer state)
- `--warm-start PATH` initialize weights only, fresh optimizer and epsilon
- `--eval-episodes N` override number of evaluation episodes
- `--n-step N` override n-step returns horizon
- `--epsilon-start F` override initial epsilon

## Benchmark

Evaluate a checkpoint against the heuristic baseline over 100 episodes.

```
python -m rl_thesis.cli benchmark --checkpoint path/to/model.pt --config engineered_v5
```

Reports mean survival, food eaten, damage taken, and death rate with a Welch t-test against the heuristic.

## Project layout

- `src/rl_thesis/environment/` world mechanics, gym wrapper, frame stacking
- `src/rl_thesis/agent/` DQN, replay buffer, network, human heuristic
- `src/rl_thesis/config/` world config, DQN config, named reward configurations
- `src/rl_thesis/training/` trainer, benchmark, metrics logger
- `src/rl_thesis/demo/` pygame visualization
- `latex/` thesis source
