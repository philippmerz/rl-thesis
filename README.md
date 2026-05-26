# rl-thesis

Reward shape and frame stacking in DQN: a reward × observation ablation in multi-objective grid survival.

A Rainbow-lite DQN agent learns to survive on a 64×64 grid by foraging food, avoiding enemies, and using shelters. The codebase studies how reward function design and observation context (frame stacking) interact, and supports a written bachelor's thesis. The full PDF is in `latex/original/thesis.pdf`.

## Install

The project is managed with [`uv`](https://github.com/astral-sh/uv). From the repository root:

```
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and produces a virtualenv at `.venv/`. Python `3.11` is pinned via `.python-version`. After this, prefix any command with `uv run`, or activate the venv directly with `source .venv/bin/activate`.

## Demo

Watch a trained agent or the scripted heuristic play in a pygame window.

```
# Heuristic baseline (default)
uv run rl-thesis demo

# Trained DQN checkpoint (frame-stack size auto-detected from checkpoint)
uv run rl-thesis demo --checkpoint runs/minimal_fs_cap50k/seed_42/checkpoints/model_best.pt

# Override frame stacking (useful when checkpoint config is missing or wrong)
uv run rl-thesis demo --checkpoint path/to/model.pt --frame-stack 4
```

Press ESC or close the window to exit. The demo runs 10 episodes.

## Train

```
uv run rl-thesis train --config minimal_cap50k --seed 42
```

Available configs in `src/rl_thesis/config/experiment_configs.py`:

| Config | Reward | Frame stack | Episode cap |
|---|---|---|---|
| `baseline` | default 11-component reward | 1 | 1,000 |
| `baseline_fs` | same | 4 | 1,000 |
| `absolute_proximity` | replaces Δφ with absolute φ | 1 | 1,000 |
| `absolute_proximity_fs` | same | 4 | 1,000 |
| `minimal_cap50k` | minimal 4-component reward | 1 | 50,000 |
| `minimal_fs_cap50k` | same | 4 | 50,000 |
| `weak_proximity` | baseline with `w_fprox = 0.02` (suicide-failure illustration) | 1 | 1,000 |

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
uv run rl-thesis benchmark --checkpoint path/to/model.pt --config minimal_cap50k
```

Reports mean survival, food eaten, damage taken, and death rate, with a paired t-test against the heuristic on matched seeds (DQN minus heuristic per world).

## Figures

Thesis figures are regenerated from the per-episode CSVs in `vast_logs/per_episode/`.

```
uv run python -m figures.ablation_grid
uv run python -m figures.failure_modes
uv run python -m figures.learning_curves
uv run python -m figures.observation_space
```

Each script writes its PDF to `latex/original/figures/`. To regenerate per-episode CSVs from trained checkpoints first, run `uv run python -m figures.dump_per_episode`.

## Project layout

- `src/rl_thesis/environment/` world mechanics, gym wrapper, frame stacking
- `src/rl_thesis/agent/` DQN, replay buffer, network, human heuristic
- `src/rl_thesis/config/` world config, DQN config, named reward configurations
- `src/rl_thesis/training/` trainer, benchmark, metrics logger
- `src/rl_thesis/demo/` pygame visualization
- `figures/` thesis figure scripts
- `vast_logs/` per-episode benchmark CSVs and bench summary
- `latex/original/` thesis LaTeX source (`thesis.tex`, `refs.bib`, figures)
