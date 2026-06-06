# rl-thesis

Reward shape and frame stacking in DQN: a reward × observation ablation in multi-objective grid survival.

A Rainbow-lite DQN agent learns to survive on a 64×64 grid by foraging food, avoiding enemies, and using shelters. The codebase studies how reward function design and observation context (frame stacking) interact, and supports a written bachelor's thesis. The full PDF is in `latex/out/thesis.pdf`.

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

## Reproduce thesis results

Assumes every cell has been trained, i.e. `runs/<config>/seed_<N>/checkpoints/model_best.pt` exists for the six ablation cells × seeds 42–45 and for `weak_proximity/seed_42`. Three commands take you from checkpoints to figures and tables.

### 1. Dump per-episode evaluation data

```
uv run python -m figures.dump_per_episode
```

For each (config, seed), loads `model_best.pt` and replays 100 evaluation episodes on the deterministic seed range 1000–1099. Also runs the scripted heuristic at both episode caps (cap=1000 for misspecified cells, cap=50000 for minimal cells). Writes one CSV per cell × seed plus `heuristic.csv` and `heuristic_cap50k.csv` into `eval_logs/per_episode/`. Slow — roughly 1–2 hours for the full sweep. Use `--skip-existing` to incrementally fill in only what's missing, or `--only <config>` to redo one cell.

### 2. Aggregate into tables

```
uv run python -m figures.aggregate
```

Reads all per-episode CSVs and writes two roll-ups:

- `eval_logs/bench_summary.csv` — per-(config, seed) cell means (consumed by `figures.ablation_grid`).
- `eval_logs/cell_paired_tests.csv` — per-cell paired t-test against the matched-cap heuristic (thesis Table 4).

Prints the paired-test table to stdout.

### 3. Regenerate the figures

```
uv run python -m figures.ablation_grid
uv run python -m figures.failure_modes
uv run python -m figures.learning_curves
uv run python -m figures.observation_space
```

Each writes a PDF into `latex/figures/`. `failure_modes` also writes `eval_logs/per_episode/failure_mode_summary.csv` (the failure-mode appendix table) as a side effect.

### Artifact map

| Thesis artifact | Producer | Underlying data |
|---|---|---|
| Table 3 (cell means + SEM) | `figures/aggregate.py` | `eval_logs/bench_summary.csv` |
| Table 4 (per-cell paired CIs) | `figures/aggregate.py` | `eval_logs/cell_paired_tests.csv` |
| Failure-mode appendix table | `figures/failure_modes.py` | `eval_logs/per_episode/failure_mode_summary.csv` |
| `ablation_grid.pdf` | `figures/ablation_grid.py` | `eval_logs/bench_summary.csv` |
| `failure_modes.pdf` | `figures/failure_modes.py` | `eval_logs/per_episode/*.csv` |
| `learning_curves.pdf` | `figures/learning_curves.py` | `runs/<cfg>/seed_<N>/logs/eval.csv` |
| `observation_space.pdf` | `figures/observation_space.py` | (synthetic sample, no run data) |

## Project layout

- `src/rl_thesis/environment/` world mechanics, gym wrapper, frame stacking
- `src/rl_thesis/agent/` DQN, replay buffer, network, human heuristic
- `src/rl_thesis/config/` world config, DQN config, named reward configurations
- `src/rl_thesis/training/` trainer, benchmark, metrics logger
- `src/rl_thesis/demo/` pygame visualization
- `figures/` thesis figure scripts
- `eval_logs/` per-episode benchmark CSVs and bench summary
- `latex/` thesis LaTeX source (`thesis.tex`, `refs.bib`, `figures/`); built PDF in `latex/out/`
