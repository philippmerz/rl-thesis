from pathlib import Path

import typer

from rl_thesis.config.config import (
    WorldConfig,
    HumanHeuristicConfig,
    VisualizationConfig,
)

app = typer.Typer(
    name="rl_thesis",
    help="Survival RL Thesis Codebase",
    pretty_exceptions_enable=False,
)


@app.command()
def demo(
    checkpoint: str = typer.Option(
        None, "--checkpoint", "-c",
        help="Path to a DQN checkpoint to demo instead of the heuristic agent",
    ),
    frame_stack: int = typer.Option(
        None, "--frame-stack",
        help="Override frame stack size (default: read from checkpoint config, 1 if absent)",
    ),
):
    from rl_thesis.demo.demo import run_demo

    run_demo(
        world_config=WorldConfig(),
        heuristic_config=HumanHeuristicConfig(),
        vis_config=VisualizationConfig(),
        checkpoint_path=checkpoint,
        frame_stack=frame_stack,
    )


@app.command()
def train(
    config: str = typer.Option(
        "baseline", "--config", "-c",
        help="Reward configuration name",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    resume: str = typer.Option(
        None, "--resume", "-r",
        help="Path to checkpoint to resume from",
    ),
    warm_start: str = typer.Option(
        None, "--warm-start",
        help="Path to checkpoint for weight initialization only (fresh optimizer/schedule/epsilon)",
    ),
    steps: int = typer.Option(
        None, "--steps",
        help="Override total timesteps (default from DQNConfig)",
    ),
    lr_schedule: str = typer.Option(
        "onecycle", "--lr-schedule",
        help="LR schedule: 'onecycle' or 'constant'",
    ),
    eval_episodes: int = typer.Option(
        None, "--eval-episodes",
        help="Override number of evaluation episodes",
    ),
    demos: int = typer.Option(
        0, "--demos",
        help="Number of heuristic demonstration episodes to pre-load (0=disabled)",
    ),
    bc_episodes: int = typer.Option(
        0, "--bc-episodes",
        help="Number of heuristic episodes for behavioral cloning pre-training (0=disabled)",
    ),
    epsilon_start: float = typer.Option(
        None, "--epsilon-start",
        help="Override initial epsilon (useful after BC pre-training, e.g. 0.1)",
    ),
    n_step: int = typer.Option(
        None, "--n-step",
        help="Override n-step return horizon (default 5)",
    ),
):
    """Train a single (config, seed) run."""
    from rl_thesis.config.reward_configs import make_dqn_config
    from rl_thesis.training.train import run_single

    cli_overrides = {
        k: v for k, v in {
            "lr_schedule": lr_schedule,
            "total_timesteps": steps,
            "eval_episodes": eval_episodes,
            "epsilon_start": epsilon_start,
            "n_step": n_step,
        }.items() if v is not None
    }
    dqn = make_dqn_config(config, **cli_overrides)
    run_single(config_name=config, seed=seed, dqn_config=dqn,
               checkpoint=resume, warm_start=warm_start,
               demo_episodes=demos, bc_episodes=bc_episodes)


@app.command(name="train-grid")
def train_grid(
    seeds: int = typer.Option(3, "--seeds", "-n", help="Number of seeds (42, 43, ...)"),
    configs: list[str] | None = typer.Option(
        None, "--config", "-c",
        help="Specific configs to run (repeatable). Omit for all.",
    ),
):
    """Run the full experiment grid: configs x seeds."""
    from rl_thesis.training.train import run_grid

    seed_list = list(range(42, 42 + seeds))
    run_grid(seeds=seed_list, configs=configs or None)


@app.command(name="reward-sweep")
def reward_sweep(
    steps: int | None = typer.Option(
        None,
        "--steps",
        help="Training timesteps per run. Defaults to DQNConfig.total_timesteps.",
    ),
    seeds: int = typer.Option(
        3,
        "--seeds",
        "-n",
        help="Number of seeds starting at --start-seed.",
    ),
    start_seed: int = typer.Option(42, "--start-seed", help="First seed value."),
    configs: list[str] | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Specific configs to run (repeatable). Omit for all reward configs.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-w",
        min=1,
        help="Total concurrent training workers. Defaults to 2x visible GPU slots or 1.",
    ),
    gpu_slots: int | None = typer.Option(
        None,
        "--gpu-slots",
        min=0,
        help="Visible GPU slots to schedule against. Defaults to autodetected visible GPUs.",
    ),
    log_dir: Path = typer.Option(
        Path("runs") / "_sweep",
        "--log-dir",
        help="Directory for sweep coordinator and worker logs.",
    ),
):
    """Run the reward sweep with a local task queue."""
    from rl_thesis.training.reward_sweep import run_reward_sweep

    exit_code = run_reward_sweep(
        steps=steps,
        seeds=seeds,
        start_seed=start_seed,
        configs=configs,
        workers=workers,
        gpu_slots=gpu_slots,
        log_dir=log_dir,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command()
def benchmark(
    checkpoint: str = typer.Option(
        None, "--checkpoint", "-c",
        help="Path to DQN checkpoint to compare against heuristic",
    ),
    config: str = typer.Option(
        None, "--config",
        help="Reward config name (for world config). Default uses WorldConfig defaults.",
    ),
    episodes: int = typer.Option(100, "--episodes", "-n", help="Number of episodes"),
    start_seed: int = typer.Option(1000, "--start-seed", help="First episode seed"),
):
    """Benchmark: evaluate heuristic (and optionally a DQN checkpoint)."""
    from rl_thesis.training.benchmark import (
        evaluate_heuristic, evaluate_dqn, summarize, compare,
    )
    from rl_thesis.config.reward_configs import make_world_config as _make

    wc = _make(config) if config else WorldConfig()
    print(f"Evaluating over {episodes} episodes (seeds {start_seed}-{start_seed + episodes - 1})")

    h_results = evaluate_heuristic(wc, episodes, start_seed)
    summarize(h_results, "Heuristic Agent")

    if checkpoint:
        d_results = evaluate_dqn(checkpoint, wc, episodes, start_seed)
        summarize(d_results, f"DQN ({checkpoint})")
        compare(h_results, d_results)


if __name__ == "__main__":
    app()
