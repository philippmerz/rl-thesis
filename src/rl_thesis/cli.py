from pathlib import Path

import typer

from rl_thesis.config.config import (
    WorldConfig,
    HumanHeuristicConfig,
    VisualizationConfig,
    DQNConfig,
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
):
    from rl_thesis.demo.demo import run_demo

    run_demo(
        world_config=WorldConfig(),
        heuristic_config=HumanHeuristicConfig(),
        vis_config=VisualizationConfig(),
        checkpoint_path=checkpoint,
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
):
    """Train a single (config, seed) run."""
    from rl_thesis.training.train import run_single

    run_single(config_name=config, seed=seed, dqn_config=DQNConfig(),
               checkpoint=resume)


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
    run_grid(seeds=seed_list, dqn_config=DQNConfig(), configs=configs or None)


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


if __name__ == "__main__":
    app()
