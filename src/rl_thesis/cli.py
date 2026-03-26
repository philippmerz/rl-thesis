from typing import List, Optional

import typer

from rl_thesis.config.config import (
    WorldConfig,
    HumanHeuristicConfig,
    VisualizationConfig,
    DQNConfig,
)

app = typer.Typer(name="rl_thesis", help="Survival RL Thesis Codebase")


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
    configs: Optional[List[str]] = typer.Option(
        None, "--config", "-c",
        help="Specific configs to run (repeatable). Omit for all.",
    ),
):
    """Run the full experiment grid: configs x seeds."""
    from rl_thesis.training.train import run_grid

    seed_list = list(range(42, 42 + seeds))
    run_grid(seeds=seed_list, dqn_config=DQNConfig(), configs=configs or None)


@app.command(name="list-configs")
def list_configs():
    """Show all available reward configurations and their weights."""
    from rl_thesis.config.reward_configs import get_config_names, describe_config

    for name in get_config_names():
        weights = describe_config(name)
        print(f"\n{name}:")
        for k, v in weights.items():
            label = k.replace("reward_", "")
            print(f"  {label:20s} {v:+.1f}")


if __name__ == "__main__":
    app()
