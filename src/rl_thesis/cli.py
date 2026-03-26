import typer

from rl_thesis.config.config import (
    WorldConfig,
    HumanHeuristicConfig,
    VisualizationConfig,
    DQNConfig,
)

app = typer.Typer(name="rl_thesis", help="My Survival RL Thesis Codebase")


@app.command()
def demo(
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Path to a DQN checkpoint to demo instead of the heuristic agent"),
):
    from rl_thesis.demo.demo import run_demo

    run_demo(
        world_config=WorldConfig(),
        heuristic_config=HumanHeuristicConfig(),
        vis_config=VisualizationConfig(),
        checkpoint_path=checkpoint,
    )


@app.command()
def train():
    from rl_thesis.training.train import run_training

    run_training(
        world_config=WorldConfig(),
        dqn_config=DQNConfig(),
    )


if __name__ == "__main__":
    app()
