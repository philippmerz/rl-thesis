from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from rl_thesis.agent.human_heuristic import HumanHeuristicAgent
from rl_thesis.visualization.renderer import create_renderer
from rl_thesis.environment.gym_env import SurvivalEnv

if TYPE_CHECKING:
    from rl_thesis.config.config import WorldConfig, HumanHeuristicConfig, VisualizationConfig


def run_demo(
    world_config: WorldConfig,
    heuristic_config: HumanHeuristicConfig,
    vis_config: VisualizationConfig,
    checkpoint_path: Optional[str] = None,
):
    if checkpoint_path is not None:
        from rl_thesis.agent.dqn import DQNAgent

        dqn_agent = DQNAgent.from_checkpoint(checkpoint_path)
        agent_label = f"DQN Agent (checkpoint: {checkpoint_path})"
    else:
        dqn_agent = None
        scripted_agent = HumanHeuristicAgent(
            hunger_threshold=heuristic_config.hunger_threshold,
            flee_radius=heuristic_config.flee_radius,
        )
        agent_label = (
            f"Human Heuristic Agent with threshold={heuristic_config.hunger_threshold}"
            f" and flee_radius={heuristic_config.flee_radius})"
        )

    env = SurvivalEnv(world_config)

    renderer = create_renderer(
        config=vis_config,
        world_width=world_config.width,
        world_height=world_config.height,
        headless=False,
    )

    print("\n" + "=" * 40)
    print(f"Survival RL Demo — {agent_label}")
    print("=" * 40)
    print("Controls: Press ESC or close window to exit")
    print("Blue dot = Agent")
    print("Red dots = Enemies")
    print("Green dots = Food")
    print("Gray squares = Shelters")
    print("=" * 40 + "\n")

    try:
        for ep in range(1, 10 + 1):
            state, _ = env.reset()
            total_reward = 0
            step = 0

            print(f"Episode {ep} starting...")

            while renderer.is_running():
                if dqn_agent is not None:
                    action = dqn_agent.select_action(state, training=False)
                else:
                    action = scripted_agent.select_action(env._world)

                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                world_state = env.get_state()
                if not renderer.render(world_state):
                    break

                time.sleep(vis_config.tick_duration_ms / 1000)

                state = next_state

                if terminated or truncated:
                    print(f"Episode {ep}: Reward={total_reward:.1f}, Steps={step}, Death={terminated}")
                    time.sleep(1)
                    break

            if not renderer.is_running():
                break

    except KeyboardInterrupt:
        print("\nDemo interrupted.")

    finally:
        renderer.close()
        print("Demo complete.")
