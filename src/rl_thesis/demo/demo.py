"""
Interactive demo script to watch an agent play.

Runs a random or trained agent in the environment with visualization.
Useful for testing the environment and visualization without training.

Usage:
    python demo.py              # Random agent
    python demo.py --trained    # Use trained checkpoint
"""
import argparse
import sys
import time
import random
from pathlib import Path

from rl_thesis.config.config import WorldConfig, HumanHeuristicConfig, VisualizationConfig
from rl_thesis.agent.human_heuristic import HumanHeuristicAgent
from rl_thesis.visualization.renderer import create_renderer
from rl_thesis.environment.gym_env import SurvivalEnv

def run_demo():

    hhc = HumanHeuristicConfig()
    scripted_agent = HumanHeuristicAgent(hunger_threshold=hhc.hunger_threshold, flee_radius=hhc.flee_radius)
    agent_label = f"Human Heuristic Agent with threshold={hhc.hunger_threshold} and flee_radius={hhc.flee_radius})"
    
    # Create environment
    config = WorldConfig()
    env = SurvivalEnv()
    
    world_w, world_h = config.width, config.height

    # Create renderer
    renderer = create_renderer(
        world_width=world_w,
        world_height=world_h,
        headless=False,
        show_metrics=True,
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
        # Run 10 episodes
        for ep in range(1, 10 + 1):
            state, _ = env.reset()
            total_reward = 0
            step = 0
            
            print(f"Episode {ep} starting...")
            
            while renderer.is_running():
                # Select action
                action = scripted_agent.select_action(env._world)
                
                # Step
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                world_state = env.get_state()
                metrics = {
                    'episodes': ep,
                    'avg_reward': total_reward,
                    'avg_survival': step,
                    'avg_food': info.get('food_eaten', 0) if 'food_eaten' in info else 0,
                    'best_reward': total_reward,
                    'best_survival': step,
                    'epsilon': 0.0,
                    'avg_q_value': 0,
                    'steps_per_sec': 0,
                    'training_time': 0,
                }
                
                if not renderer.render(world_state, metrics):
                    break
                
                time.sleep(VisualizationConfig.tick_duration_ms / 1000)
                
                state = next_state
                
                if terminated or truncated:
                    print(f"Episode {ep}: Reward={total_reward:.1f}, Steps={step}, Death={terminated}")
                    time.sleep(1)  # Pause on death
                    break
            
            if not renderer.is_running():
                break
    
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    
    finally:
        renderer.close()
        print("Demo complete.")