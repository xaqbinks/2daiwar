# src/main.py

import pygame
import pymunk.pygame_util
import torch

# Import our local modules
import config
from environment import Environment
from agent import Agent
from ai_model import DQNAgent

def main():
    """
    Main function to run the AI training loop.
    """
    # --- Pygame and Environment Setup ---
    # We still use pygame for rendering the simulation, even during training
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Py-AI Warehouse - Training")
    clock = pygame.time.Clock()
    env = Environment()

    # Add static boundaries to the environment
    # Floor
    env.add_static_segment((0, config.SCREEN_HEIGHT - 5), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - 5))
    # Left wall
    env.add_static_segment((5, 0), (5, config.SCREEN_HEIGHT))
    # Right wall
    env.add_static_segment((config.SCREEN_WIDTH - 5, 0), (config.SCREEN_WIDTH - 5, config.SCREEN_HEIGHT))
    # Ceiling
    env.add_static_segment((0, 5), (config.SCREEN_WIDTH, 5))

    # --- Agent and AI Setup ---
    agent = Agent(env.space, start_pos=(100, config.SCREEN_HEIGHT - 100))
    ai_agent = DQNAgent(agent.n_observations, agent.n_actions)
    device = ai_agent.device  # Use the same device as the AI agent

    # --- Training Parameters ---
    num_episodes = 500
    max_steps_per_episode = 1000

    # --- Main Training Loop ---
    for i_episode in range(num_episodes):
        agent.reset()
        state_tuple = agent.get_state()
        state = torch.tensor(state_tuple, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0

        for t in range(max_steps_per_episode):
            # Select and perform an action
            action = ai_agent.select_action(state)
            agent.perform_action(action.item())

            # Step the physics simulation
            env.space.step(1.0 / config.FPS)

            # Observe new state and determine reward and done flag
            next_state_tuple = agent.get_state()

            # This is the core of the reward function for this task
            done = agent.torso_body.position.x >= config.SCREEN_WIDTH - 50

            # --- Reward Calculation ---
            # Define the components of our reward function
            goal_reward = 100.0
            living_penalty = -0.1 # Small penalty for each step to encourage speed
            distance_reward_factor = 100.0 # Scales the reward for moving right

            if done:
                reward = goal_reward
            else:
                # Reward for moving closer to the goal (shaping reward)
                distance_reward = (next_state_tuple[0] - state_tuple[0]) * distance_reward_factor
                reward = distance_reward + living_penalty

            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device)
            done_tensor = torch.tensor([done], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state_tuple, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in the AI's memory
            ai_agent.memory.push(state, action, next_state, reward_tensor, done_tensor)

            # Move to the next state
            state = next_state
            state_tuple = next_state_tuple

            # Perform one step of the optimization
            ai_agent.learn()

            # Render the environment every N episodes to check progress
            if i_episode % 20 == 0:
                screen.fill(config.COLOR_BACKGROUND)
                env.space.debug_draw(pymunk.pygame_util.DrawOptions(screen))
                pygame.display.flip()
                clock.tick(config.FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return # Exit training early

            if done:
                break

        print(f"Episode {i_episode+1}/{num_episodes} | Total Reward: {total_reward:.2f}")

    pygame.quit()


if __name__ == "__main__":
    main()
