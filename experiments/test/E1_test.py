import os
import sys
import torch
import numpy as np

# Adjust the path if necessary
root_dir = os.path.join(os.getcwd(), '..', '..')
if root_dir not in sys.path:
    sys.path.append(root_dir)
from models.PPO import Agent
from modules.gym_wrapper import SnakeEnv
import time


if __name__ == "__main__":
    # Initialize the environment with display enabled
    env = SnakeEnv(length=3, is_grow=True, board_size=21, silent_mode=False)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the agent
    n_actions = 4  # Ensure this matches your environment's action space
    agent = Agent(n_actions=n_actions, input_channels=3, embedding_dim=100)

    # Load the trained models
    agent.load_models()

    # Set the models to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    # Number of testing episodes
    n_test_episodes = 5

    for i in range(n_test_episodes):
        obs, _ = env.reset()
        done = False
        score = 0

        while not done:
            # Preprocess the observation
            observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0

            # Ensure no gradients are calculated during testing
            with torch.no_grad():
                action, _, _ = agent.choose_action(observation.to(device))

            # Take the action in the environment
            obs_, reward, done, _, info = env.step(action)
            score += reward

            # Render the environment
            env.render()

            # Small delay to control the speed of rendering
            time.sleep(0.1)

            # Update the observation
            obs = obs_

            # Optionally, print the action taken
            # print(f"Action taken: {action}")

        print(f"Test Episode {i+1}: Score = {score}")
        print("Game Over!")

    env.close()
