import os
import sys
root_dir = os.path.join(os.getcwd(), '..', '..')
if root_dir not in sys.path:
    sys.path.append(root_dir)
from modules.gym_wrapper import SnakeEnv
import torch
import numpy as np
from models.PPO import Agent

device = torch.device('cuda')
if __name__ == "__main__":
    env = SnakeEnv(length=3, is_grow=True, board_size=21)
    
    N = 20
    batch_size = 5
    n_epochs = 20
    alpha = 1e-4
    agent = Agent(n_actions=4, batch_size=batch_size, alpha=alpha, \
                  n_epochs=n_epochs, input_channels=3, embedding_dim=100)

    n_games = 10000
    score_history = []
    learn_iters = 0
    avg_score= 0 
    n_steps = 0
    best_score = -np.inf

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
            action, prob, val = agent.choose_action(observation.to(device))

            observation_, reward, done, _, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation.cpu(), action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1# Preprocess the new observation
            observation = torch.tensor(observation_, dtype=torch.float32).permute(2, 0, 1) / 255.0


        score_history.append(score)
        avg_score = np.mean(score_history)
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episodes', i, "score %.6f" % score, 'avg score %.6f' % avg_score,
                "time_steps", n_steps, 'learning_steps', learn_iters)
        
        print("Game Over!")
