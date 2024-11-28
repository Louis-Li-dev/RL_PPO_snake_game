import gymnasium as gym
import numpy as np
from modules.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, length, is_grow, max_length=None, formation='空', seed=0, board_size=21, silent_mode=True, limit_step=True, random_states=[]):
        super().__init__()
        self.game = SnakeGame(length, is_grow, seed=seed, formation=formation, board_size=board_size, silent_mode=silent_mode, random_states=random_states)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(4)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(board_size, board_size, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.max_length = max_length or self.grid_size
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size
        self.done = False

        self.step_limit = self.grid_size * 2 if limit_step else float('inf')
        self.reward_step_counter = 0
        self.total_steps = 0

    def reset(self):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        self.total_steps = 0
        obs = self._generate_observation()
        return obs, None

    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()
        reward = 0.0
        self.reward_step_counter += 1

        if info["snake_size"] == self.max_length: # Snake fills up the entire board. Game over.
            # reward = self.max_growth * 0.1
            reward = 0.5
            self.done = True
            if not self.silent_mode:
                self.render()
            #     self.game.sound_victory.play()
            return obs, reward, self.done, False, info
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        truncate = False
        # if self.total_steps > 5 * self.step_limit:
        #     truncate = True
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            # reward = reward * 0.002  # original: * 0.1
            reward = -(info["snake_size"] + 66) * 2 / self.grid_size
            return obs, reward, self.done, False, info
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = (info["snake_size"] + 66) / self.grid_size
            reward = reward * 10
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:
            '''FOR COACH'''
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            # if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
            #     reward = 2 / info["snake_size"]
            # else:
            #     reward = -2.4 / info["snake_size"]
            # if info['snake_size'] >= 100:
            #     reward = -self.reward_step_counter * 0.00005  # if the snake is long enough, it shouldn't get step reward
            # else:
            #     reward = reward * 0.1

            '''FOR FINAL AGENT'''
            if info['snake_size'] >= 80:
                reward = -self.reward_step_counter * 0.0001  # if the snake is long enough, it shouldn't get step reward
            else:
                reward = -self.reward_step_counter * 0.0005

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        if not self.silent_mode:
            self.render()
            import time
            FRAME_DELAY = 0.05
            time.sleep(FRAME_DELAY)

        reward *= 0.01  # smallalize the reward
        self.total_steps += 1
        return obs, reward, self.done, truncate, info


    def render(self):
        if self.silent_mode: return
        self.game.render()

    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.float32)
        obs = np.stack((obs, obs, obs), axis=-1)
        obs[tuple(self.game.snake[0])] = [0, 255, 0]  # Head
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]  # Tail
        obs[self.game.food] = [0, 0, 255]  # Food
        return obs
