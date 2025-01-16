import numpy as np 
import torch
import gymnasium as gym
from gymnasium import spaces
import random

class FarmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dirs = [[0,1],[1,0],[-1,0],[0,-1]]
        self.world_size = 10
        self.view_size = 2

        self.crop_time = 20
        self.n_crops = 10
        
        self.shaped_observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "crops": spaces.Box(low=0, high=1, shape=(10, 3), dtype=np.float32),
        })
        self.observation_space = spaces.flatten_space(self.shaped_observation_space)
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        self.moves_left = 1000
        self.target = 10
        
        self.agent = (random.randint(0, self.world_size-1), random.randint(0, self.world_size-1))
        
        positions = set()
        while len(positions) < self.n_crops: 
            pos = (random.randint(0, self.world_size-1), random.randint(0, self.world_size-1))
            positions.add(pos)

        self.crops = {pos: 0 for pos in positions}
        
        return self.get_observation(), {}

    def step(self, action):
        reward = 0
        truncated = False
        terminated = False
        info = {}

        a = action

        for crop in self.crops:
            self.crops[crop] = max(self.crops[crop]-1, 0)

        if a < 4:  # Move
            self.agent = (
                max(min(self.agent[0] + self.dirs[a][0], self.world_size-1), 0), 
                max(min(self.agent[1] + self.dirs[a][1], self.world_size-1), 0))

        else:  # Interact
            if self.agent in self.crops and self.crops[self.agent] == 0:
                self.crops[self.agent] = self.crop_time
                reward += 1
                self.target -= 1

        reward -= 0.1
        observation = self.get_observation()
        self.moves_left -= 1

        if self.target == 0:
            terminated = True

        if self.moves_left == 0:
            truncated = True

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        obs = []
        for i in range(self.agent[0] - self.view_size, self.agent[0] + self.view_size + 1):
            for j in range(self.agent[1] - self.view_size, self.agent[1] + self.view_size + 1):
                if (i, j) in self.crops:
                    obs += [(i - self.agent[0]) / self.view_size, (j - self.agent[1]) / self.view_size] + [1 - (self.crops[(i,j)] / self.crop_time)]
        obs = np.array(obs[:30] + [0] * (30 - len(obs)))
        return np.concatenate((np.array(self.agent) / 10, obs))

    def render(self):
        return  {"agent":self.agent, "crops":self.crops}

    def close(self):
        pass

if __name__ == "__main__":
    env = FarmEnv()
    env.reset()
    x = env.get_observation()
    print(x)
    print(gym.spaces.unflatten(env.shaped_observation_space, x))