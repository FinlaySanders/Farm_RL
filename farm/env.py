import numpy as np 
import torch
import gymnasium as gym
from gymnasium import spaces
import random

from farm.terrain import generate_connected_grid

class FarmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.dirs = [[0,1],[1,0],[-1,0],[0,-1]]
        self.world_size = 10
        self.view_size = 2

        self.n_crops = 10
        self.crop_time = 20
        
        self.shaped_observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "crops": spaces.Box(low=-1, high=1, shape=(self.n_crops, 3), dtype=np.float32),
            "terrain": spaces.Box(low=0, high=1, shape=(1, 2 * self.view_size + 1, 2 * self.view_size + 1), dtype=np.float32),
        })
        self.observation_space = spaces.flatten_space(self.shaped_observation_space)

        self.action_space = spaces.Discrete(5)

        self.terrain = generate_connected_grid(self.world_size, self.world_size, wall_probability=0.3)

    def reset(self, seed=None, options=None):
        self.moves_left = 1000
        self.target = 10
                
        positions = set()
        while len(positions) < 1 + self.n_crops: 
            pos = (random.randint(0, self.world_size-1), random.randint(0, self.world_size-1))
            if self.terrain[pos[0]][pos[1]] == 0:
                positions.add(pos)
        positions = list(positions)

        self.agent = positions[0]
        self.crops = {pos: 0 for pos in positions[1:]}
        
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
            next_pos = (
                max(min(self.agent[0] + self.dirs[a][0], self.world_size-1), 0), 
                max(min(self.agent[1] + self.dirs[a][1], self.world_size-1), 0))
            if not self.terrain[next_pos[0]][next_pos[1]]:
                self.agent = next_pos

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
        crop_obs = []
        for i in range(self.agent[0] - self.view_size, self.agent[0] + self.view_size + 1):
            for j in range(self.agent[1] - self.view_size, self.agent[1] + self.view_size + 1):
                if (i, j) in self.crops:
                    crop_obs += [(i - self.agent[0]) / self.view_size, (j - self.agent[1]) / self.view_size] + [1 - (self.crops[(i,j)] / self.crop_time)]
        
        crop_obs = np.array(crop_obs[:30] + [0] * (30 - len(crop_obs)))

        terrain = np.array(self.terrain)
        pad = self.view_size - 1
        padded_terrain = np.pad(terrain, pad_width=((pad, pad), (pad, pad)), mode='constant', constant_values=1)
        terrain_obs = padded_terrain[self.agent[0] + pad - self.view_size: self.agent[0] + pad + self.view_size + 1, self.agent[1] + pad - self.view_size: self.agent[1] + pad + self.view_size + 1].flatten()

        return np.concatenate((np.array(self.agent) / self.world_size, crop_obs, terrain_obs), dtype=np.float32)

    def render(self):
        return  {"agent":self.agent, "crops":self.crops}

    def close(self):
        pass