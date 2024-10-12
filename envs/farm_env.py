import numpy as np 
import torch

from world_editor import World_Generator

class Farm_Env:
    dirs = np.array([[0,1],[1,0],[-1,0],[0,-1]], dtype=np.int32)

    def __init__(self, size):
        self.world_size = size
        self.obs_channels = 3
        self.act_dim = 5

    def reset(self): 
        self.world = World_Generator(size=self.world_size).generate_world(n_fields=3, field_size=3)
        self.render_info = self.world["render"]

        self.crops = np.array(self.world["crops"])
        self.agent = np.array(self.world["agent"])
        self.obstacles = np.array(self.world["obstacles"])

        self.n_wp_left = 20
        self.crop_growth_steps = 30
        self.crop_growth_steps_remaining = np.zeros(len(self.crops), dtype=int)

        return self.get_observation(), {}

    def step(self, a):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        self.crop_growth_steps_remaining = np.maximum(0, self.crop_growth_steps_remaining - 1)

        if a < 4:            
            # check if hit obstacle
            overlap = np.all(self.obstacles == self.agent + self.dirs[a], axis=1)
            if not np.any(overlap):
                self.agent += self.dirs[a]

                # check if out of bounds 
                if self.agent[0] < 0 or self.agent[0] > self.world_size - 1 or self.agent[1] < 0 or self.agent[1] > self.world_size - 1:
                    self.agent = np.clip(self.agent, 0, self.world_size-1)
        else: 
            harvestable_mask = self.crop_growth_steps_remaining == 0

            overlap = (np.all((self.crops == self.agent), axis=1)) & harvestable_mask
            if np.any(overlap):
                reward += 1
                
                self.crop_growth_steps_remaining[overlap] = self.crop_growth_steps

                self.n_wp_left -= 1
                if self.n_wp_left == 0:
                    terminated = True

        reward -= 0.1

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        observation = np.zeros((self.obs_channels, self.world_size, self.world_size), dtype=np.float32)
        
        observation[0, self.agent[0], self.agent[1]] = 1.0
        observation[1, self.crops[:, 0], self.crops[:, 1]] = 1 - (self.crop_growth_steps_remaining / self.crop_growth_steps)
        observation[2, self.obstacles[:, 0], self.obstacles[:, 1]] = 1.0

        return torch.from_numpy(observation)

    def render(self):
        grid = np.full((self.world_size, self.world_size), '.', dtype=str)
        for i in range(len(self.crops)):
            grid[int(self.crops[i][0]), int(self.crops[i][1])] = "W"
        grid[int(self.agent[0]), int(self.agent[1])] = 'A'
        print("\n".join(["".join(row) for row in grid]))
        print()

