import numpy as np 
import torch

class Farm_Env:
    dirs = np.array([[0,1],[1,0],[-1,0],[0,-1]], dtype=np.int32)

    def __init__(self, world_size, world_generator):
        self.world_size = world_size
        self.world_generator = world_generator
        self.obs_channels = 4
        self.act_dim = 6

    def reset(self): 
        self.world = self.world_generator.generate_world(n_fields=3, field_size=3)
        self.render_info = self.world["render"]

        self.crops = np.array(self.world["train"]["crops"])
        self.agent = np.array(self.world["train"]["agent"])
        self.obstacles = np.array(self.world["train"]["obstacles"])

        self.managers = [Crop_Manager(self.crops)]

        self.tools = [Hoe(self.managers[0])]
        self.tool_poses = np.array([[2,2]])

        self.tool_manager = Tool_Manager(self.tools, self.tool_poses)

        return self.get_observation(), {}

    def step(self, a):
        reward = 0
        truncated = False
        terminated = False
        info = {}

        for manager in self.managers:
            manager.step()

        if a < 4:     
            overlap = np.all(self.obstacles == self.agent + self.dirs[a], axis=1)
            if not np.any(overlap):
                self.agent += self.dirs[a]

                if self.agent[0] < 0 or self.agent[0] > self.world_size - 1 or self.agent[1] < 0 or self.agent[1] > self.world_size - 1:
                    self.agent = np.clip(self.agent, 0, self.world_size-1)
        elif a == 4:
            tool = self.tool_manager.get_active_tool()
            if tool != None:
                reward, terminated = tool.interact(self.agent)
        else:
            self.tool_manager.change_tool(self.agent)

        reward -= 0.1
        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        observation = np.zeros((self.obs_channels, self.world_size, self.world_size), dtype=np.float32)
        
        observation[0, self.agent[0], self.agent[1]] = 1.0
        observation[1, self.crops[:, 0], self.crops[:, 1]] = self.managers[0].get_observation()
        observation[2, self.obstacles[:, 0], self.obstacles[:, 1]] = 1.0

        for i, pos in enumerate(self.tool_manager.tool_poses):
            if i != self.tool_manager.active_tool:
                observation[3, pos[0], pos[1]] = 1.0

        return torch.from_numpy(observation)

    def render(self):
        grid = np.full((self.world_size, self.world_size), '.', dtype=str)
        for i in range(len(self.crops)):
            grid[int(self.crops[i][0]), int(self.crops[i][1])] = "W"
        grid[int(self.agent[0]), int(self.agent[1])] = 'A'
        print("\n".join(["".join(row) for row in grid]))
        print()

"""
bug that when we c an item to pick it up we cannot c again to put it down in the same spot
"""    

class Hoe:
    def __init__(self, manager):
        self.manager = manager

    def interact(self, agent):
        terminated = False
        reward = 0
        harvestable_mask = self.manager.crop_growth_steps_remaining == 0

        overlap = (np.all((self.manager.crops == agent), axis=1)) & harvestable_mask
        if np.any(overlap):
            reward += 1
                
            self.manager.crop_growth_steps_remaining[overlap] = self.manager.crop_growth_steps

            self.manager.n_wp_left -= 1
            if self.manager.n_wp_left == 0:
                terminated = True
        
        return reward, terminated

class Tool_Manager:
    def __init__(self, tools, tool_poses):
        self.tools = tools
        self.tool_poses = tool_poses
        self.active_tool = None

    def get_observation(self):
        pass

    def get_active_tool(self):
        if self.active_tool == None:
            return None
        return self.tools[self.active_tool]

    def change_tool(self, agent):
        overlap = np.all((self.tool_poses == agent), axis=1)
        idx = np.where(overlap)[0]

        if self.active_tool != None:
            self.tool_poses[self.active_tool] = np.array(agent)

        if len(idx) > 0:
            self.active_tool = int(idx)
            self.tool_poses[self.active_tool] = [-1,-1]
        else:
            self.active_tool = None

class Crop_Manager:
    def __init__(self, crops):
        self.crops = crops
        self.n_wp_left = 20
        self.crop_growth_steps = 30
        self.crop_growth_steps_remaining = np.zeros(len(crops), dtype=int)
    
    def step(self):
        self.crop_growth_steps_remaining = np.maximum(0, self.crop_growth_steps_remaining - 1)
    
    def get_observation(self):
        return 1 - (self.crop_growth_steps_remaining / self.crop_growth_steps)