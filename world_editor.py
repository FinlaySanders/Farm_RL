import numpy as np
from collections import deque
import random

EMPTY = E = 0
PLAYER = P = 1
CROP = C = 2
WATER = W = 3
BRIDGE = B = 4
FISH = F = 5

obstacle_grid = np.array([
    [W,W,W,W,W,W,W,W,W,W],
    [W,0,0,0,0,0,0,0,0,W],
    [W,0,0,0,0,0,0,0,W,W],
    [W,W,0,0,B,0,W,W,W,W],
    [W,W,W,W,B,W,W,W,W,W],
    [W,W,W,W,B,W,W,W,W,W],
    [W,0,0,0,B,0,0,0,0,W],
    [W,0,0,0,0,0,0,0,0,W],
    [W,W,W,W,W,W,W,W,W,W],
    [W,W,W,W,W,W,W,W,W,W],
])

class World_Generator:
    def __init__(self, size):
        self.world = np.array(obstacle_grid)
        self.world_size = size
        self.dirs = np.array([[0,1], [1,0], [-1,0], [0,-1]], dtype=np.int32)

        self.EMPTY = 0
        self.PLAYER = 1
        self.CROP = 2
        self.OBSTACLE = 3

    def generate_world(self, n_fields, field_size):
        walkable = np.argwhere((self.world == EMPTY) | (self.world == BRIDGE) | (self.world == CROP) | (self.world == self.PLAYER))

        water = np.argwhere(self.world == WATER)
        bridge = np.argwhere(self.world == BRIDGE)
        fish = np.argwhere(self.world == FISH)
        obstacles = np.concatenate((water, fish))

        for _ in range(n_fields):
            self.place_crops_bfs(self.random_pos_at(self.EMPTY), field_size)
        crops = np.argwhere(self.world == CROP)

        agent = np.array(self.random_pos_at(self.EMPTY))

        return {"crops":crops, "obstacles":obstacles, "agent":agent, "walkable": walkable, 
                "render":{"water":water, "bridge":bridge, "fish":fish}}

    def place_crops_bfs(self, start_pos, amount):
        queue = deque([start_pos])
        visited = set()
        visited.add(start_pos)
        self.world[start_pos[0], start_pos[1]] = self.CROP

        while len(visited) < amount:
            pos = queue.popleft()
            pos_array = np.array(pos)

            neighbors = self.dirs + pos_array 

            valid_indices = self.are_valid_poses(neighbors)
            valid_neighbors = neighbors[valid_indices]

            for neighbor in valid_neighbors:
                neighbor_tuple = tuple(neighbor)
                if not neighbor_tuple in visited:
                    if self.world[neighbor_tuple] == self.EMPTY:
                        visited.add(neighbor_tuple)
                        queue.append(neighbor_tuple)
                        self.world[neighbor[0], neighbor[1]] = self.CROP
                    # allows for overlapping fields to grow into one large field
                    elif self.world[neighbor_tuple] == self.CROP:
                        queue.append(neighbor_tuple)
       
       # the above code allows for a few more crops to be generated than expected
        return list(visited)[0:amount]

    def place_random_obstacles(self, num):
        poses = set()
        invalid_poses = set()


        while len(poses) < num:
            if len(poses) + len(invalid_poses) == self.world_size * self.world_size:
                break

            new_pos = self.random_pos_at(self.EMPTY)

            # check if already known to be invalid
            if new_pos in invalid_poses:
                continue

            self.world[new_pos] = self.OBSTACLE

            # check if invalid
            empty_found = self.bfs_search(self.random_pos_at(self.EMPTY))

            if len(empty_found) < len(np.argwhere(self.world == self.EMPTY)):
                self.world[new_pos] = self.EMPTY
                invalid_poses.add(new_pos)
                continue
            
            poses.add(new_pos)
        
        return list(poses)

    def bfs_search(self, start_pos):
        queue = deque([start_pos])
        visited = set()
        visited.add(start_pos)

        while queue:
            pos = queue.popleft()
            pos_array = np.array(pos)

            neighbors = self.dirs + pos_array 

            valid_indices = self.are_valid_poses(neighbors)
            valid_neighbors = neighbors[valid_indices]

            for neighbor in valid_neighbors:
                neighbor_tuple = tuple(neighbor)
                if not neighbor_tuple in visited and self.world[neighbor_tuple] == self.EMPTY:
                    visited.add(neighbor_tuple)
                    queue.append(neighbor_tuple)
       
        return list(visited)

    def random_pos_at(self, id):
        poses = np.argwhere(self.world == id)
        return tuple(random.choice(poses))

    def are_valid_poses(self, poses):
        within_bounds = (
            (poses[:, 0] >= 0) & (poses[:, 1] >= 0) &
            (poses[:, 0] <= self.world_size - 1) & (poses[:, 1] <= self.world_size - 1)
        )
        return within_bounds



if __name__ == "__main__":
    gen = World_Generator(size=20)
    world = gen.generate_world(n_fields=2, field_size=1)
    print(len(world["walkable"]))