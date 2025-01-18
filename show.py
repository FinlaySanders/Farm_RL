import argparse
import pygame
import torch
import numpy as np

from render.scripts.utils import WallClock, Animation, load_images, load_image
from render.scripts.entities import Player

import torch.nn as nn
from torch.distributions import Categorical

from farm.env import FarmEnv
from ppo import Agent

class Render:
    def __init__(self, env, model_name):
        self.env = env
        self.env.reset()

        self.tile_size = 8
        screen_size = 800
        display_size = self.tile_size * self.env.world_size

        pygame.init()

        pygame.display.set_caption('RPG Farmer')
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.display = pygame.Surface((display_size, display_size))
        self.clock = pygame.time.Clock()

        self.tick_speed = 0.2
        self.action_clock = WallClock(self.tick_speed)

        self.textures = {
            "grass":load_image("render/assets/grass.png"),
            "bush":load_image("render/assets/bush.png"),
            "rock":load_image("render/assets/rock.png"),
            "water":load_image("render/assets/water.png"),
            "water_land_above":load_image("render/assets/water_land_above.png"),
            "land_water_below":load_image("render/assets/land_water_below.png"),
            "bridge":load_image("render/assets/bridge.png"),
            "tilled_dirt":load_image("render/assets/tilled_dirt.png"),
            "shovel1":load_image("render/assets/shovel.png"),
            "shovel2":load_image("render/assets/shovel2.png"),
            "wheat_sprites":load_images("render/assets/wheat_sprites"),
        }


        self.agent = Agent(None)
        self.agent.load_state_dict(torch.load(model_name, weights_only=True))
        self.model_paused = False
        self.lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size),
            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size),
        )

        self.player = Player(self, self.env.agent)

    def run(self):
        while True:
            self.screen.fill((100,12,145))
            self.display.fill(0)
            dt = self.clock.tick(120) / 1000

            render_info = self.env.render()
            self.render_world(render_info)
            self.render_agent_view()

            self.player.update(dt)
            self.player.render()

            if self.action_clock.update(dt) and not self.model_paused:
                obs = torch.tensor(self.env.get_observation(), dtype=torch.float32).unsqueeze(0)
                action = self.choose_action(obs, deterministic=False)
                
                self.env.step(action)
            
                if action < 4:
                    self.player.pos = self.player.target_pos
                    self.player.move_to(self.env.agent, self.tick_speed)

            self.handle_input()

            self.screen.blit(pygame.transform.scale(self.display, self.screen.get_size()), (0, 0))
            pygame.display.update()
    
    def choose_action(self, obs, deterministic=True):
        obs = self.agent.preprocess(obs)
        hidden, self.lstm_state = self.agent.get_states(obs, self.lstm_state, torch.tensor([0]))
        logits = self.agent.actor(hidden)

        if deterministic:
            return torch.argmax(logits).item()
        else:
            return Categorical(logits=logits).sample()
            
    def render_world(self, render_info):
        for j in range(self.env.world_size):
            for i in range(self.env.world_size):
                self.display.blit(self.textures["grass"], (i * self.tile_size, j * self.tile_size))
        
        terrain = np.array(self.env.terrain)
        water_poses = np.argwhere(terrain == 1).tolist()
        for pos in water_poses:
            if [pos[0] - 1, pos[1]] not in water_poses and pos[0] - 1 > 0:
                self.display.blit(self.textures["land_water_below"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
            elif [pos[0] - 1, pos[1]] in water_poses and [pos[0] - 2, pos[1]] not in water_poses and pos[0] - 2 > 0:
                self.display.blit(self.textures["water_land_above"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
            else:
                self.display.blit(self.textures["water"], (pos[1] * self.tile_size, pos[0] * self.tile_size))

        for pos in render_info["crops"]:
            progress = 1 - (self.env.crops[pos] / self.env.crop_time)
            boundaries = 1 / (len(self.textures["wheat_sprites"]) - 1)
            idx = int(progress // boundaries)

            self.display.blit(self.textures["tilled_dirt"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
            self.display.blit(self.textures["wheat_sprites"][idx], (pos[1] * self.tile_size, pos[0] * self.tile_size - self.tile_size/4))
        
    def render_agent_view(self):
        agent_y, agent_x = self.player.pos
        view_radius = self.env.view_size

        rect_x = (agent_x - view_radius) * self.tile_size
        rect_y = (agent_y - view_radius) * self.tile_size

        rect_width = (view_radius * 2 + 1) * self.tile_size
        rect_height = (view_radius * 2 + 1) * self.tile_size

        pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(rect_x, rect_y, rect_width, rect_height), width=1)
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                action = -1
                if event.key == pygame.K_UP:
                    action = 2
                if event.key == pygame.K_LEFT:
                    action = 3
                if event.key == pygame.K_DOWN:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 0
                if event.key == pygame.K_SPACE:
                    action = 4
                if event.key == pygame.K_c:
                    action = 5
                if event.key == pygame.K_p:
                    self.model_paused = not self.model_paused

                if action != -1:
                    self.env.step(action)
                    if action < 4:
                        self.player.pos = self.player.target_pos
                        self.player.move_to(self.env.agent, self.tick_speed)

                    
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
if __name__ == "__main__":
    cmd = argparse.ArgumentParser()

    cmd.add_argument(
        "-m",
        "--model",
        help="choose model to be shown",
        dest="model",
        type=str,
        default="example.pth",
    )

    args = cmd.parse_args()

    env = FarmEnv()
    ren = Render(env=env, model_name="models/" + args.model)
    ren.run()