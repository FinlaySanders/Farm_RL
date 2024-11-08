import pygame
import torch
import numpy as np

from game_utils.scripts.utils import WallClock, Animation, load_images, load_image
from game_utils.scripts.entities import Player, Animated_Tile

from policynets import CNNPolicyNetwork
from envs.farm_env import Farm_Env
from torch.distributions import Categorical

class Game:
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
            "grass":load_image("game_utils/assets/grass.png"),
            "bush":load_image("game_utils/assets/bush.png"),
            "rock":load_image("game_utils/assets/rock.png"),
            "water":load_image("game_utils/assets/water.png"),
            "water_land_above":load_image("game_utils/assets/water_land_above.png"),
            "land_water_below":load_image("game_utils/assets/land_water_below.png"),
            "bridge":load_image("game_utils/assets/bridge.png"),
            "tilled_dirt":load_image("game_utils/assets/tilled_dirt.png"),
            "shovel1":load_image("game_utils/assets/shovel.png"),
            "shovel2":load_image("game_utils/assets/shovel2.png"),

            "wheat_sprites":load_images("game_utils/assets/wheat_sprites"),
        }
        self.animations = {
            "fish":Animation(load_images("game_utils/assets/fish_sprites"), anim_dur=0.4, max_loops=-1)
        }

        self.model = CNNPolicyNetwork(self.env.world_size, self.env.obs_channels, self.env.act_dim)
        self.model.load_state_dict(torch.load("models/" + model_name))
        self.model_paused = False

        self.player = Player(self, self.env.agent.tolist())

        self.animated_tiles = []
        #for pos in self.env.render_info["fish"].tolist():
        #    self.animated_tiles.append(Animated_Tile(self, self.animations["fish"], pos))

    def run(self):
        while True:
            self.screen.fill((100,12,145))
            self.display.fill(0)
            dt = self.clock.tick(120) / 1000

            self.render_bg(self, self.env.world_size, self.env.world_size)

            self.player.update(dt)
            self.player.render()

            for tile in self.animated_tiles:
                tile.update(dt)
                tile.render()

            if self.action_clock.update(dt) and not self.model_paused:
                #action = torch.argmax(self.model.forward(self.env.get_observation()))
                obs = self.env.get_observation()
                logits = self.model.forward(obs)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                
                self.env.step(action)

                if action < 4:
                    new_pos = self.env.agent.tolist()

                    if new_pos != None:
                        self.player.pos = self.player.target_pos
                        self.player.move_to(new_pos, self.tick_speed)
            
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
                            new_pos = self.env.agent.tolist()

                            if new_pos != None:
                                self.player.pos = self.player.target_pos
                                self.player.move_to(new_pos, self.tick_speed)

                    
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            self.screen.blit(pygame.transform.scale(self.display, self.screen.get_size()), (0, 0))
            pygame.display.update()
            
    def render_bg(self, game, width, height):
        for j in range(height):
            for i in range(width):
                self.display.blit(self.textures["grass"], (i * self.tile_size, j * self.tile_size))
        
        # this is slightly cursed but works
        water_poses = self.env.render_info["water"].tolist()
        for pos in water_poses:
            if [pos[0] - 1, pos[1]] not in water_poses and pos[0] - 1 > 0:
                self.display.blit(game.textures["land_water_below"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
            elif [pos[0] - 1, pos[1]] in water_poses and [pos[0] - 2, pos[1]] not in water_poses and pos[0] - 2 > 0:
                self.display.blit(game.textures["water_land_above"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
            else:
                self.display.blit(game.textures["water"], (pos[1] * self.tile_size, pos[0] * self.tile_size))
        
        for pos in self.env.crops.tolist():
            self.display.blit(game.textures["tilled_dirt"], (pos[1] * self.tile_size, pos[0] * self.tile_size))

        for pos in self.env.render_info["bridge"].tolist():
            self.display.blit(game.textures["bridge"], (pos[1] * self.tile_size, pos[0] * self.tile_size))

        for i in range(len(self.env.crops)):
            pos = self.env.crops.tolist()[i]
            progress = 1 - (self.env.managers[0].crop_growth_steps_remaining[i] / self.env.managers[0].crop_growth_steps)
            boundaries = 1 / len(self.textures["wheat_sprites"])

            idx = int(progress // boundaries)
            self.display.blit(self.textures["wheat_sprites"][idx], (pos[1] * self.tile_size, pos[0] * self.tile_size - self.tile_size/4))
        
        for i, pos in enumerate(self.env.tool_manager.tool_poses):
            if i != self.env.tool_manager.active_tool:
                self.display.blit(self.textures[f"shovel{i+1}"], (pos[1] * self.tile_size, pos[0] * self.tile_size - self.tile_size/4))
            else:
                self.display.blit(self.textures[f"shovel{i+1}"], (self.player.pos[1] * self.tile_size, (self.player.pos[0] - 1) * self.tile_size - self.tile_size/4))