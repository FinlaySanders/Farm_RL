import pygame
import torch
import numpy as np

from game_utils.scripts.utils import WallClock, Animation, load_images, load_image
from game_utils.scripts.entities import Player, Animated_Tile

from policynets import CNNPolicyNetwork
from envs.farm_env import Farm_Env
from torch.distributions import Categorical

screen_x, screen_y = 800, 800
display_x, display_y = 80, 80

class Game:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption('RPG Farmer')
        self.screen = pygame.display.set_mode((screen_x, screen_y))
        self.display = pygame.Surface((display_x, display_y))
        self.clock = pygame.time.Clock()

        self.tick_speed = 0.2
        self.action_clock = WallClock(self.tick_speed)

        self.tile_size = 8
        self.textures = {
            "grass":load_image("game_utils/assets/grass.png"),
            "bush":load_image("game_utils/assets/bush.png"),
            "rock":load_image("game_utils/assets/rock.png"),
            "water":load_image("game_utils/assets/water.png"),
            "water_land_above":load_image("game_utils/assets/water_land_above.png"),
            "land_water_below":load_image("game_utils/assets/land_water_below.png"),
            "bridge":load_image("game_utils/assets/bridge.png"),
            "tilled_dirt":load_image("game_utils/assets/tilled_dirt.png"),

            "wheat_sprites":load_images("game_utils/assets/wheat_sprites"),
        }
        self.animations = {
            "fish":Animation(load_images("game_utils/assets/fish_sprites"), anim_dur=0.4, max_loops=-1)
        }

        self.env = Farm_Env(size=10)
        self.env.reset()

        self.model = CNNPolicyNetwork(self.env.world_size, self.env.obs_channels, self.env.act_dim)
        self.model.load_state_dict(torch.load("models/ppo_actor_600509.pth"))

        self.player = Player(self, self.env.agent.tolist())

        self.animated_tiles = []
        #for pos in self.env.render_info["fish"].tolist():
        #    self.animated_tiles.append(Animated_Tile(self, self.animations["fish"], pos))

    def run(self):
        while True:
            self.screen.fill((100,12,145))
            self.display.fill(0)
            dt = self.clock.tick(120) / 1000

            self.render_bg(self, 10, 10)

            self.player.update(dt)
            self.player.render()

            for tile in self.animated_tiles:
                tile.update(dt)
                tile.render()
                #print(tile.anim.dt)
                #print("tile: ", tile.anim.frame)

            if self.action_clock.update(dt):
                #action = torch.argmax(self.model.forward(self.env.get_observation()))
                logits = self.model.forward(self.env.get_observation())
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                
                self.env.step(action)

                if action < 4:
                    new_pos = self.env.agent.tolist()

                    if new_pos != None:
                        self.player.pos = self.player.target_pos
                        self.player.move_to(new_pos, self.tick_speed)
            
            for event in pygame.event.get():
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
        
        for pos in self.env.waypoints.tolist():
            self.display.blit(game.textures["tilled_dirt"], (pos[1] * self.tile_size, pos[0] * self.tile_size))

        for pos in self.env.render_info["bridge"].tolist():
            self.display.blit(game.textures["bridge"], (pos[1] * self.tile_size, pos[0] * self.tile_size))

        for i in range(len(self.env.waypoints)):
            pos = self.env.waypoints.tolist()[i]
            progress = 1 - (self.env.harvestable_steps_remaining[i] / 30)
            boundaries = 1 / len(self.textures["wheat_sprites"])

            idx = int(progress // boundaries)
            self.display.blit(self.textures["wheat_sprites"][idx], (pos[1] * self.tile_size, pos[0] * self.tile_size - self.tile_size/4))
    
if __name__ == "__main__":
    Game().run()


