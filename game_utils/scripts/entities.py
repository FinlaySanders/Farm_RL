from game_utils.scripts.utils import Animation, Lerper, load_images

class Animated_Tile:
    def __init__(self, game, animation, pos):
        self.display = game.display
        self.tile_size = game.tile_size
        self.pos = pos
        self.anim = animation.copy()

    def update(self, dt):
        self.anim.update(dt)

    def render(self, offset=[0,0]):
        self.display.blit(self.anim.img(), (self.pos[1] * self.tile_size + offset[1], self.pos[0] * self.tile_size + offset[0]))

class Player:
    def __init__(self, game, pos):
        self.display = game.display
        self.tile_size = game.tile_size
        self.pos = pos
        self.target_pos = pos

        self.anims = {
            "up":Animation(load_images("game_utils/assets/farmer_sprites/up"), anim_dur=0.2, max_loops=1),
            "down":Animation(load_images("game_utils/assets/farmer_sprites/down"), anim_dur=0.2, max_loops=1),
            "left":Animation(load_images("game_utils/assets/farmer_sprites/left"), anim_dur=0.2, max_loops=1),
            "right":Animation(load_images("game_utils/assets/farmer_sprites/right"), anim_dur=0.2, max_loops=1),
        }
        self.current_anim = self.anims["down"].copy()

        self.lerper = Lerper()
    
    def move_to(self, target, duration):
        self.target_pos = target
        self.lerper.set_lerp(self.pos, self.target_pos, duration=duration)

        dy = self.target_pos[0] - self.pos[0]
        dx = self.target_pos[1] - self.pos[1]

        if abs(dx) > abs(dy):
            if dx > 0:
                self.current_anim = self.anims["right"].copy()
            else:
                self.current_anim = self.anims["left"].copy()
        else:
            if dy > 0:
                self.current_anim = self.anims["down"].copy()
            else:
                self.current_anim = self.anims["up"].copy()

    def update(self, dt):
        if self.lerper.lerping:
            self.pos = self.lerper.update(dt)
        self.current_anim.update(dt)

    def render(self, offset=[0,0]):
        self.display.blit(self.current_anim.img(), (self.pos[1] * self.tile_size + offset[1], self.pos[0] * self.tile_size + offset[0]))
