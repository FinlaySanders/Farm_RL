import pygame
import os

def load_image(path):
    img = pygame.image.load(path).convert_alpha()
    #img = pygame.transform.flip(img, False, False)
    return img 

def load_images(path):
    images = []
    for img_name in sorted(os.listdir(path)):
        images.append(load_image(path + '/' + img_name))
    return images

class Animation:
    def __init__(self, images, anim_dur=1, max_loops=-1):
        self.images = images
        self.anim_dur = anim_dur

        self.max_loop = max_loops
        self.current_loop = 0

        self.done = False
        self.dt = 0
        self.frame = 0

    def copy(self):
        return Animation(self.images, self.anim_dur, self.max_loop)

    def update(self, dt):
        if not self.done:
            self.dt += dt
            if self.dt >= self.anim_dur:
                self.dt -= self.anim_dur
                self.current_loop += 1
                
                if self.current_loop == self.max_loop:
                    self.current_loop = 0
                    self.dt = 0
                    self.frame = 0
                    self.done = True
            else:
                self.frame = int(self.dt / self.anim_dur * len(self.images))

    def img(self):
        return self.images[self.frame]

class Lerper:
    def __init__(self):
        self.lerping = False
    
    def set_lerp(self, start_pos, end_pos, duration):
        self.a = start_pos
        self.b = end_pos
        self.duration = duration
        self.dt = 0.0
        self.lerping = True

    def update(self, dt):
        if self.lerping:
            self.dt += dt 
            t = self.dt / self.duration
            
            if t >= 1:
                self.lerping = False
                return self.b

            return [self.a[0] + (self.b[0] - self.a[0]) * t, self.a[1] + (self.b[1] - self.a[1]) * t]
        return 

class WallClock:
    def __init__(self, delay):
        self.delay = delay
        self.dt = 0
        self.ticks = -1

    def update(self, dt):
        self.dt += dt 
        if self.dt > self.delay:
            self.dt -= self.delay
            self.ticks += 1
            return True
        return False