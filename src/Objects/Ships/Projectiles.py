import json
import math
import src.Const as Const
from src.Objects.Object import Object

class ThrustMarker(Object):
    def __init__(self, x, y):
        super().__init__(
            player_num=0,
            max_hp=1,
            start_hp=1,
            inertia=False,
            sprite_location=None,
            sprite_scale=1.0,
            size=[6, 6]
        )
        self.position = [x, y]
        self.life = 30
        self.can_collide = False
        self.can_expire = True
        self.expiration_timer = self.life

    def update(self):
        super().update()
        self.expiration_timer -= 1
        return self.expiration_timer > 0

    def get_color(self):
        fade_ratio = self.expiration_timer / 30
        start_color = (255, 255, 0)
        end_color = (150, 0, 0)
        r = int(start_color[0] * fade_ratio + end_color[0] * (1 - fade_ratio))
        g = int(start_color[1] * fade_ratio + end_color[1] * (1 - fade_ratio))
        b = int(start_color[2] * fade_ratio + end_color[2] * (1 - fade_ratio))
        return (r, g, b)
