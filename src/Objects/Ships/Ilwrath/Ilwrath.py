from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.Ilwrath.A1.IlwrathA1 import IlwrathA1
from src.Objects.Ships.Ilwrath.A2.IlwrathA2 import IlwrathA2
import src.const as const
import pygame
import math


class Ilwrath(SpaceShip):
    # Removed the shared_sprites_white dictionary since we no longer need white variants
    _shared_sprites_black = {}
    _uncloak_sound = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]
        self.FADE_DURATION = ship_data.get("FADE_DURATION", 8)
        self.fade_timer = self.FADE_DURATION
        self.ship_name = ship_name

        # Load shared resources if not already loaded
        if Ability.sound_enabled and not self._uncloak_sound:
            try:
                sound_path = self.sprite_location / "A2" / "IlwrathA2end.wav"
                self._uncloak_sound = pygame.mixer.Sound(str(sound_path))
                self._uncloak_sound.set_volume(const.SOUND_EFFECT_VOLUME)
            except pygame.error:
                print(f"Error loading uncloak sound for {ship_name}")

        # Only load black sprite variants (removed creation of white sprites)
        if ship_name not in self._shared_sprites_black:
            self._shared_sprites_black[ship_name] = []
            for i in range(const.SHIP_DIRECTIONS):
                sprite = self.sprites[i].copy()

                # Create black variant
                black_sprite = sprite.copy()
                black_pixels = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
                pygame.draw.rect(black_pixels, (0, 0, 0, 255), black_pixels.get_rect())
                black_sprite.blit(black_pixels, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                self._shared_sprites_black[ship_name].append(black_sprite)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            if self.cloaked:
                if self.fade_timer >= self.FADE_DURATION:
                    self.face_opponent()
                self.uncloak()
            ability_obj = IlwrathA1(self)
            if ability_obj.launch_sound:
                ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2():
            if self.cloaked:
                self.uncloak()
                if Ability.sound_enabled and self._uncloak_sound:
                    self._uncloak_sound.play()
            else:
                self.current_energy -= self.a2_cost
                self.cloak()
                ability_obj = IlwrathA2(self)
                if ability_obj.launch_sound:
                    ability_obj.launch_sound.play()
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
        return None

    def perform_action3(self):
        return None, False

    def face_opponent(self):
        if self.opponent and self.opponent.trackable:
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            if abs(dx) > const.ARENA_SIZE / 2:
                dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
            if abs(dy) > const.ARENA_SIZE / 2:
                dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360

            direction_step = 360 / const.SHIP_DIRECTIONS
            self.heading = int(target_angle / direction_step) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE

    def can_action2(self):
        if self.cloaked:
            return self.action2_timer == 0
        else:
            return self.action2_timer == 0 and self.current_energy >= self.a2_cost

    def cloak(self):
        self.cloaked = True
        self.trackable = False
        self.fade_timer = 0

    def uncloak(self):
        self.cloaked = False
        self.trackable = True

    def set_sprite(self):
        black_sprite = self._shared_sprites_black[self.ship_name][self.heading]
        normal_sprite = self.sprites[self.heading]

        # If we're still within the fade timer, do a fade transition; otherwise pick final
        if self.fade_timer < self.FADE_DURATION:
            progress = self.fade_timer / self.FADE_DURATION
            final_sprite = pygame.Surface(normal_sprite.get_size(), pygame.SRCALPHA)

            if self.cloaked:
                # Fade from normal → black
                normal_copy = normal_sprite.copy()
                black_copy = black_sprite.copy()
                alpha_normal = int(255 * (1 - progress))
                alpha_black = int(255 * progress)
                normal_copy.set_alpha(alpha_normal)
                black_copy.set_alpha(alpha_black)
                final_sprite.blit(normal_copy, (0, 0))
                final_sprite.blit(black_copy, (0, 0))
            else:
                final_sprite = normal_sprite

            self.fade_timer += 1
        else:
            final_sprite = black_sprite if self.cloaked else normal_sprite
        return final_sprite
