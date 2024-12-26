from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Ilwrath.A1.IlwrathA1 import IlwrathA1
from src.Objects.Ships.Ilwrath.A2.IlwrathA2 import IlwrathA2
from src.Objects.object import ThrustMarker
import src.const as const
import pygame
import math


class Ilwrath(SpaceShip):
    _shared_sprites_black = {}
    _shared_sprites_white = {}
    _uncloak_sound = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]
        self.trackable = True
        self.cloaked = False
        self.fade_timer = 0
        self.fade_duration = 10

        # Load shared resources if not already loaded
        if not self._uncloak_sound:
            try:
                sound_path = self.sprite_location / "A2" / "IlwrathA2end.wav"
                self._uncloak_sound = pygame.mixer.Sound(str(sound_path))
                self._uncloak_sound.set_volume(const.SOUND_EFFECT_VOLUME)
            except pygame.error:
                print(f"Error loading uncloak sound for {ship_name}")

        # Load black and white sprite variants if not already loaded
        if ship_name not in self._shared_sprites_black:
            self._shared_sprites_black[ship_name] = []
            self._shared_sprites_white[ship_name] = []
            for i in range(const.SHIP_DIRECTIONS):
                sprite = self.sprites[i].copy()

                # Create black variant
                black_sprite = sprite.copy()
                black_pixels = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
                pygame.draw.rect(black_pixels, (0, 0, 0, 255), black_pixels.get_rect())
                black_sprite.blit(black_pixels, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                self._shared_sprites_black[ship_name].append(black_sprite)

                # Create white variant
                white_sprite = sprite.copy()
                white_pixels = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
                pygame.draw.rect(white_pixels, (255, 255, 255, 255), white_pixels.get_rect())
                white_sprite.blit(white_pixels, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                self._shared_sprites_white[ship_name].append(white_sprite)


    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            if self.cloaked:
                self.face_opponent()
                self.uncloak()
            ability_obj = IlwrathA1(self)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2():
            if self.cloaked:
                self.uncloak()
                if self._uncloak_sound:
                    self._uncloak_sound.play()
            else:
                self.current_energy -= self.a2_cost
                self.cloak()
                ability_obj = IlwrathA2(self)
                if ability_obj.launch_sound: ability_obj.launch_sound.play()
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
        return None

    def perform_action3(self):
        return None, False

    def cloak(self):
        self.cloaked = True
        self.trackable = False

    def uncloak(self):
        self.cloaked = False
        self.trackable = True

    def face_opponent(self):
        if self.opponent:
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

    def apply_thrust(self, max_thrust, thrust_increment, angle, can_thrust, make_marker = True):
        if can_thrust:
            angle_rad = math.radians(self.rotation + angle)
            thrust_direction = [math.sin(angle_rad), -math.cos(angle_rad)]

            if self.inertia:
                new_velocity = [
                    self.velocity[0] + thrust_direction[0] * thrust_increment,
                    self.velocity[1] + thrust_direction[1] * thrust_increment
                ]

                speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
                scale = 1.0

                _, planet_distance = self.distance_to(self.planet)
                if speed > max_thrust and planet_distance > const.GRAVITY_RANGE:
                    scale = max_thrust / speed
                if speed > const.MAX_GRAV_WHIP:
                    scale = const.MAX_GRAV_WHIP / speed

                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                diff_vector = [target_velocity[0] - self.velocity[0], target_velocity[1] - self.velocity[1]]

                diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
                if diff_magnitude > thrust_increment:
                    scale = thrust_increment / diff_magnitude
                    self.add_impulse(diff_vector[0] * scale, diff_vector[1] * scale)
                else:
                    self.add_impulse(diff_vector[0] , diff_vector[1] )
            else:
                self.add_impulse(
                    thrust_direction[0] * max_thrust,
                    thrust_direction[1] * max_thrust
                )

            self.thrust_timer = int(self.thrust_wait * const.THRUST_WAIT_SCALE)
            make_marker = not self.cloaked
            if make_marker:
                marker_x, marker_y = self.get_thrust_marker_position()
                marker = ThrustMarker(marker_x, marker_y)
                return marker
        return None

    def draw(self, screen, scale_factor, translation):
        sprite = self.sprites[self.heading]
        sprite_rect = sprite.get_rect()

        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        # Calculate screen position with translation
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        # Draw the ship at all potential wrap-around positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                # Only draw if the position would be visible
                if (0 <= pos_x <= const.SCREEN_HEIGHT and
                        0 <= pos_y <= const.SCREEN_HEIGHT) and not self.cloaked:
                    screen.blit(scaled_sprite, (
                        const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))