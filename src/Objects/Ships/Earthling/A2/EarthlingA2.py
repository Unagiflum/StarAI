from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import pygame
import math
import src.const as const


class EarthlingA2(Ability):
    def __init__(self, parent, target=None):
        super().__init__("EarthlingA2", parent)
        ability_data = ABILITIES_DATA["EarthlingA2"]
        self.LASER_RANGE = ability_data.get("LASER_RANGE", 400)
        self.LASER_COLOR = tuple(ability_data.get("LASER_COLOR", [255, 255, 255]))
        self.LASER_WIDTH = ability_data.get("LASER_WIDTH", 4)
        self.target = target
        self.end_position = [0, 0]
        if self.target:
            self.calculate_end_position()

    def _distance_to(self, target):
        dx = target.position[0] - self.parent.position[0]
        dy = target.position[1] - self.parent.position[1]
        if abs(dx) > const.ARENA_SIZE / 2:
            dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
        if abs(dy) > const.ARENA_SIZE / 2:
            dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE
        return (dx * dx + dy * dy) ** 0.5

    def _is_in_range(self, target):
        return self._distance_to(target) <= self.LASER_RANGE

    def get_shots(self, max_shots):
        valid_targets = []

        if self.parent.opponent and self._is_in_range(self.parent.opponent) and self.parent.opponent.trackable:
            valid_targets.append(self.parent.opponent)

        for obj in sorted(self.parent.enemy_objects, key=self._distance_to):
            if self._is_in_range(obj):
                valid_targets.append(obj)

        for obj in sorted(self.parent.friendly_objects, key=self._distance_to):
            if self._is_in_range(obj):
                valid_targets.append(obj)

        for obj in sorted(self.parent.asteroids, key=self._distance_to):
            if self._is_in_range(obj):
                valid_targets.append(obj)

        shots = min(max_shots, len(valid_targets))
        if shots == 0:
            return None

        return [EarthlingA2(self.parent, valid_targets[i]) for i in range(shots)]

    def calculate_end_position(self):
        dx = self.target.position[0] - self.position[0]
        dy = self.target.position[1] - self.position[1]

        if abs(dx) > const.ARENA_SIZE / 2:
            dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
        if abs(dy) > const.ARENA_SIZE / 2:
            dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

        angle = math.atan2(dy, dx)
        self.end_position[0] = (self.position[0] + math.cos(angle) * self.LASER_RANGE)
        self.end_position[1] = (self.position[1] + math.sin(angle) * self.LASER_RANGE)

    def update(self):
        if not self.currently_alive:
            return False
        self.expiration_timer -= 1
        return self.expiration_timer > 0

    def draw(self, screen, scale_factor, translation):
        self.position = self.parent.position.copy()
        self.calculate_end_position()

        screen_start_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_start_y = int((self.position[1] + translation[1]) * scale_factor)
        screen_end_x = int((self.end_position[0] + translation[0]) * scale_factor)
        screen_end_y = int((self.end_position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                start_x = screen_start_x + dx * const.ARENA_SIZE * scale_factor
                start_y = screen_start_y + dy * const.ARENA_SIZE * scale_factor
                end_x = screen_end_x + dx * const.ARENA_SIZE * scale_factor
                end_y = screen_end_y + dy * const.ARENA_SIZE * scale_factor

                if (0 <= start_x <= const.ARENA_SIZE and
                    0 <= start_y <= const.ARENA_SIZE) or \
                        (0 <= end_x <= const.ARENA_SIZE and
                         0 <= end_y <= const.ARENA_SIZE):
                    pygame.draw.line(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + start_x, start_y),
                        (const.SCREEN_LEFT + end_x, end_y),
                        max(1, int(self.LASER_WIDTH * scale_factor))
                    )