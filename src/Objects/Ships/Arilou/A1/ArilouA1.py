from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import pygame
import math
import src.const as const

class ArilouA1(Ability):
    def __init__(self, parent):
        super().__init__("ArilouA1", parent)
        ability_data = ABILITIES_DATA["ArilouA1"]
        self.LASER_RANGE = ability_data.get("LASER_RANGE", 400)
        self.LASER_COLOR = tuple(ability_data.get("LASER_COLOR", [255, 255, 0]))
        self.LASER_WIDTH = ability_data.get("LASER_WIDTH", 6)
        self.end_position = [0, 0]
        self.calculate_end_position()

    def calculate_end_position(self):
        if not self.opponent:
            return
        if self.opponent.trackable:
            # Calculate direction to opponent
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            # Handle arena wrapping
            if abs(dx) > const.ARENA_SIZE / 2:
                dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
            if abs(dy) > const.ARENA_SIZE / 2:
                dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

            # Calculate angle and quantize to nearest SHIP_DIRECTION
            angle = math.atan2(dx, -dy)
            direction = round(angle / (2 * math.pi) * const.SHIP_DIRECTIONS) % const.SHIP_DIRECTIONS
        else:
            direction = self.parent.heading  # When untracked, use parent's heading directly

        angle = direction * (2 * math.pi / const.SHIP_DIRECTIONS)

        # Calculate end position using sin/cos flipped to match coordinate system
        self.end_position[0] = (self.position[0] + math.sin(angle) * self.LASER_RANGE)
        self.end_position[1] = (self.position[1] - math.cos(angle) * self.LASER_RANGE)

    def update(self):
        if not self.currently_alive:
            return False

        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation):
        # Update position with parent ship
        self.position = self.parent.position.copy()
        self.calculate_end_position()

        screen_start_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_start_y = int((self.position[1] + translation[1]) * scale_factor)
        screen_end_x = int((self.end_position[0] + translation[0]) * scale_factor)
        screen_end_y = int((self.end_position[1] + translation[1]) * scale_factor)

        # Draw laser at all potential wrap-around positions
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
                        max(1,int(self.LASER_WIDTH * scale_factor))
                    )
