import pygame
import sys
import json
import random
import math
from src.UI import UI

# Constants
ARENA_SIZE = 8000
SPEED_SCALE = 0.75  # Adjust this to tune all velocities
TURN_WAIT_SCALE = 2.0
THRUST_WAIT_SCALE = 2.0

MIN_SHIP_SEPARATION = ARENA_SIZE//4
CENTER_BUFFER = ARENA_SIZE // 4  # Ships won't spawn in center quarter of arena

# Thrust marker constants
THRUST_MARKER_RADIUS = 3
THRUST_MARKER_LIFE = 30  # frames
THRUST_MARKER_START_COLOR = (255, 255, 0)  # Yellow
THRUST_MARKER_END_COLOR = (150, 0, 0)  # Red

def load_settings():
    """Load control settings."""
    try:
        with open('Config/Gamesettings.json', 'r') as f:
            loaded_settings = json.load(f)
            return {key: value for key, value in loaded_settings.items()}
    except Exception as e:
        print(f"Error loading settings: {e}. Using default settings.")
        return UI.DEFAULT_KEYS


def get_random_position():
    """Get random position avoiding center area."""
    while True:
        x = random.randint(0, ARENA_SIZE)
        y = random.randint(0, ARENA_SIZE)

        # Check if point is in center buffer
        center = ARENA_SIZE // 2
        dx = abs(x - center)
        dy = abs(y - center)

        if dx > CENTER_BUFFER or dy > CENTER_BUFFER:
            return x, y

def validate_ship_positions(pos1, pos2):
    """Check if two positions are far enough apart."""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])

    # Account for arena wrapping
    dx = min(dx, ARENA_SIZE - dx)
    dy = min(dy, ARENA_SIZE - dy)

    return math.sqrt(dx * dx + dy * dy) >= MIN_SHIP_SEPARATION

def get_valid_ship_positions():
    """Get two valid ship positions."""
    while True:
        pos1 = get_random_position()
        pos2 = get_random_position()
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2


class ThrustMarker:
    def __init__(self, x, y):
        self.position = [x, y]
        self.life = THRUST_MARKER_LIFE

    def update(self):
        self.life -= 1
        return self.life > 0

    def get_color(self):
        fade_ratio = self.life / THRUST_MARKER_LIFE
        return tuple(int(start * fade_ratio + end * (1 - fade_ratio))
                     for start, end in zip(THRUST_MARKER_START_COLOR, THRUST_MARKER_END_COLOR))


class Ship:
    def __init__(self, ship_obj, position, heading):
        self.obj = ship_obj
        self.position = list(position)
        self.heading = heading
        self.velocity = [0.0, 0.0]
        self.rotation = heading * 22.5
        self.thrust_timer = 0
        self.turn_timer = 0
        self.thrust_markers = []

        self.sprites = []
        for i in range(16):
            sprite_path = f'Ships/{ship_obj.name}/{ship_obj.name}{i:02d}.png'
            self.sprites.append(pygame.image.load(sprite_path).convert_alpha())

        # Calculate thrust offset using upward-facing sprite (index 0)
        sprite_rect = self.sprites[0].get_rect()
        self.thrust_offset = (sprite_rect.height / 2) + (THRUST_MARKER_RADIUS * 2)

    def can_thrust(self):
        return self.thrust_timer == 0

    def can_turn(self):
        return self.turn_timer == 0

    def update_timers(self):
        if self.thrust_timer > 0:
            self.thrust_timer -= 1
        if self.turn_timer > 0:
            self.turn_timer -= 1

        settings = load_settings()
        player_prefix = f"Player {self.obj.player}: "
        forward_key = settings[player_prefix + "Forward"]

        if not self.obj.inertia and self.thrust_timer == 0 and not pygame.key.get_pressed()[forward_key]:
            self.velocity = [0.0, 0.0]

    def apply_thrust(self):
        if self.can_thrust():
            angle_rad = math.radians(self.rotation)
            if self.obj.inertia:
                self.velocity[0] += math.sin(angle_rad) * self.obj.thrust_increment * SPEED_SCALE
                self.velocity[1] -= math.cos(angle_rad) * self.obj.thrust_increment * SPEED_SCALE
            else:
                speed = self.obj.max_thrust * SPEED_SCALE
                self.velocity[0] = math.sin(angle_rad) * speed
                self.velocity[1] = -math.cos(angle_rad) * speed

            # Calculate thrust marker position based on offset
            marker_x = self.position[0] - math.sin(angle_rad) * self.thrust_offset
            marker_y = self.position[1] + math.cos(angle_rad) * self.thrust_offset

            self.thrust_timer = int(self.obj.thrust_wait * THRUST_WAIT_SCALE)
            self.thrust_markers.append(ThrustMarker(marker_x, marker_y))

    def update_thrust_markers(self):
        self.thrust_markers = [marker for marker in self.thrust_markers if marker.update()]

    def turn_left(self):
        if self.can_turn():
            self.heading = (self.heading - 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.obj.turn_wait * TURN_WAIT_SCALE)

    def turn_right(self):
        if self.can_turn():
            self.heading = (self.heading + 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.obj.turn_wait * TURN_WAIT_SCALE)


def run(screen, ship1, ship2):
    """Run the battle simulation."""
    clock = pygame.time.Clock()
    settings = load_settings()
    scale_factor = UI.SCREEN_HEIGHT / ARENA_SIZE

    pos1, pos2 = get_valid_ship_positions()
    player1 = Ship(ship1, pos1, random.randint(0, 15))
    player2 = Ship(ship2, pos2, random.randint(0, 15))

    border_rect = pygame.Rect(0, 0, UI.SCREEN_HEIGHT, UI.SCREEN_HEIGHT)
    border_color = (50, 50, 50)  # Dark gray

    running = True
    while running:
        clock.tick(UI.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()

        player1.update_timers()
        player2.update_timers()

        if keys[settings["Player 1: Left"]]:
            player1.turn_left()
        if keys[settings["Player 1: Right"]]:
            player1.turn_right()
        if keys[settings["Player 1: Forward"]]:
            player1.apply_thrust()

        if keys[settings["Player 2: Left"]]:
            player2.turn_left()
        if keys[settings["Player 2: Right"]]:
            player2.turn_right()
        if keys[settings["Player 2: Forward"]]:
            player2.apply_thrust()


        for player in [player1, player2]:
            speed = math.sqrt(player.velocity[0] ** 2 + player.velocity[1] ** 2)
            max_speed = player.obj.max_thrust * SPEED_SCALE
            if speed > max_speed:
                player.velocity[0] *= max_speed / speed
                player.velocity[1] *= max_speed / speed

            player.position[0] = (player.position[0] + player.velocity[0]) % ARENA_SIZE
            player.position[1] = (player.position[1] + player.velocity[1]) % ARENA_SIZE
            player.update_thrust_markers()

        screen.fill(UI.BLACK)
        screen.set_clip(border_rect)

        for player in [player1, player2]:
            for marker in player.thrust_markers:
                screen_x = int(marker.position[0] * scale_factor)
                screen_y = int(marker.position[1] * scale_factor)

                positions = [(screen_x, screen_y)]

                if screen_x < THRUST_MARKER_RADIUS * 2:
                    positions.append((screen_x + UI.SCREEN_HEIGHT, screen_y))
                elif screen_x > UI.SCREEN_HEIGHT - THRUST_MARKER_RADIUS * 2:
                    positions.append((screen_x - UI.SCREEN_HEIGHT, screen_y))

                if screen_y < THRUST_MARKER_RADIUS * 2:
                    positions.append((screen_x, screen_y + UI.SCREEN_HEIGHT))
                elif screen_y > UI.SCREEN_HEIGHT - THRUST_MARKER_RADIUS * 2:
                    positions.append((screen_x, screen_y - UI.SCREEN_HEIGHT))

                # Add corner positions when wrapping both horizontally and vertically
                if len(positions) > 2:
                    positions.append((
                        screen_x + (UI.SCREEN_HEIGHT if screen_x < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT),
                        screen_y + (UI.SCREEN_HEIGHT if screen_y < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT)
                    ))

                for pos_x, pos_y in positions:
                    pygame.draw.circle(screen, marker.get_color(), (pos_x, pos_y),
                                       max(1, int(THRUST_MARKER_RADIUS * scale_factor)))


        for player in [player1, player2]:
            sprite = player.sprites[player.heading]
            sprite_rect = sprite.get_rect()

            scaled_sprite = pygame.transform.scale(
                sprite,
                (int(sprite_rect.width * scale_factor),
                 int(sprite_rect.height * scale_factor))
            )
            scaled_rect = scaled_sprite.get_rect()

            screen_x = int(player.position[0] * scale_factor)
            screen_y = int(player.position[1] * scale_factor)

            positions = [(screen_x, screen_y)]

            if screen_x < scaled_rect.width // 2:
                positions.append((screen_x + UI.SCREEN_HEIGHT, screen_y))
            elif screen_x > UI.SCREEN_HEIGHT - scaled_rect.width // 2:
                positions.append((screen_x - UI.SCREEN_HEIGHT, screen_y))

            if screen_y < scaled_rect.height // 2:
                positions.append((screen_x, screen_y + UI.SCREEN_HEIGHT))
            elif screen_y > UI.SCREEN_HEIGHT - scaled_rect.height // 2:
                positions.append((screen_x, screen_y - UI.SCREEN_HEIGHT))

            if len(positions) > 2:
                positions.append((
                    screen_x + (UI.SCREEN_HEIGHT if screen_x < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT),
                    screen_y + (UI.SCREEN_HEIGHT if screen_y < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT)
                ))

            for pos_x, pos_y in positions:
                screen.blit(scaled_sprite, (
                    pos_x - scaled_rect.width // 2,
                    pos_y - scaled_rect.height // 2
                ))

        pygame.draw.rect(screen, border_color, border_rect, 2)
        screen.set_clip(None)
        pygame.display.flip()