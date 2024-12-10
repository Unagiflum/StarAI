import pygame
import sys
import json
import random
import math
from src.UI import UI
from src.Battle.SpaceObject import Planet
from src.Ships.SpaceShip import SpaceShip

# Constants
ARENA_SIZE = 3000
SPEED_SCALE = 0.75  # Adjust this to tune all velocities

MIN_SHIP_SEPARATION = ARENA_SIZE // 4
CENTER_BUFFER = ARENA_SIZE // 4  # Ships won't spawn in center quarter of arena

THRUST_MARKER_RADIUS = 3  # Needed for drawing markers

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

def run(screen, ship1: SpaceShip, ship2: SpaceShip):
    """Run the battle simulation."""
    clock = pygame.time.Clock()
    settings = load_settings()
    scale_factor = UI.SCREEN_HEIGHT / ARENA_SIZE

    pos1, pos2 = get_valid_ship_positions()

    player1 = ship1
    player1.initialize_in_battle(pos1, random.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, random.randint(0, 15))

    # Load sprites for player ships
    player1_sprites = []
    for i in range(16):
        sprite_path = f'Ships/{player1.name}/{player1.name}{i:02d}.png'
        player1_sprites.append(pygame.image.load(sprite_path).convert_alpha())

    player2_sprites = []
    for i in range(16):
        sprite_path = f'Ships/{player2.name}/{player2.name}{i:02d}.png'
        player2_sprites.append(pygame.image.load(sprite_path).convert_alpha())

    # Initialize planet at arena center
    planet = Planet()
    planet.position = [ARENA_SIZE / 2, ARENA_SIZE / 2]

    # Scale planet image
    planet_size = int(planet.diameter * scale_factor)
    planet.image = pygame.transform.scale(planet.image, (planet_size, planet_size))

    border_rect = pygame.Rect(0, 0, UI.SCREEN_HEIGHT, UI.SCREEN_HEIGHT)
    border_color = (50, 50, 50)  # Dark gray

    running = True

    def draw_markers(player):
        """Draw the thrust markers for a given player."""
        for marker in player.thrust_markers:
            screen_x = int(marker.position[0] * scale_factor)
            screen_y = int(marker.position[1] * scale_factor)

            positions = [(screen_x, screen_y)]

            # Handle horizontal wrapping
            if screen_x < THRUST_MARKER_RADIUS * 2:
                positions.append((screen_x + UI.SCREEN_HEIGHT, screen_y))
            elif screen_x > UI.SCREEN_HEIGHT - THRUST_MARKER_RADIUS * 2:
                positions.append((screen_x - UI.SCREEN_HEIGHT, screen_y))

            # Handle vertical wrapping
            if screen_y < THRUST_MARKER_RADIUS * 2:
                positions.append((screen_x, screen_y + UI.SCREEN_HEIGHT))
            elif screen_y > UI.SCREEN_HEIGHT - THRUST_MARKER_RADIUS * 2:
                positions.append((screen_x, screen_y - UI.SCREEN_HEIGHT))

            # Add corner position if both horizontally and vertically wrapping
            if len(positions) > 2:
                positions.append((
                    screen_x + (UI.SCREEN_HEIGHT if screen_x < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT),
                    screen_y + (UI.SCREEN_HEIGHT if screen_y < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT)
                ))

            for pos_x, pos_y in positions:
                pygame.draw.circle(
                    screen, marker.get_color(), (pos_x, pos_y),
                    max(1, int(THRUST_MARKER_RADIUS * scale_factor))
                )

    def draw_ship(player, sprites):
        sprite = sprites[player.heading]
        sprite_rect = sprite.get_rect()

        # Apply both SpriteScale and screen scale factor
        total_scale = scale_factor * player.sprite_scale
        scaled_sprite = pygame.transform.scale(
            sprite,
            (int(sprite_rect.width * total_scale),
             int(sprite_rect.height * total_scale))
        )
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int(player.position[0] * scale_factor)
        screen_y = int(player.position[1] * scale_factor)

        positions = [(screen_x, screen_y)]

        # Handle horizontal wrapping
        if screen_x < scaled_rect.width // 2:
            positions.append((screen_x + UI.SCREEN_HEIGHT, screen_y))
        elif screen_x > UI.SCREEN_HEIGHT - scaled_rect.width // 2:
            positions.append((screen_x - UI.SCREEN_HEIGHT, screen_y))

        # Handle vertical wrapping
        if screen_y < scaled_rect.height // 2:
            positions.append((screen_x, screen_y + UI.SCREEN_HEIGHT))
        elif screen_y > UI.SCREEN_HEIGHT - scaled_rect.height // 2:
            positions.append((screen_x, screen_y - UI.SCREEN_HEIGHT))

        # Add corner position if both horizontally and vertically wrapping
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

        # Update timers for each player
        player1_forward_pressed = keys[settings[f"Player {player1.player}: Forward"]]
        player2_forward_pressed = keys[settings[f"Player {player2.player}: Forward"]]

        player1.update_timers(player1_forward_pressed)
        player2.update_timers(player2_forward_pressed)

        # Player 1 controls
        if keys[settings["Player 1: Left"]]:
            player1.turn_left()
        if keys[settings["Player 1: Right"]]:
            player1.turn_right()
        if player1_forward_pressed:
            player1.apply_thrust()

        # Player 2 controls
        if keys[settings["Player 2: Left"]]:
            player2.turn_left()
        if keys[settings["Player 2: Right"]]:
            player2.turn_right()
        if player2_forward_pressed:
            player2.apply_thrust()

        # Apply speed capping and gravity
        for player in [player1, player2]:
            # Cap speed
            speed = math.sqrt(player.velocity[0] ** 2 + player.velocity[1] ** 2)
            max_speed = player.max_thrust * SPEED_SCALE
            if speed > max_speed:
                player.velocity[0] *= max_speed / speed
                player.velocity[1] *= max_speed / speed

            # Apply planet gravity
            dx = planet.position[0] - player.position[0]
            dy = planet.position[1] - player.position[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if player.inertia:
                if distance > planet.diameter / 2:  # Only apply gravity outside planet
                    gravity_force = 1000 * planet.gravity / (distance * distance)
                    player.velocity[0] += gravity_force * dx / distance
                    player.velocity[1] += gravity_force * dy / distance

            # Update position with wrapping
            player.position[0] = (player.position[0] + player.velocity[0]) % ARENA_SIZE
            player.position[1] = (player.position[1] + player.velocity[1]) % ARENA_SIZE

        # Drawing
        screen.fill(UI.BLACK)
        screen.set_clip(border_rect)

        # Draw planet
        planet_x = int(planet.position[0] * scale_factor)
        planet_y = int(planet.position[1] * scale_factor)
        screen.blit(planet.image, (
            planet_x - planet_size // 2,
            planet_y - planet_size // 2
        ))

        # Draw thrust markers for both players
        draw_markers(player1)
        draw_markers(player2)

        # Draw ships
        draw_ship(player1, player1_sprites)
        draw_ship(player2, player2_sprites)

        pygame.draw.rect(screen, border_color, border_rect, 2)
        screen.set_clip(None)
        pygame.display.flip()
