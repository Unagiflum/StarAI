import pygame
import sys
import json
import random
import math
from src.UI import UI
import src.Const as Const
from src.Battle.SpaceObject import Planet, Star
from src.GameObject import SpaceShip, ThrustMarker

def load_settings():
    try:
        with open(Const.GAME_JSON_PATH, 'r') as f:
            loaded_settings = json.load(f)
            return {key: value for key, value in loaded_settings.items()}
    except Exception as e:
        print(f"Error loading settings: {e}. Using default settings.")
        return UI.DEFAULT_KEYS

def get_random_position():
    while True:
        x = random.randint(0, Const.ARENA_SIZE)
        y = random.randint(0, Const.ARENA_SIZE)
        center = Const.ARENA_SIZE // 2
        dx = abs(x - center)
        dy = abs(y - center)
        if dx > Const.CENTER_BUFFER or dy > Const.CENTER_BUFFER:
            return x, y

def validate_ship_positions(pos1, pos2):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, Const.ARENA_SIZE - dx)
    dy = min(dy, Const.ARENA_SIZE - dy)
    return math.sqrt(dx * dx + dy * dy) >= Const.MIN_SHIP_SEPARATION

def get_valid_ship_positions():
    while True:
        pos1 = get_random_position()
        pos2 = get_random_position()
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2

def run(screen, ship1: SpaceShip, ship2: SpaceShip):
    clock = pygame.time.Clock()
    settings = load_settings()
    scale_factor = UI.SCREEN_HEIGHT / Const.ARENA_SIZE

    # Initialize game objects list with stars
    game_objects = []
    for _ in range(Const.STAR_COUNT):
        star = Star()
        star.position = [
            random.randint(0, Const.ARENA_SIZE),
            random.randint(0, Const.ARENA_SIZE)
        ]
        game_objects.append(star)

    pos1, pos2 = get_valid_ship_positions()

    player1 = ship1
    player1.initialize_in_battle(pos1, random.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, random.randint(0, 15))

    game_objects.append(player1)
    game_objects.append(player2)

    player1_sprites = []
    for i in range(16):
        sprite_path = f'Ships/{player1.name}/{player1.name}{i:02d}.png'
        player1_sprites.append(pygame.image.load(sprite_path).convert_alpha())

    player2_sprites = []
    for i in range(16):
        sprite_path = f'Ships/{player2.name}/{player2.name}{i:02d}.png'
        player2_sprites.append(pygame.image.load(sprite_path).convert_alpha())

    planet = Planet()
    planet.position = [Const.ARENA_SIZE / 2, Const.ARENA_SIZE / 2]
    game_objects.append(planet)

    planet_size = int(planet.diameter * scale_factor)
    planet.image = pygame.transform.scale(planet.image, (planet_size, planet_size))

    border_rect = pygame.Rect(0, 0, UI.SCREEN_HEIGHT, UI.SCREEN_HEIGHT)
    border_color = (50, 50, 50)

    running = True

    def draw_marker(marker):
        screen_x = int(marker.position[0] * scale_factor)
        screen_y = int(marker.position[1] * scale_factor)

        positions = [(screen_x, screen_y)]

        if screen_x < 6:
            positions.append((screen_x + UI.SCREEN_HEIGHT, screen_y))
        elif screen_x > UI.SCREEN_HEIGHT - 6:
            positions.append((screen_x - UI.SCREEN_HEIGHT, screen_y))

        if screen_y < 6:
            positions.append((screen_x, screen_y + UI.SCREEN_HEIGHT))
        elif screen_y > UI.SCREEN_HEIGHT - 6:
            positions.append((screen_x, screen_y - UI.SCREEN_HEIGHT))

        if len(positions) > 2:
            positions.append((
                screen_x + (UI.SCREEN_HEIGHT if screen_x < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT),
                screen_y + (UI.SCREEN_HEIGHT if screen_y < UI.SCREEN_HEIGHT // 2 else -UI.SCREEN_HEIGHT)
            ))

        for pos_x, pos_y in positions:
            pygame.draw.circle(
                screen, marker.get_color(), (pos_x, pos_y),
                max(1, int(3 * scale_factor))
            )

    def draw_ship(player, sprites):
        sprite = sprites[player.heading]
        sprite_rect = sprite.get_rect()

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

        # Handle player controls
        player1_forward_pressed = keys[settings[f"Player {player1.player}: Forward"]]
        player2_forward_pressed = keys[settings[f"Player {player2.player}: Forward"]]

        player1.update_timers(player1_forward_pressed)
        player2.update_timers(player2_forward_pressed)

        if keys[settings["Player 1: Left"]]:
            player1.turn_left()
        if keys[settings["Player 1: Right"]]:
            player1.turn_right()
        if player1_forward_pressed:
            marker = player1.apply_thrust()
            if marker:
                game_objects.append(marker)

        if keys[settings["Player 2: Left"]]:
            player2.turn_left()
        if keys[settings["Player 2: Right"]]:
            player2.turn_right()
        if player2_forward_pressed:
            marker = player2.apply_thrust()
            if marker:
                game_objects.append(marker)

        for obj in game_objects[:]:
            if not obj.update():
                game_objects.remove(obj)

            if isinstance(obj, SpaceShip):
                obj.apply_gravity(
                    planet.position,
                    planet.gravity,
                    min_distance=planet.diameter / 2
                )



        # Drawing
        screen.fill(UI.BLACK)
        screen.set_clip(border_rect)

        # Draw stars first (background)
        for obj in game_objects:
            if isinstance(obj, Star):
                star_size = int(obj.diameter * scale_factor)
                scaled_star = pygame.transform.scale(obj.image, (star_size, star_size))
                screen_x = int(obj.position[0] * scale_factor) - star_size // 2
                screen_y = int(obj.position[1] * scale_factor) - star_size // 2
                screen.blit(scaled_star, (screen_x, screen_y))

        # Draw planet
        for obj in game_objects:
            if isinstance(obj, Planet):
                screen.blit(obj.image, (
                    int(obj.position[0] * scale_factor) - planet_size // 2,
                    int(obj.position[1] * scale_factor) - planet_size // 2
                ))

        for obj in game_objects:
            if isinstance(obj, ThrustMarker):
                draw_marker(obj)

        for obj in game_objects:
            if isinstance(obj, SpaceShip):
                draw_ship(obj, player1_sprites if obj == player1 else player2_sprites)

        pygame.draw.rect(screen, border_color, border_rect, 2)
        screen.set_clip(None)
        pygame.display.flip()