import pygame
import sys

from src.UI import UI
from src.Objects.Ships.SpaceShip import SpaceShip, ThrustMarker
from src.Objects.Space.SpaceObject import Planet, Star
from src.Battle.BattleInit import initialize_battle

def run(screen, ship1: SpaceShip, ship2: SpaceShip):
    clock = pygame.time.Clock()
    battle_state = initialize_battle(screen, ship1, ship2)
    settings = battle_state['settings']
    scale_factor = battle_state['scale_factor']
    game_objects = battle_state['game_objects']
    border_rect = battle_state['border_rect']
    border_color = battle_state['border_color']
    player1 = battle_state['player1']
    player2 = battle_state['player2']

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
                planet = next(obj for obj in game_objects if isinstance(obj, Planet))
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
                obj.draw(screen, scale_factor, [0, 0])

        # Draw planet
        for obj in game_objects:
            if isinstance(obj, Planet):
                obj.draw(screen, scale_factor, [0, 0])

        for obj in game_objects:
            if isinstance(obj, ThrustMarker):
                obj.draw(screen, scale_factor, [0, 0])

        for obj in game_objects:
            if isinstance(obj, SpaceShip):
                obj.draw(screen, scale_factor, [0, 0])

        pygame.draw.rect(screen, border_color, border_rect, 2)
        screen.set_clip(None)
        pygame.display.flip()