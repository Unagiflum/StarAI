import pygame
import sys

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.Space.space_obj import Asteroid
from src.Battle.battle_init import initialize_battle
from src.Battle.battle_draw import draw_battle
import src.const as const

def run(screen, ship1: SpaceShip, ship2: SpaceShip):
    clock = pygame.time.Clock()

    pygame.mixer.music.load(const.BATTLE_MUSIC_PATH)
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
    pygame.mixer.music.set_volume(const.BATTLE_MUSIC_VOLUME)

    battle_state = initialize_battle(screen, ship1, ship2)
    settings = battle_state['settings']
    game_objects = battle_state['game_objects']
    border_rect = battle_state['border_rect']
    border_color = battle_state['border_color']
    player1 = battle_state['player1']
    player2 = battle_state['player2']

    running = True
    # Track key states
    key_states = {
        settings["Player 1: Forward"]: False,
        settings["Player 1: Left"]: False,
        settings["Player 1: Right"]: False,
        settings["Player 1: Action 1"]: False,
        settings["Player 1: Action 2"]: False,
        settings["Player 2: Forward"]: False,
        settings["Player 2: Left"]: False,
        settings["Player 2: Right"]: False,
        settings["Player 2: Action 1"]: False,
        settings["Player 2: Action 2"]: False,
    }

    while running:

        clock.tick(const.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop()
                    running = False
                elif event.key in key_states:
                    key_states[event.key] = True
                    if event.key in [settings["Player 1: Forward"], settings["Player 1: Left"],
                                     settings["Player 1: Right"], settings["Player 1: Action 1"],
                                     settings["Player 1: Action 2"]]:
                        p1_objects = player1.handle_actions(
                            event.key, True,
                            settings["Player 1: Forward"],
                            settings["Player 1: Left"],
                            settings["Player 1: Right"],
                            settings["Player 1: Action 1"],
                            settings["Player 1: Action 2"]
                        )
                        game_objects.extend(p1_objects)
                    else:
                        p2_objects = player2.handle_actions(
                            event.key, True,
                            settings["Player 2: Forward"],
                            settings["Player 2: Left"],
                            settings["Player 2: Right"],
                            settings["Player 2: Action 1"],
                            settings["Player 2: Action 2"]
                        )
                        game_objects.extend(p2_objects)

            elif event.type == pygame.KEYUP:
                if event.key in key_states:
                    key_states[event.key] = False
                    if event.key in [settings["Player 1: Forward"], settings["Player 1: Left"],
                                     settings["Player 1: Right"], settings["Player 1: Action 1"],
                                     settings["Player 1: Action 2"]]:
                        p1_objects = player1.handle_actions(
                            event.key, False,
                            settings["Player 1: Forward"],
                            settings["Player 1: Left"],
                            settings["Player 1: Right"],
                            settings["Player 1: Action 1"],
                            settings["Player 1: Action 2"]
                        )
                        game_objects.extend(p1_objects)
                    else:
                        p2_objects = player2.handle_actions(
                            event.key, False,
                            settings["Player 2: Forward"],
                            settings["Player 2: Left"],
                            settings["Player 2: Right"],
                            settings["Player 2: Action 1"],
                            settings["Player 2: Action 2"]
                        )
                        game_objects.extend(p2_objects)


        # Call handle_actions with no key for repeating actions
        p1_objects = player1.handle_actions(
            None, False,
            settings["Player 1: Forward"],
            settings["Player 1: Left"],
            settings["Player 1: Right"],
            settings["Player 1: Action 1"],
            settings["Player 1: Action 2"]
        )
        game_objects.extend(p1_objects)

        p2_objects = player2.handle_actions(
            None, False,
            settings["Player 2: Forward"],
            settings["Player 2: Left"],
            settings["Player 2: Right"],
            settings["Player 2: Action 1"],
            settings["Player 2: Action 2"]
        )
        game_objects.extend(p2_objects)

        # Update tracking arrays for all ships and abilities
        for obj in game_objects:
            if isinstance(obj, (SpaceShip, Ability)):
                obj.friendly_objects = [x for x in game_objects if isinstance(x, Ability) and x.player == obj.player]
                obj.enemy_objects = [x for x in game_objects if isinstance(x, Ability) and x.player != obj.player]
                obj.asteroids = [x for x in game_objects if isinstance(x, Asteroid)]

        # Drawing
        draw_battle(screen, game_objects, border_rect, border_color)

        # Update all objects
        for obj in game_objects[:]:
            if not obj.update():
                game_objects.remove(obj)
