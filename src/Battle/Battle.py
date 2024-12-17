import pygame
import sys

from src.Objects.Ships.SpaceShip import SpaceShip
from src.Battle.BattleInit import initialize_battle
from src.Battle.BattleDraw import draw_battle
import src.Const as Const

def run(screen, ship1: SpaceShip, ship2: SpaceShip):
    clock = pygame.time.Clock()

    pygame.mixer.music.load(Const.BATTLE_MUSIC_PATH)
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
    pygame.mixer.music.set_volume(Const.BATTLE_MUSIC_VOLUME)

    battle_state = initialize_battle(screen, ship1, ship2)
    settings = battle_state['settings']
    game_objects = battle_state['game_objects']
    border_rect = battle_state['border_rect']
    border_color = battle_state['border_color']
    player1 = battle_state['player1']
    player2 = battle_state['player2']

    running = True
    while running:
        clock.tick(Const.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop()
                    running = False

        keys = pygame.key.get_pressed()

        # Handle player controls
        player1_forward_pressed = keys[settings[f"Player {player1.player}: Forward"]]
        player2_forward_pressed = keys[settings[f"Player {player2.player}: Forward"]]
        player1_action1_pressed = keys[settings["Player 1: Action 1"]]
        player1_action2_pressed = keys[settings["Player 1: Action 2"]]
        player2_action1_pressed = keys[settings["Player 2: Action 1"]]
        player2_action2_pressed = keys[settings["Player 2: Action 2"]]

        player1.update_timers(player1_forward_pressed)
        player2.update_timers(player2_forward_pressed)

        # Movement controls
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

        # Action controls for Player 1
        if player1_action1_pressed and player1_action2_pressed:
            result, is_valid = player1.perform_action3()
            if result:
                game_objects.append(result)
            elif not is_valid:
                # Fallback to individual actions if Action 3 is invalid
                result = player1.perform_action1()
                if result:
                    game_objects.append(result)
                result = player1.perform_action2()
                if result:
                    game_objects.append(result)
        elif player1_action1_pressed:
            result = player1.perform_action1()
            if result:
                game_objects.append(result)
        elif player1_action2_pressed:
            result = player1.perform_action2()
            if result:
                game_objects.append(result)

        # Action controls for Player 2
        if player2_action1_pressed and player2_action2_pressed:
            result, is_valid = player2.perform_action3()
            if result:
                game_objects.append(result)
            elif not is_valid:
                # Fallback to individual actions if Action 3 is invalid
                result = player2.perform_action1()
                if result:
                    game_objects.append(result)
                result = player2.perform_action2()
                if result:
                    game_objects.append(result)
        elif player2_action1_pressed:
            result = player2.perform_action1()
            if result:
                game_objects.append(result)
        elif player2_action2_pressed:
            result = player2.perform_action2()
            if result:
                game_objects.append(result)

        for obj in game_objects[:]:
            if not obj.update():
                game_objects.remove(obj)

        # Drawing
        draw_battle(screen, game_objects, border_rect, border_color)
