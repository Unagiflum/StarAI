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

        # Convert key presses to actions for player 1
        p1_objects = player1.handle_actions(
            thrust=keys[settings["Player 1: Forward"]],
            turn_left=keys[settings["Player 1: Left"]],
            turn_right=keys[settings["Player 1: Right"]],
            action1=keys[settings["Player 1: Action 1"]],
            action2=keys[settings["Player 1: Action 2"]]
        )

        # Convert key presses to actions for player 2
        p2_objects = player2.handle_actions(
            thrust=keys[settings["Player 2: Forward"]],
            turn_left=keys[settings["Player 2: Left"]],
            turn_right=keys[settings["Player 2: Right"]],
            action1=keys[settings["Player 2: Action 1"]],
            action2=keys[settings["Player 2: Action 2"]]
        )

        game_objects.extend(p1_objects)
        game_objects.extend(p2_objects)

        for obj in game_objects[:]:
            if not obj.update():
                game_objects.remove(obj)

        # Drawing
        draw_battle(screen, game_objects, border_rect, border_color)
