import pygame
from src.Objects.Ships.SpaceShip import SpaceShip

def handle_player_input(settings, player1: SpaceShip, player2: SpaceShip, game_objects: list):
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