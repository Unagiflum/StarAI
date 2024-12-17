import pygame
from src.Objects.Ships.SpaceShip import SpaceShip

def handle_player_input(settings, player1: SpaceShip, player2: SpaceShip, game_objects: list):
    keys = pygame.key.get_pressed()

    # Get key states
    player1_forward = keys[settings["Player 1: Forward"]]
    player2_forward = keys[settings["Player 2: Forward"]]
    player1_action1 = keys[settings["Player 1: Action 1"]]
    player1_action2 = keys[settings["Player 1: Action 2"]]
    player2_action1 = keys[settings["Player 2: Action 1"]]
    player2_action2 = keys[settings["Player 2: Action 2"]]

    # Update timers
    player1.update_timers(player1_forward)
    player2.update_timers(player2_forward)

    # Handle movement
    new_objects = []

    if keys[settings["Player 1: Left"]]:
        player1.turn_left()
    if keys[settings["Player 1: Right"]]:
        player1.turn_right()
    if player1_forward:
        marker = player1.apply_thrust()
        if marker:
            new_objects.append(marker)

    if keys[settings["Player 2: Left"]]:
        player2.turn_left()
    if keys[settings["Player 2: Right"]]:
        player2.turn_right()
    if player2_forward:
        marker = player2.apply_thrust()
        if marker:
            new_objects.append(marker)

    # Handle actions for Player 1
    if player1_action1 and player1_action2:
        result, is_valid = player1.perform_action3()
        if result:
            new_objects.append(result)
        elif not is_valid:
            result = player1.perform_action1()
            if result:
                new_objects.append(result)
            result = player1.perform_action2()
            if result:
                new_objects.append(result)
    elif player1_action1:
        result = player1.perform_action1()
        if result:
            new_objects.append(result)
    elif player1_action2:
        result = player1.perform_action2()
        if result:
            new_objects.append(result)

    # Handle actions for Player 2
    if player2_action1 and player2_action2:
        result, is_valid = player2.perform_action3()
        if result:
            new_objects.append(result)
        elif not is_valid:
            result = player2.perform_action1()
            if result:
                new_objects.append(result)
            result = player2.perform_action2()
            if result:
                new_objects.append(result)
    elif player2_action1:
        result = player2.perform_action1()
        if result:
            new_objects.append(result)
    elif player2_action2:
        result = player2.perform_action2()
        if result:
            new_objects.append(result)

    game_objects.extend(new_objects)