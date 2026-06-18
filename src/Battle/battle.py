import pygame
import sys
import random
import math

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.object import ThrustMarker
from src.Battle.battle_init import initialize_battle
from src.Battle.collisions import handle_collisions
from src.Battle.battle_draw import draw_battle
from src.Battle.effects import BattleEffect
import src.const as const


AFTERMATH_SELECT_DELAY_FRAMES = const.FPS * 2
EXPLOSION_PLACEMENT_INTERVAL_FRAMES = 3
VICTORY_PAUSE_FRAMES = max(1, const.FPS // 2)


def run(screen, ship1: SpaceShip, ship2: SpaceShip, player1_ships=None, player2_ships=None):
    clock = pygame.time.Clock()

    play_battle_music()

    battle_state = initialize_battle(screen, ship1, ship2)
    settings = battle_state['settings']
    game_objects = battle_state['game_objects']
    border_rect = battle_state['border_rect']
    border_color = battle_state['border_color']
    player1 = battle_state['player1']
    player2 = battle_state['player2']
    reset_ship_controls(player1)
    reset_ship_controls(player2)

    running = True
    frame_id = 0
    aftermath = None
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
    pygame.event.clear(pygame.KEYDOWN)
    pygame.event.clear(pygame.KEYUP)

    while running:
        clock.tick(const.FPS)
        frame_id += 1

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
                        game_objects.extend(handle_ship_input(
                            player1,
                            event.key,
                            True,
                            settings["Player 1: Forward"],
                            settings["Player 1: Left"],
                            settings["Player 1: Right"],
                            settings["Player 1: Action 1"],
                            settings["Player 1: Action 2"],
                            frame_id
                        ))
                    else:
                        game_objects.extend(handle_ship_input(
                            player2,
                            event.key,
                            True,
                            settings["Player 2: Forward"],
                            settings["Player 2: Left"],
                            settings["Player 2: Right"],
                            settings["Player 2: Action 1"],
                            settings["Player 2: Action 2"],
                            frame_id
                        ))

            elif event.type == pygame.KEYUP:
                if event.key in key_states:
                    key_states[event.key] = False
                    if event.key in [settings["Player 1: Forward"], settings["Player 1: Left"],
                                     settings["Player 1: Right"], settings["Player 1: Action 1"],
                                     settings["Player 1: Action 2"]]:
                        game_objects.extend(handle_ship_input(
                            player1,
                            event.key,
                            False,
                            settings["Player 1: Forward"],
                            settings["Player 1: Left"],
                            settings["Player 1: Right"],
                            settings["Player 1: Action 1"],
                            settings["Player 1: Action 2"],
                            frame_id
                        ))
                    else:
                        game_objects.extend(handle_ship_input(
                            player2,
                            event.key,
                            False,
                            settings["Player 2: Forward"],
                            settings["Player 2: Left"],
                            settings["Player 2: Right"],
                            settings["Player 2: Action 1"],
                            settings["Player 2: Action 2"],
                            frame_id
                        ))

        # Call handle_actions with no key for repeating actions
        game_objects.extend(handle_ship_input(
            player1,
            None,
            False,
            settings["Player 1: Forward"],
            settings["Player 1: Left"],
            settings["Player 1: Right"],
            settings["Player 1: Action 1"],
            settings["Player 1: Action 2"],
            frame_id
        ))

        game_objects.extend(handle_ship_input(
            player2,
            None,
            False,
            settings["Player 2: Forward"],
            settings["Player 2: Left"],
            settings["Player 2: Right"],
            settings["Player 2: Action 1"],
            settings["Player 2: Action 2"],
            frame_id
        ))

        # Update tracking arrays for all ships and abilities
        projectiles = [
            obj for obj in game_objects
            if isinstance(obj, Ability) and obj.type == 'projectile'
        ]
        asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid)]
        ships = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
        tracked_objects = [
            obj for obj in game_objects
            if isinstance(obj, (SpaceShip, Ability))
        ]
        players = {obj.player for obj in tracked_objects}
        projectiles_by_player = {
            player: [obj for obj in projectiles if obj.player == player]
            for player in players
        }
        enemy_projectiles_by_player = {
            player: [obj for obj in projectiles if obj.player != player]
            for player in players
        }
        for obj in tracked_objects:
            obj.friendly_objects = projectiles_by_player.get(obj.player, [])
            obj.enemy_objects = enemy_projectiles_by_player.get(obj.player, [])
            obj.asteroids = asteroids
        for asteroid in asteroids:
            asteroid.ships = ships
            asteroid.asteroids = asteroids

        # Update all objects
        for obj in game_objects[:]:
            if isinstance(obj, SpaceShip) and obj.current_hp <= 0:
                continue
            if not obj.update():
                game_objects.remove(obj)

        handle_collisions(game_objects)

        newly_dead = [
            ship for ship in (player1, player2)
            if ship.current_hp <= 0 and ship.currently_alive
        ]
        if newly_dead:
            aftermath = start_or_update_aftermath(aftermath, newly_dead, player1, player2, game_objects, frame_id)

        if aftermath is not None:
            newly_dead = [
                ship for ship in (player1, player2)
                if ship.current_hp <= 0 and ship.currently_alive
            ]
            if newly_dead:
                aftermath = start_or_update_aftermath(
                    aftermath,
                    newly_dead,
                    player1,
                    player2,
                    game_objects,
                    frame_id,
                )

            update_aftermath(aftermath, player1, player2, game_objects, frame_id)

            if aftermath_ready_for_selection(aftermath, frame_id):
                from src.Menus import pick_ship

                pygame.mixer.music.stop()
                selected = pick_ship.run(
                    screen,
                    player1_ships,
                    player2_ships,
                    start_battle=False,
                    preselect_player1=player1 if player1.currently_alive else None,
                    preselect_player2=player2 if player2.currently_alive else None,
                )
                if not selected or not all(selected):
                    pygame.mixer.music.stop()
                    running = False
                    continue

                previous_player1, previous_player2 = player1, player2
                player1, player2 = selected
                reset_round_objects(game_objects, player1, player2, previous_player1, previous_player2)
                reset_key_states(key_states)
                pygame.event.clear(pygame.KEYDOWN)
                pygame.event.clear(pygame.KEYUP)
                aftermath = None
                play_battle_music()

        # Drawing
        draw_battle(
            screen,
            game_objects,
            border_rect,
            border_color,
            camera_targets=aftermath_camera_targets(aftermath, player1, player2),
        )


def play_battle_music():
    pygame.mixer.music.load(const.BATTLE_MUSIC_PATH)
    pygame.mixer.music.play(-1)
    pygame.mixer.music.set_volume(const.BATTLE_MUSIC_VOLUME)


def handle_ship_input(ship, key, pressed, forward_key, left_key, right_key, action1_key, action2_key, frame_id):
    if not ship.currently_alive or ship.current_hp <= 0:
        return []

    return ship.handle_actions(
        key,
        pressed,
        forward_key,
        left_key,
        right_key,
        action1_key,
        action2_key,
        frame_id,
    )


def start_or_update_aftermath(aftermath, dead_ships, player1, player2, game_objects, frame_id):
    if aftermath is None:
        aftermath = {
            "started_frame": frame_id,
            "latest_death_frame": frame_id,
            "dead_players": set(),
            "death_effects": {},
            "death_sound_done_frame": frame_id,
            "pending_explosions": [],
            "ships_pending_hide": set(),
            "camera_hold_targets": [],
            "victory_pause_started_frame": None,
            "ditty_started": False,
        }

    for ship in dead_ships:
        ship.current_hp = 0
        ship.currently_alive = False
        reset_ship_controls(ship)
        aftermath["dead_players"].add(ship.player)
        sound_frames = BattleEffect.play_ship_death()
        aftermath["death_sound_done_frame"] = max(
            aftermath["death_sound_done_frame"],
            frame_id + sound_frames,
        )
        aftermath["death_effects"][ship.player] = []
        aftermath["pending_explosions"].extend(create_ship_explosion_schedule(ship, frame_id))
        aftermath["ships_pending_hide"].add(ship)
        aftermath["camera_hold_targets"].append(ship)
        aftermath["latest_death_frame"] = frame_id

    pygame.mixer.music.stop()
    aftermath["ditty_started"] = False
    aftermath["victory_pause_started_frame"] = None

    return aftermath


def create_ship_explosion_schedule(ship, start_frame):
    count = max(4, min(9, int(max(ship.size) / 35) + 3))
    schedule = []
    angle = math.radians(ship.rotation)
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)

    for index in range(count):
        local_x = random.uniform(-ship.size[0] * 0.45, ship.size[0] * 0.45)
        local_y = random.uniform(-ship.size[1] * 0.45, ship.size[1] * 0.45)
        position = [
            (ship.position[0] + local_x * cos_a - local_y * sin_a) % const.ARENA_SIZE,
            (ship.position[1] + local_x * sin_a + local_y * cos_a) % const.ARENA_SIZE,
        ]
        schedule.append({
            "frame": start_frame + index * EXPLOSION_PLACEMENT_INTERVAL_FRAMES,
            "ship": ship,
            "position": position,
            "scale": random.uniform(0.85, 1.15),
            "is_final": index == count - 1,
        })

    return schedule


def update_aftermath(aftermath, player1, player2, game_objects, frame_id):
    ready_explosions = [
        item for item in aftermath["pending_explosions"]
        if item["frame"] <= frame_id
    ]
    aftermath["pending_explosions"] = [
        item for item in aftermath["pending_explosions"]
        if item["frame"] > frame_id
    ]

    for item in ready_explosions:
        effect = BattleEffect.ship_explosion(item["position"], scale=item["scale"])
        aftermath["death_effects"][item["ship"].player].append(effect)
        game_objects.append(effect)
        if item["is_final"]:
            hide_dead_ship(item["ship"], game_objects)
            aftermath["ships_pending_hide"].discard(item["ship"])

    active_effects = set(game_objects)
    all_explosions_done = (
        not aftermath["pending_explosions"] and
        not any(
            effect in active_effects
            for effects in aftermath["death_effects"].values()
            for effect in effects
        )
    )
    sound_done = frame_id >= aftermath["death_sound_done_frame"]

    if all_explosions_done and sound_done and aftermath["victory_pause_started_frame"] is None:
        aftermath["victory_pause_started_frame"] = frame_id

    living_ships = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    if (
        len(living_ships) == 1 and
        not aftermath["ditty_started"] and
        aftermath["victory_pause_started_frame"] is not None and
        frame_id - aftermath["victory_pause_started_frame"] >= VICTORY_PAUSE_FRAMES
    ):
        play_victory_ditty(living_ships[0])
        aftermath["ditty_started"] = True


def hide_dead_ship(ship, game_objects):
    game_objects[:] = [
        obj for obj in game_objects
        if obj is not ship
    ]


def aftermath_camera_targets(aftermath, player1, player2):
    if aftermath is None or aftermath["ditty_started"]:
        return None

    targets = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    targets.extend(aftermath["camera_hold_targets"])
    return targets or None


def play_victory_ditty(ship):
    ditty_path = ship.sprite_location / f"{ship.name}-ditty.mp3"
    try:
        pygame.mixer.music.load(ditty_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(const.BATTLE_MUSIC_VOLUME)
    except pygame.error:
        pass


def aftermath_ready_for_selection(aftermath, frame_id):
    elapsed = frame_id - aftermath["started_frame"]
    if len(aftermath["dead_players"]) == 2:
        if aftermath["victory_pause_started_frame"] is None:
            return False
        return (
            frame_id - aftermath["latest_death_frame"] >= AFTERMATH_SELECT_DELAY_FRAMES and
            frame_id - aftermath["victory_pause_started_frame"] >= VICTORY_PAUSE_FRAMES
        )
    return aftermath["ditty_started"] and elapsed >= const.FPS and not pygame.mixer.music.get_busy()


def reset_round_objects(game_objects, player1, player2, previous_player1, previous_player2):
    selected_ships = [player1, player2]
    preserved_ships = {
        ship for ship in (previous_player1, previous_player2)
        if ship in selected_ships and ship.currently_alive and ship.current_hp > 0
    }

    persistent_objects = [
        obj for obj in game_objects
        if not isinstance(obj, (SpaceShip, Ability, ThrustMarker, BattleEffect))
    ]
    preserved_abilities = [
        obj for obj in game_objects
        if (
            isinstance(obj, Ability) and
            obj.parent in preserved_ships and
            obj.currently_alive and
            obj.current_hp > 0
        )
    ]
    game_objects[:] = persistent_objects + preserved_abilities

    planets = [obj for obj in game_objects if isinstance(obj, Planet)]
    planet = planets[0] if planets else None

    initialize_new_round_ships(selected_ships, preserved_ships, planet)

    player1.opponent = player2
    player2.opponent = player1
    update_preserved_abilities(preserved_abilities, player1, player2, planet)

    game_objects.extend(selected_ships)


def initialize_new_round_ships(selected_ships, preserved_ships, planet):
    new_ships = [ship for ship in selected_ships if ship not in preserved_ships]
    preserved_list = list(preserved_ships)

    if len(new_ships) == 2:
        positions = list(random_ship_positions())
    elif len(new_ships) == 1 and preserved_list:
        positions = [random_position_away_from(preserved_list[0].position)]
    else:
        positions = []

    for ship, position in zip(new_ships, positions):
        ship.initialize_in_battle(position, random.randint(0, const.SHIP_DIRECTIONS - 1))
        ship.currently_alive = True
        reset_ship_controls(ship)

    for ship in selected_ships:
        if planet:
            ship.set_planet(planet)
        reset_ship_controls(ship)


def update_preserved_abilities(abilities, player1, player2, planet):
    for ability in abilities:
        opponent = player2 if ability.player == player1.player else player1
        ability.opponent = opponent
        if hasattr(ability, "target") and (
            ability.target is None or
            not getattr(ability.target, "currently_alive", True) or
            getattr(ability.target, "current_hp", 1) <= 0
        ):
            ability.target = opponent
        if planet:
            ability.planet = planet


def random_position_away_from(position):
    from src.Battle.battle_init import get_random_position, validate_ship_positions

    for _ in range(1000):
        candidate = get_random_position()
        if validate_ship_positions(position, candidate):
            return candidate

    return get_random_position()


def random_ship_positions():
    from src.Battle.battle_init import get_valid_ship_positions
    return get_valid_ship_positions()


def reset_key_states(key_states):
    for key in key_states:
        key_states[key] = False


def reset_ship_controls(ship):
    ship.thrust_active = False
    ship.turn_left_active = False
    ship.turn_right_active = False
    ship.action1_active = False
    ship.action2_active = False
    ship.input_pressed_frames.clear()
