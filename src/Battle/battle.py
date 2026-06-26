import pygame
import sys
import random

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.object import ThrustMarker
from src.Battle.battle_init import initialize_battle
from src.Battle.collisions import handle_collisions
from src.Battle.battle_draw import StarFieldRenderer, draw_battle
from src.Battle.battle_entry import (
    EntryState,
    entry_complete,
    finish_entry,
    start_entry,
)
from src.Battle import battle_aftermath
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.audio import (
    compatibility_audio_service,
    use_audio_service,
)
from src.resources import use_asset_manager
import src.const as const


CONTROL_NAMES = ("Forward", "Left", "Right", "Action 1", "Action 2")
ACTION_ALIASES = {
    "Forward": ("forward",),
    "Left": ("left", "turn_left"),
    "Right": ("right", "turn_right"),
    "Action 1": ("action1", "action_1", "primary"),
    "Action 2": ("action2", "action_2", "secondary"),
}
SHIP_CONTROL_NAMES = {
    "Forward": "thrust",
    "Left": "turn_left",
    "Right": "turn_right",
    "Action 1": "action1",
    "Action 2": "action2",
}


class BattleSimulation:
    def __init__(self, screen, ship1: SpaceShip, ship2: SpaceShip,
                 player1_ships=None, player2_ships=None, sound_enabled=True,
                 audio_service=None, seed=None, rng=None, resources=None,
                 include_stars=True):
        if seed is not None and rng is not None:
            raise ValueError("Pass either seed or rng, not both")
        self.rng = rng if rng is not None else (
            random.Random(seed) if seed is not None else random
        )
        self.resources = resources or getattr(ship1, "resources", None)
        self.audio = (
            audio_service
            if audio_service is not None
            else compatibility_audio_service(sound_enabled)
        )
        self.sound_enabled = self.audio.enabled
        self._bind_runtime_to_ships(
            ship1, ship2, player1_ships, player2_ships
        )
        self.audio.start_battle_music()

        battle_state = initialize_battle(
            screen,
            ship1,
            ship2,
            rng=self.rng,
            resources=self.resources,
            include_stars=include_stars,
        )
        self.settings = battle_state['settings']
        self.world = battle_state['world']
        self.border_rect = battle_state['border_rect']
        self.border_color = battle_state['border_color']
        self.player1 = battle_state['player1']
        self.player2 = battle_state['player2']
        self.player1_ships = player1_ships
        self.player2_ships = player2_ships

        reset_ship_controls(self.player1)
        reset_ship_controls(self.player2)
        self.key_states = self._initial_key_states()
        self.frame_id = 0
        self.aftermath: battle_aftermath.AftermathState | None = None
        self.entry_animations_enabled = screen is not None
        self.entry: EntryState | None = (
            start_entry(
                (self.player1, self.player2),
                self.player1,
                self.player2,
                self.frame_id,
            )
            if self.entry_animations_enabled else None
        )
        self.needs_selection = False
        self.running = True

    def _bind_runtime_to_ships(self, *ship_groups):
        for group in ship_groups:
            ships = group if isinstance(group, (list, tuple)) else (group,)
            for ship in ships or ():
                if ship is not None:
                    ship.audio_service = self.audio
                    ship.rng = self.rng

    @property
    def game_objects(self):
        """Compatibility access to the World's authoritative object list."""
        return self.world.objects

    @game_objects.setter
    def game_objects(self, objects):
        self.world = World(objects)

    def _initial_key_states(self):
        return {
            self.settings[f"Player {player}: {control}"]: False
            for player in (1, 2)
            for control in CONTROL_NAMES
        }

    def step(self, actions=None, key_changes=None):
        if not self.running:
            return self.state()

        self.frame_id += 1
        self.needs_selection = False

        for key, pressed in key_changes or []:
            self.handle_key_change(key, pressed)

        if actions is not None:
            self.apply_actions(actions)

        with (
            use_audio_service(getattr(self, "audio", None)),
            use_asset_manager(getattr(self, "resources", None)),
        ):
            entry = getattr(self, "entry", None)
            excluded_ships = entry.entering_ships if entry else ()
            self._process_ship_inputs(excluded_ships)
            self._update_tracking_lists()
            self._update_objects(excluded_ships)
            handle_collisions(
                self.world,
                rng=getattr(self, "rng", None),
                resources=getattr(self, "resources", None),
                excluded_objects=excluded_ships,
            )
            self._update_aftermath()
            if entry is not None and entry_complete(entry, self.frame_id):
                finish_entry(entry)
                self.entry = None

        return self.state()

    def handle_key_change(self, key, pressed):
        if key not in self.key_states:
            return

        self.key_states[key] = pressed
        binding = self._binding_for_key(key)
        if binding is None:
            return

        player, control = binding
        ship = self.player1 if player == 1 else self.player2
        ship.set_control_state(control, pressed, self.frame_id)

    def apply_actions(self, actions):
        for player in (1, 2):
            player_actions = actions.get(player, actions.get(str(player)))
            if player_actions is None:
                continue

            player_actions = player_actions or {}
            for control in CONTROL_NAMES:
                key = self.settings[f"Player {player}: {control}"]
                pressed = self._action_pressed(player_actions, control)
                if self.key_states.get(key) != pressed:
                    self.handle_key_change(key, pressed)

    def _action_pressed(self, player_actions, control):
        for alias in ACTION_ALIASES[control]:
            if alias in player_actions:
                return bool(player_actions[alias])
        return False

    def _binding_for_key(self, key):
        for player in (1, 2):
            for control in CONTROL_NAMES:
                if key == self.settings[f"Player {player}: {control}"]:
                    return player, SHIP_CONTROL_NAMES[control]
        return None

    def _process_ship_inputs(self, excluded_ships=()):
        excluded_ids = {id(ship) for ship in excluded_ships}
        for ship in (self.player1, self.player2):
            if (
                id(ship) not in excluded_ids
                and ship.currently_alive
                and ship.current_hp > 0
            ):
                self.world.add_all(ship.process_controls(self.frame_id))

    def _update_tracking_lists(self):
        projectiles = self.world.abilities_of_kind('projectile', 'fighter')
        asteroids = self.world.asteroids
        ships = self.world.ships
        tracked_objects = self.world.objects_of_types(SpaceShip, Ability)
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

    def _update_objects(self, excluded_objects=()):
        self.world.update_objects(excluded_objects)

    def _update_aftermath(self):
        audio = getattr(self, "audio", None)
        newly_dead = [
            ship for ship in (self.player1, self.player2)
            if ship.current_hp <= 0 and ship.currently_alive
        ]
        reborn_ships = [
            ship for ship in newly_dead
            if self._attempt_rebirth(ship)
        ]
        if newly_dead:
            self.aftermath = battle_aftermath.start_or_update_aftermath(
                self.aftermath,
                newly_dead,
                self.player1,
                self.player2,
                self.world,
                self.frame_id,
                self.sound_enabled,
                audio,
                getattr(self, "rng", random),
                rebirth_ships=reborn_ships,
            )
            permanent_deaths = [
                ship for ship in newly_dead
                if ship not in self.aftermath.pending_rebirths
            ]
            winner = self.winner() if permanent_deaths else None
            if winner is not None:
                on_battle_won = getattr(winner, "on_battle_won", None)
                if on_battle_won is not None:
                    on_battle_won()

        if self.aftermath is None:
            return

        battle_aftermath.update_aftermath(
            self.aftermath,
            self.player1,
            self.player2,
            self.world,
            self.frame_id,
            self.sound_enabled,
            audio,
        )
        self._complete_ready_rebirths()

        if self.aftermath is None:
            return

        if (
            not self.aftermath.pending_rebirths
            and battle_aftermath.aftermath_ready_for_selection(
                self.aftermath,
                self.frame_id,
                self.sound_enabled,
            )
        ):
            self.needs_selection = True

    @staticmethod
    def _attempt_rebirth(ship):
        attempt_rebirth = getattr(ship, "attempt_rebirth", None)
        return attempt_rebirth is not None and attempt_rebirth()

    def _complete_ready_rebirths(self):
        if self.aftermath is None or not self.aftermath.pending_rebirths:
            return

        ready = self.aftermath.rebirths_ready(self.frame_id)
        if not ready:
            return

        for ship in ready:
            ship.complete_rebirth()
        self.aftermath.finish_rebirths(ready)
        self._reenter_reborn_ships(ready)
        self.world.add_all(ready)
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1

        living_ships = [
            ship for ship in (self.player1, self.player2)
            if ship.currently_alive and ship.current_hp > 0
        ]
        if len(living_ships) == 2:
            self.aftermath = None
            self.audio.start_battle_music()
        else:
            winner = self.winner()
            on_battle_won = getattr(winner, "on_battle_won", None)
            if on_battle_won is not None:
                on_battle_won()

    def _reenter_reborn_ships(self, ships):
        if len(ships) == 2:
            positions = random_ship_positions(self.rng)
        else:
            ship = ships[0]
            opponent = self.player2 if ship is self.player1 else self.player1
            positions = (random_position_away_from(opponent.position, self.rng),)

        for ship, position in zip(ships, positions):
            ship.initialize_in_battle(
                position,
                self.rng.randint(0, const.SHIP_DIRECTIONS - 1),
            )
            ship.current_hp = ship.max_hp
            ship.currently_alive = True
            reset_ship_controls(ship)

        entering_ships = list(ships)
        trail_styles = {
            ship: ship.rebirth_entry_trail_style()
            for ship in ships
        }
        if self.entry is not None:
            entering_ships = [
                *self.entry.entering_ships,
                *entering_ships,
            ]
            trail_styles = {
                **self.entry.trail_styles,
                **trail_styles,
            }
            finish_entry(self.entry)
        self.entry = start_entry(
            tuple(dict.fromkeys(entering_ships)),
            self.player1,
            self.player2,
            self.frame_id,
            trail_styles=trail_styles,
        )

    def select_next_round(self, selected):
        if not selected or not all(selected):
            self.audio.stop_music()
            self.running = False
            return ()

        previous_player1, previous_player2 = self.player1, self.player2
        self.player1, self.player2 = selected
        self._bind_runtime_to_ships(self.player1, self.player2)
        entering_ships = reset_round_objects(
            self.world,
            self.player1,
            self.player2,
            previous_player1,
            previous_player2,
            rng=self.rng,
        )
        reset_key_states(self.key_states)
        reset_ship_controls(self.player1)
        reset_ship_controls(self.player2)
        self.aftermath = None
        self.needs_selection = False
        self.entry = (
            start_entry(
                entering_ships,
                self.player1,
                self.player2,
                self.frame_id,
            )
            if self.entry_animations_enabled else None
        )
        self.audio.start_battle_music()
        return entering_ships

    def state(self):
        return {
            "frame_id": self.frame_id,
            "running": self.running,
            "needs_selection": self.needs_selection,
            "player1": self.player1,
            "player2": self.player2,
            "game_objects": self.game_objects,
            "aftermath": self.aftermath,
            "entry": getattr(self, "entry", None),
            "winner": self.winner(),
        }

    def winner(self):
        if self.aftermath is not None and self.aftermath.pending_rebirths:
            return None
        living = [
            ship for ship in (self.player1, self.player2)
            if ship.currently_alive and ship.current_hp > 0
        ]
        if len(living) == 1:
            return living[0]
        if len(living) == 0:
            return None
        return None


def run(screen, ship1: SpaceShip, ship2: SpaceShip, player1_ships=None,
        player2_ships=None, audio_service=None, menu_sound_manager=None):
    clock = pygame.time.Clock()
    simulation = BattleSimulation(
        screen, ship1, ship2, player1_ships, player2_ships,
        audio_service=audio_service,
    )
    star_field_renderer = StarFieldRenderer()

    running = True
    is_paused = False
    pygame.event.clear(pygame.KEYDOWN)
    pygame.event.clear(pygame.KEYUP)

    state = simulation.state()
    video_frame = 0
    accumulated_key_changes = []

    while running:
        clock.tick(const.VIDEO_FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    is_paused = not is_paused
                    if is_paused:
                        simulation.audio.pause()
                    else:
                        simulation.audio.unpause()
                elif event.key == pygame.K_ESCAPE:
                    simulation.audio.stop_music()
                    running = False
                elif event.key in simulation.key_states:
                    accumulated_key_changes.append((event.key, True))
            elif event.type == pygame.KEYUP:
                if event.key in simulation.key_states:
                    accumulated_key_changes.append((event.key, False))

        interp_t = (video_frame % const.VIDEO_FPS_MULTIPLIER) / const.VIDEO_FPS_MULTIPLIER
        is_physics_frame = (video_frame % const.VIDEO_FPS_MULTIPLIER == 0)

        if is_physics_frame or is_paused:
            if is_paused:
                for key, pressed in accumulated_key_changes:
                    simulation.handle_key_change(key, pressed)
                accumulated_key_changes.clear()
                state = simulation.state()
            else:
                state = simulation.step(key_changes=accumulated_key_changes)
                accumulated_key_changes.clear()

            if state["needs_selection"]:
                from src.Menus import pick_ship

                stop_tracking_projectiles(simulation.world)
                simulation.audio.stop_music()
                selected = pick_ship.run(
                    screen,
                    player1_ships,
                    player2_ships,
                    start_battle=False,
                    preselect_player1=simulation.player1 if simulation.player1.currently_alive else None,
                    preselect_player2=simulation.player2 if simulation.player2.currently_alive else None,
                    choose_second_player=simulation.aftermath.choose_second_player,
                    audio_service=simulation.audio,
                    menu_sound_manager=menu_sound_manager,
                )
                simulation.select_next_round(selected)
                pygame.event.clear(pygame.KEYDOWN)
                pygame.event.clear(pygame.KEYUP)
                if not simulation.running:
                    running = False
                    continue
                state = simulation.state()

        # Drawing
        draw_battle(
            screen,
            simulation.world,
            simulation.border_rect,
            simulation.border_color,
            star_field_renderer,
            camera_targets=(
                simulation.entry.camera_targets
                if simulation.entry
                else battle_aftermath.aftermath_camera_targets(
                    simulation.aftermath,
                    simulation.player1,
                    simulation.player2,
                    simulation.frame_id,
                )
            ),
            entry_state=simulation.entry,
            frame_id=simulation.frame_id,
            original_ships=(simulation.player1, simulation.player2),
            is_paused=is_paused,
            interp_t=0.0 if is_paused else interp_t,
        )

        video_frame += 1


def play_battle_music():
    compatibility_audio_service(True).start_battle_music()


def set_battle_sound_enabled(enabled):
    """Compatibility helper returning an isolated service instead of mutating classes."""
    return compatibility_audio_service(enabled)


def reset_round_objects(
    game_objects,
    player1,
    player2,
    previous_player1,
    previous_player2,
    *,
    rng=None,
):
    rng = rng or random
    world = World.coerce(game_objects)
    selected_ships = [player1, player2]
    preserved_ships = {
        ship for ship in (previous_player1, previous_player2)
        if ship in selected_ships and ship.is_alive()
    }

    persistent_objects = world.objects_excluding_types(
        SpaceShip, Ability, ThrustMarker, BattleEffect
    )
    preserved_abilities = [
        obj for obj in world.abilities
        if (
            obj.parent in preserved_ships and
            obj.is_alive()
        )
    ]
    world.retain(persistent_objects + preserved_abilities)

    planets = world.planets
    planet = planets[0] if planets else None

    entering_ships = initialize_new_round_ships(
        selected_ships, preserved_ships, planet, rng=rng
    )

    player1.opponent = player2
    player2.opponent = player1
    update_preserved_abilities(preserved_abilities, player1, player2, planet)

    world.add_all(selected_ships)
    return entering_ships


def stop_tracking_projectiles(game_objects):
    for obj in World.coerce(game_objects).abilities:
        if obj.is_alive():
            obj.stop_and_track()


def initialize_new_round_ships(
    selected_ships, preserved_ships, planet, *, rng=None
):
    rng = rng or random
    new_ships = [ship for ship in selected_ships if ship not in preserved_ships]
    preserved_list = list(preserved_ships)

    if len(new_ships) == 2:
        positions = list(random_ship_positions(rng))
    elif len(new_ships) == 1 and preserved_list:
        positions = [random_position_away_from(
            preserved_list[0].position, rng
        )]
    else:
        positions = []

    for ship, position in zip(new_ships, positions):
        ship.initialize_in_battle(
            position, rng.randint(0, const.SHIP_DIRECTIONS - 1)
        )
        ship.currently_alive = True
        reset_ship_controls(ship)

    for ship in selected_ships:
        if planet:
            ship.set_planet(planet)
        reset_ship_controls(ship)

    return new_ships


def update_preserved_abilities(abilities, player1, player2, planet):
    for ability in abilities:
        opponent = player2 if ability.player == player1.player else player1
        ability.opponent = opponent
        ability.stop_and_track()
        if (
            ability.target is None or
            not World.is_alive(ability.target)
        ):
            ability.target = opponent
        if planet:
            ability.planet = planet


def random_position_away_from(position, rng=None):
    from src.Battle.battle_init import get_random_position, validate_ship_positions
    rng = rng or random

    for _ in range(1000):
        candidate = get_random_position(rng)
        if validate_ship_positions(position, candidate):
            return candidate

    return get_random_position(rng)


def random_ship_positions(rng=None):
    from src.Battle.battle_init import get_valid_ship_positions
    return get_valid_ship_positions(rng or random)


def reset_key_states(key_states):
    for key in key_states:
        key_states[key] = False


def reset_ship_controls(ship):
    ship.reset_controls()
