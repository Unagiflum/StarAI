import pygame
import sys
import random
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.object import ThrustMarker
from src.Battle.battle_init import initialize_battle
from src.Battle.collisions import handle_collisions
from src.Battle.battle_draw import StarFieldRenderer, draw_battle
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.resources import default_assets
from src.audio import (
    PygameAudioService,
    compatibility_audio_service,
    use_audio_service,
)
import src.const as const


EXPLOSION_PLACEMENT_INTERVAL_FRAMES = 3

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


@dataclass
class ScheduledExplosion(Mapping):
    frame: int
    ship: SpaceShip
    position: list[float]
    scale: float
    is_final: bool

    _COMPATIBILITY_KEYS = ("frame", "ship", "position", "scale", "is_final")

    def __getitem__(self, key):
        if key not in self._COMPATIBILITY_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._COMPATIBILITY_KEYS)

    def __len__(self):
        return len(self._COMPATIBILITY_KEYS)

    @classmethod
    def from_mapping(cls, item):
        return cls(
            frame=item["frame"],
            ship=item["ship"],
            position=item["position"],
            scale=item["scale"],
            is_final=item["is_final"],
        )


@dataclass
class AftermathState(Mapping):
    started_frame: int
    latest_death_frame: int
    dead_players: set[int] = field(default_factory=set)
    death_effects: dict[int, list[BattleEffect]] = field(default_factory=dict)
    pending_explosions: list[ScheduledExplosion] = field(default_factory=list)
    ships_pending_hide: set[SpaceShip] = field(default_factory=set)
    camera_hold_targets: list[SpaceShip] = field(default_factory=list)
    ditty_started: bool = False
    tie_break_ship: SpaceShip | None = None
    choose_second_player: int | None = None

    _COMPATIBILITY_KEYS = (
        "started_frame",
        "latest_death_frame",
        "dead_players",
        "death_effects",
        "pending_explosions",
        "ships_pending_hide",
        "camera_hold_targets",
        "ditty_started",
        "tie_break_ship",
        "choose_second_player",
    )

    def __getitem__(self, key):
        if key not in self._COMPATIBILITY_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._COMPATIBILITY_KEYS)

    def __len__(self):
        return len(self._COMPATIBILITY_KEYS)

    @classmethod
    def from_mapping(cls, aftermath):
        started_frame = aftermath["started_frame"]
        return cls(
            started_frame=started_frame,
            latest_death_frame=aftermath.get("latest_death_frame", started_frame),
            dead_players=aftermath.get("dead_players", set()),
            death_effects=aftermath.get("death_effects", {}),
            pending_explosions=[
                item if isinstance(item, ScheduledExplosion) else ScheduledExplosion.from_mapping(item)
                for item in aftermath.get("pending_explosions", [])
            ],
            ships_pending_hide=aftermath.get("ships_pending_hide", set()),
            camera_hold_targets=aftermath.get("camera_hold_targets", []),
            ditty_started=aftermath.get("ditty_started", False),
            tie_break_ship=aftermath.get("tie_break_ship"),
            choose_second_player=aftermath.get("choose_second_player"),
        )


class BattleSimulation:
    def __init__(self, screen, ship1: SpaceShip, ship2: SpaceShip,
                 player1_ships=None, player2_ships=None, sound_enabled=True,
                 audio_service=None):
        self.audio = (
            audio_service
            if audio_service is not None
            else compatibility_audio_service(sound_enabled)
        )
        self.sound_enabled = self.audio.enabled
        self._bind_audio_to_ships(ship1, ship2, player1_ships, player2_ships)
        self.audio.start_battle_music()

        battle_state = initialize_battle(screen, ship1, ship2)
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
        self.aftermath: AftermathState | None = None
        self.needs_selection = False
        self.running = True

    def _bind_audio_to_ships(self, *ship_groups):
        for group in ship_groups:
            ships = group if isinstance(group, (list, tuple)) else (group,)
            for ship in ships or ():
                if ship is not None:
                    ship.audio_service = self.audio

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

        with use_audio_service(getattr(self, "audio", None)):
            self._process_ship_inputs()
            self._update_tracking_lists()
            self._update_objects()
            handle_collisions(self.world)
            self._update_aftermath()

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

    def _process_ship_inputs(self):
        for ship in (self.player1, self.player2):
            if ship.currently_alive and ship.current_hp > 0:
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

    def _update_objects(self):
        self.world.update_objects()

    def _update_aftermath(self):
        audio = getattr(self, "audio", None)
        newly_dead = [
            ship for ship in (self.player1, self.player2)
            if ship.current_hp <= 0 and ship.currently_alive
        ]
        if newly_dead:
            self.aftermath = start_or_update_aftermath(
                self.aftermath,
                newly_dead,
                self.player1,
                self.player2,
                self.world,
                self.frame_id,
                self.sound_enabled,
                audio,
            )

        if self.aftermath is None:
            return

        update_aftermath(
            self.aftermath,
            self.player1,
            self.player2,
            self.world,
            self.frame_id,
            self.sound_enabled,
            audio,
        )

        if aftermath_ready_for_selection(self.aftermath, self.frame_id, self.sound_enabled):
            self.needs_selection = True

    def select_next_round(self, selected):
        if not selected or not all(selected):
            self.audio.stop_music()
            self.running = False
            return

        previous_player1, previous_player2 = self.player1, self.player2
        self.player1, self.player2 = selected
        self._bind_audio_to_ships(self.player1, self.player2)
        reset_round_objects(self.world, self.player1, self.player2, previous_player1, previous_player2)
        reset_key_states(self.key_states)
        reset_ship_controls(self.player1)
        reset_ship_controls(self.player2)
        self.aftermath = None
        self.needs_selection = False
        self.audio.start_battle_music()

    def state(self):
        return {
            "frame_id": self.frame_id,
            "running": self.running,
            "needs_selection": self.needs_selection,
            "player1": self.player1,
            "player2": self.player2,
            "game_objects": self.game_objects,
            "aftermath": self.aftermath,
            "winner": self.winner(),
        }

    def winner(self):
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
    pygame.event.clear(pygame.KEYDOWN)
    pygame.event.clear(pygame.KEYUP)

    while running:
        clock.tick(const.FPS)
        key_changes = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    simulation.audio.stop_music()
                    running = False
                elif event.key in simulation.key_states:
                    key_changes.append((event.key, True))
            elif event.type == pygame.KEYUP:
                if event.key in simulation.key_states:
                    key_changes.append((event.key, False))

        state = simulation.step(key_changes=key_changes)

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

        # Drawing
        draw_battle(
            screen,
            simulation.world,
            simulation.border_rect,
            simulation.border_color,
            star_field_renderer,
            camera_targets=aftermath_camera_targets(
                simulation.aftermath,
                simulation.player1,
                simulation.player2,
                simulation.frame_id,
            ),
        )


def play_battle_music():
    compatibility_audio_service(True).start_battle_music()


def set_battle_sound_enabled(enabled):
    """Compatibility helper returning an isolated service instead of mutating classes."""
    return compatibility_audio_service(enabled)


def start_or_update_aftermath(aftermath, dead_ships, player1, player2,
                              game_objects, frame_id, sound_enabled=True,
                              audio_service=None):
    audio = (
        audio_service
        if audio_service is not None
        else compatibility_audio_service(sound_enabled)
    )
    if aftermath is None:
        state = AftermathState(started_frame=frame_id, latest_death_frame=frame_id)
    else:
        state = _coerce_aftermath_state(aftermath)

    for ship in dead_ships:
        ship.current_hp = 0
        ship.currently_alive = False
        reset_ship_controls(ship)
        state.dead_players.add(ship.player)
        with use_audio_service(audio):
            BattleEffect.play_ship_death()
        state.death_effects[ship.player] = []
        state.pending_explosions.extend(create_ship_explosion_schedule(ship, frame_id))
        state.ships_pending_hide.add(ship)
        state.camera_hold_targets.append(ship)
        state.latest_death_frame = frame_id

    if player1.current_hp <= 0 and player2.current_hp <= 0:
        self_destructors = [
            ship for ship in (player1, player2)
            if getattr(ship, "shofixti_self_destruct", False)
        ]
        if len(self_destructors) == 1:
            state.tie_break_ship = self_destructors[0]
            state.choose_second_player = self_destructors[0].player

    audio.stop_music()
    state.ditty_started = False
    _sync_legacy_aftermath(aftermath, state)

    return state


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
        schedule.append(ScheduledExplosion(
            frame=start_frame + index * EXPLOSION_PLACEMENT_INTERVAL_FRAMES,
            ship=ship,
            position=position,
            scale=random.uniform(0.85, 1.15),
            is_final=index == count - 1,
        ))

    return schedule


def update_aftermath(aftermath, player1, player2, game_objects, frame_id,
                     sound_enabled=True, audio_service=None):
    audio = (
        audio_service
        if audio_service is not None
        else compatibility_audio_service(sound_enabled)
    )
    state = _coerce_aftermath_state(aftermath)
    world = World.coerce(game_objects)
    ready_explosions = [
        item for item in state.pending_explosions
        if item.frame <= frame_id
    ]
    state.pending_explosions = [
        item for item in state.pending_explosions
        if item.frame > frame_id
    ]

    for item in ready_explosions:
        effect = BattleEffect.ship_explosion(item.position, scale=item.scale)
        state.death_effects[item.ship.player].append(effect)
        world.add(effect)
        if item.is_final:
            with use_audio_service(audio):
                BattleEffect.play_ship_death()
            hide_dead_ship(item.ship, world)
            state.ships_pending_hide.discard(item.ship)

    living_ships = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    death_view_done = frame_id - state.started_frame >= const.POST_DEATH_ANIMATION_VIEW_FRAMES
    if len(living_ships) == 1 and not state.ditty_started and death_view_done:
        if audio_service is None and sound_enabled:
            play_victory_ditty(living_ships[0])
        else:
            audio.play_victory_ditty(living_ships[0])
        state.ditty_started = True
    elif (
        not living_ships and
        state.tie_break_ship is not None and
        not state.ditty_started and
        death_view_done
    ):
        if audio_service is None and sound_enabled:
            play_victory_ditty(state.tie_break_ship)
        else:
            audio.play_victory_ditty(state.tie_break_ship)
        state.ditty_started = True

    _sync_legacy_aftermath(aftermath, state)


def hide_dead_ship(ship, game_objects):
    World.coerce(game_objects).remove_where(lambda obj: obj is ship)


def aftermath_camera_targets(aftermath, player1, player2, frame_id=None):
    if aftermath is None:
        return None
    state = _coerce_aftermath_state(aftermath)
    if frame_id is not None and frame_id - state.started_frame >= const.POST_DEATH_ANIMATION_VIEW_FRAMES:
        return None

    targets = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    targets.extend(state.camera_hold_targets)
    return targets or None


def play_victory_ditty(ship):
    PygameAudioService(
        getattr(ship, "resources", default_assets())
    ).play_victory_ditty(ship)


def aftermath_ready_for_selection(aftermath, frame_id, sound_enabled=True):
    state = _coerce_aftermath_state(aftermath)
    elapsed = frame_id - state.started_frame
    return elapsed >= const.POST_DEATH_CONTROL_FRAMES


def _coerce_aftermath_state(aftermath):
    if isinstance(aftermath, AftermathState):
        return aftermath
    return AftermathState.from_mapping(aftermath)


def _sync_legacy_aftermath(aftermath, state):
    if not isinstance(aftermath, MutableMapping):
        return
    for key in state:
        value = state[key]
        if key == "pending_explosions":
            value = [dict(item) for item in value]
        aftermath[key] = value


def reset_round_objects(game_objects, player1, player2, previous_player1, previous_player2):
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

    initialize_new_round_ships(selected_ships, preserved_ships, planet)

    player1.opponent = player2
    player2.opponent = player1
    update_preserved_abilities(preserved_abilities, player1, player2, planet)

    world.add_all(selected_ships)


def stop_tracking_projectiles(game_objects):
    for obj in World.coerce(game_objects).abilities:
        if obj.is_alive():
            obj.stop_and_track()


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
        ability.stop_and_track()
        if (
            ability.target is None or
            not World.is_alive(ability.target)
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
    ship.reset_controls()
