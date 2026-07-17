import pygame
import sys
import math
import random
import time
from collections.abc import Mapping

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.ability import Ability
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.object import ThrustMarker
from src.Battle.battle_init import (
    apply_training_starting_velocities,
    apply_vux_starting_conditions,
    get_training_spawn_position,
    initialize_battle,
)
from src.Battle.collisions import CollisionMetrics, handle_collisions
from src.Battle.battle_draw import DisplayStarField, draw_battle
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
from src.frame_timing import PresentationClock
from src.training.event_ledger import damage_source_owner
from src.UI.match_dialog import confirm_end_match
import src.const as const


# Lazily populated compatibility hooks. Keeping these names at module scope
# preserves existing patch/injection boundaries without importing model code in
# headless simulation workers.
BattleAIManager = None
InferenceModelCache = None
TrainingModelRepository = None

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


def _add_battle_timing_seconds(timing_seconds, bucket, started_at):
    if timing_seconds is None:
        return
    timing_seconds[bucket] = timing_seconds.get(bucket, 0.0) + max(
        0.0,
        time.perf_counter() - float(started_at),
    )


def _battle_timing_started_at(timing_seconds):
    return time.perf_counter() if timing_seconds is not None else 0.0


def _add_battle_timing_count(timing_seconds, bucket, count):
    if timing_seconds is None:
        return
    timing_seconds[bucket] = timing_seconds.get(bucket, 0.0) + max(
        0.0,
        float(count),
    )


def _add_collision_timing_counts(timing_seconds, metrics):
    if timing_seconds is None or metrics is None:
        return
    _add_battle_timing_count(
        timing_seconds,
        "collision_possible_physical_pairs",
        metrics.possible_physical_pairs,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_candidate_pairs",
        metrics.physical_candidate_pairs,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_dispatched_pairs",
        metrics.physical_dispatched_pairs,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_possible_laser_targets",
        metrics.possible_laser_targets,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_laser_candidates",
        metrics.laser_candidates,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_possible_area_targets",
        metrics.possible_area_targets,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_area_candidates",
        metrics.area_candidates,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_area_full_scan_fallbacks",
        metrics.area_full_scan_fallbacks,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_spatial_queries",
        metrics.spatial.queries,
    )
    _add_battle_timing_count(
        timing_seconds,
        "collision_spatial_returned_candidates",
        metrics.spatial.returned_candidates,
    )


class FixedStepScheduler:
    """Accumulate wall time for a fixed-rate simulation."""

    def __init__(self, fps, max_catch_up_steps=5, start_ready=True):
        self.step_seconds = 1.0 / fps
        self.max_catch_up_steps = max_catch_up_steps
        self.accumulator = self.step_seconds if start_ready else 0.0

    def advance(self, elapsed_seconds):
        max_accumulator = self.step_seconds * self.max_catch_up_steps
        self.accumulator = min(
            self.accumulator + max(0.0, elapsed_seconds),
            max_accumulator,
        )
        steps = min(
            int((self.accumulator + 1e-12) / self.step_seconds),
            self.max_catch_up_steps,
        )
        self.accumulator -= steps * self.step_seconds
        interpolation = min(1.0, max(0.0, self.accumulator / self.step_seconds))
        return steps, interpolation

    def reset(self, *, start_ready=False):
        self.accumulator = self.step_seconds if start_ready else 0.0


class BattleSimulation:
    def __init__(
        self,
        screen,
        ship1: SpaceShip,
        ship2: SpaceShip,
        player1_ships=None,
        player2_ships=None,
        sound_enabled=True,
        audio_service=None,
        seed=None,
        rng=None,
        resources=None,
        include_stars=True,
        training_event_ledger=None,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Pass either seed or rng, not both")
        self.rng = (
            rng
            if rng is not None
            else (random.Random(seed) if seed is not None else random)
        )
        self.resources = resources or getattr(ship1, "resources", None)
        # Preserve ordinary battle semantics by default. Coordinated training
        # explicitly disables visual-only objects when no display is requested.
        self.visual_effects_enabled = True
        self.audio = (
            audio_service
            if audio_service is not None
            else compatibility_audio_service(sound_enabled)
        )
        self.sound_enabled = self.audio.enabled
        self._bind_runtime_to_ships(ship1, ship2, player1_ships, player2_ships)
        self.audio.start_battle_music()

        battle_state = initialize_battle(
            screen,
            ship1,
            ship2,
            rng=self.rng,
            resources=self.resources,
            include_stars=include_stars,
            training_vux_close_start_chance=(
                const.TRAINING_VUX_CLOSE_START_CHANCE
                if training_event_ledger is not None
                else None
            ),
        )
        self.settings = battle_state["settings"]
        self.world = battle_state["world"]
        self.training_event_ledger = training_event_ledger
        self.training_mode = training_event_ledger is not None
        self.world.set_training_event_ledger(training_event_ledger)
        self.border_rect = battle_state["border_rect"]
        self.border_color = battle_state["border_color"]
        self.player1 = battle_state["player1"]
        self.player2 = battle_state["player2"]
        self.player1_ships = player1_ships
        self.player2_ships = player2_ships
        self.training_episode_kills: tuple[int, ...] = ()
        self.training_episode_deaths: tuple[int, ...] = ()
        self._training_pending_explosions = []
        if getattr(self, "training_mode", False):
            self._prepare_training_spawn(self.player1)
            self._prepare_training_spawn(self.player2)
        self.training_spawn_initialized = self.training_mode
        self._notify_round_started()

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
            if self.entry_animations_enabled
            else None
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
                    ship.visual_effects_enabled = self.visual_effects_enabled

    def set_visual_effects_enabled(self, enabled):
        self.visual_effects_enabled = bool(enabled)
        for ship in (self.player1, self.player2):
            ship.visual_effects_enabled = self.visual_effects_enabled

    def _notify_round_started(self):
        for ship in (self.player1, self.player2):
            ship.on_round_started()

    @property
    def game_objects(self):
        """Compatibility access to the World's authoritative object list."""
        return self.world.objects

    @game_objects.setter
    def game_objects(self, objects):
        self.world = World(objects)
        self.world.set_training_event_ledger(
            getattr(self, "training_event_ledger", None)
        )

    def _initial_key_states(self):
        return {
            self.settings[f"Player {player}: {control}"]: False
            for player in (1, 2)
            for control in CONTROL_NAMES
        }

    def step(self, actions=None, key_changes=None, timing_seconds=None):
        if not self.running:
            return self.state()

        self.frame_id += 1
        self.training_episode_kills = ()
        self.training_episode_deaths = ()
        training_event_ledger = getattr(self, "training_event_ledger", None)
        if training_event_ledger is not None:
            training_event_ledger.current_frame = self.frame_id
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
            stage_started_at = _battle_timing_started_at(timing_seconds)
            self._process_ship_inputs(excluded_ships)
            _add_battle_timing_seconds(
                timing_seconds,
                "simulation_ship_inputs",
                stage_started_at,
            )
            stage_started_at = _battle_timing_started_at(timing_seconds)
            self._update_tracking_lists()
            _add_battle_timing_seconds(
                timing_seconds,
                "simulation_tracking",
                stage_started_at,
            )
            stage_started_at = _battle_timing_started_at(timing_seconds)
            self._update_objects(excluded_ships)
            _add_battle_timing_seconds(
                timing_seconds,
                "simulation_update_objects",
                stage_started_at,
            )
            collision_metrics = (
                CollisionMetrics()
                if timing_seconds is not None
                else None
            )
            stage_started_at = _battle_timing_started_at(timing_seconds)
            handle_collisions(
                self.world,
                rng=getattr(self, "rng", None),
                resources=getattr(self, "resources", None),
                excluded_objects=excluded_ships,
                metrics=collision_metrics,
                visual_effects_enabled=getattr(
                    self,
                    "visual_effects_enabled",
                    True,
                ),
            )
            _add_battle_timing_seconds(
                timing_seconds,
                "simulation_collision",
                stage_started_at,
            )
            _add_collision_timing_counts(timing_seconds, collision_metrics)
            stage_started_at = _battle_timing_started_at(timing_seconds)
            self._update_aftermath()
            _add_battle_timing_seconds(
                timing_seconds,
                "simulation_aftermath",
                stage_started_at,
            )
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

            if self._apply_direct_actions(player, player_actions):
                continue

            player_actions = player_actions or {}
            for control in CONTROL_NAMES:
                key = self.settings[f"Player {player}: {control}"]
                pressed = self._action_pressed(player_actions, control)
                if self.key_states.get(key) != pressed:
                    self.handle_key_change(key, pressed)

    def _apply_direct_actions(self, player, player_actions):
        if isinstance(player_actions, Mapping):
            return False
        if not all(
            hasattr(player_actions, attribute)
            for attribute in ("thrust", "turn_left", "turn_right")
        ):
            return False

        ship = self.player1 if int(player) == 1 else self.player2
        direct_controls = (
            ("thrust", bool(player_actions.thrust)),
            ("turn_left", bool(player_actions.turn_left)),
            ("turn_right", bool(player_actions.turn_right)),
            (
                "action1",
                bool(
                    getattr(
                        player_actions,
                        "action1",
                        getattr(player_actions, "a1", False),
                    )
                ),
            ),
            (
                "action2",
                bool(
                    getattr(
                        player_actions,
                        "action2",
                        getattr(player_actions, "a2", False),
                    )
                ),
            ),
        )
        for control, pressed in direct_controls:
            ship.set_control_state(control, pressed, self.frame_id)
        return True

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
        projectiles = self.world.abilities_of_kind("projectile", "special_object")
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
            ship
            for ship in (self.player1, self.player2)
            if ship.current_hp <= 0 and ship.currently_alive
        ]
        if getattr(self, "training_mode", False):
            if newly_dead:
                self._start_training_deaths(newly_dead)
                self._respawn_training_ships(newly_dead)
            self._update_training_death_effects()
            return
        reborn_ships = [ship for ship in newly_dead if self._attempt_rebirth(ship)]
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
            self._notify_recorded_victory()

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
        self._notify_recorded_victory()

        if self.aftermath is None:
            return

        if (
            getattr(self, "entry", None) is None
            and not self.aftermath.pending_rebirths
            and battle_aftermath.aftermath_ready_for_selection(
                self.aftermath,
                self.frame_id,
                self.sound_enabled,
            )
        ):
            self.needs_selection = True

    def _prepare_training_spawn(self, ship):
        start_hp = max(
            1,
            int(getattr(ship, "start_hp", getattr(ship, "current_hp", 1))),
        )
        ship.current_hp = max(1, math.ceil(float(self.rng.random()) * start_hp))
        ship.currently_alive = True
        if getattr(ship, "name", None) == "Shofixti":
            ship.shofixti_arming_stage = getattr(ship, "ARMED", 2)
        spawn_satellites = getattr(ship, "spawn_satellites", None)
        if spawn_satellites is not None:
            spawn_satellites(rng=self.rng, randomized_health=True)
            self.world.add_all(ship.drain_spawned_objects())

    def _start_training_deaths(self, dead_ships):
        for ship in dead_ships:
            with use_audio_service(self.audio):
                BattleEffect.play_ship_death()
            schedule = battle_aftermath.create_ship_explosion_schedule(
                ship,
                self.frame_id,
                self.rng,
            )
            if self.visual_effects_enabled:
                self._training_pending_explosions.extend(schedule)

    def _update_training_death_effects(self):
        ready = [
            item
            for item in self._training_pending_explosions
            if item.frame <= self.frame_id
        ]
        self._training_pending_explosions = [
            item
            for item in self._training_pending_explosions
            if item.frame > self.frame_id
        ]
        for item in ready:
            self.world.add(
                BattleEffect.ship_explosion(item.position, scale=item.scale)
            )
            if item.is_final:
                with use_audio_service(self.audio):
                    BattleEffect.play_ship_death()

    def _respawn_training_ships(self, dead_ships):
        dead_ships = tuple(dead_ships)
        dead_players = tuple(sorted(ship.player for ship in dead_ships))
        credited_killers = set()
        for ship in dead_ships:
            source_owner = damage_source_owner(
                getattr(ship, "last_lethal_damage_source", None)
            )
            killer_player = getattr(source_owner, "player", None)
            if killer_player is not None and killer_player != ship.player:
                credited_killers.add(killer_player)
        living_ships = [
            ship
            for ship in (self.player1, self.player2)
            if ship not in dead_ships and ship.currently_alive and ship.current_hp > 0
        ]
        dead_by_player = {ship.player: ship for ship in dead_ships}
        for ship in dead_ships:
            battle_aftermath.release_dead_opponents(self.world, (ship,))
            battle_aftermath.hide_dead_ship(ship, self.world)

        planets = self.world.planets
        planet = planets[0] if planets else None
        replacements = []
        placement_objects = ship_spawn_obstacles(self.world)
        for player in dead_players:
            previous = dead_by_player[player]
            replacement = create_ship(
                previous.name,
                player,
                resources=self.resources,
                audio_service=self.audio,
            )
            self._bind_runtime_to_ships(replacement)
            if planet is not None:
                replacement.set_planet(planet)
            heading = self.rng.randint(0, const.SHIP_DIRECTIONS - 1)
            replacement.heading = heading
            replacement.rotation = heading * const.TURN_ANGLE
            position = get_training_spawn_position(
                self.rng,
                replacement,
                placement_objects,
            )
            replacement.initialize_in_battle(position, heading)
            replacements.append(replacement)
            placement_objects.append(replacement)

        ships_by_player = {ship.player: ship for ship in living_ships}
        ships_by_player.update({ship.player: ship for ship in replacements})
        self.player1 = ships_by_player[1]
        self.player2 = ships_by_player[2]
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1

        close_start_vux = apply_vux_starting_conditions(
            self.player1,
            self.player2,
            preserved_ships=living_ships,
            rng=self.rng,
            arena_objects=[*ship_spawn_obstacles(self.world), *replacements],
            training_close_start_chance=const.TRAINING_VUX_CLOSE_START_CHANCE,
        )
        apply_training_starting_velocities(
            replacements,
            rng=self.rng,
            stationary_ships=close_start_vux,
        )

        for ship in replacements:
            self.world.add(ship)
            self._prepare_training_spawn(ship)
            ship.on_round_started()
        battle_aftermath.restore_reborn_opponents(self.world, replacements)
        reset_key_states(self.key_states)
        self.training_episode_kills = tuple(sorted(credited_killers))
        self.training_episode_deaths = dead_players

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
        battle_aftermath.restore_reborn_opponents(self.world, ready)

        living_ships = [
            ship
            for ship in (self.player1, self.player2)
            if ship.currently_alive and ship.current_hp > 0
        ]
        if len(living_ships) == 2:
            self.aftermath = None
            self.audio.start_battle_music()
        else:
            battle_aftermath.record_resolved_victory(
                self.aftermath,
                self.player1,
                self.player2,
            )

    def _notify_recorded_victory(self):
        if self.aftermath is None or self.aftermath.victory_notified:
            return
        victor = self.aftermath.initial_victor
        if victor is None:
            return
        on_battle_won = getattr(victor, "on_battle_won", None)
        if on_battle_won is not None:
            on_battle_won()
        self.aftermath.victory_notified = True

    def _reenter_reborn_ships(self, ships):
        obstacles = ship_spawn_obstacles(self.world, excluding=ships)
        if len(ships) == 2:
            positions = random_ship_positions(self.rng, obstacles, ships)
        else:
            ship = ships[0]
            opponent = self.player2 if ship is self.player1 else self.player1
            positions = (
                random_position_away_from(
                    opponent.position,
                    self.rng,
                    obstacles,
                    ship,
                ),
            )

        for ship, position in zip(ships, positions):
            ship.initialize_in_battle(
                position,
                self.rng.randint(0, const.SHIP_DIRECTIONS - 1),
            )
            ship.current_hp = ship.max_hp
            ship.currently_alive = True
            reset_ship_controls(ship)

        entering_ships = list(ships)
        trail_styles = {ship: ship.rebirth_entry_trail_style() for ship in ships}
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
        self._notify_round_started()
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
            if self.entry_animations_enabled
            else None
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
            "training_episode_kills": getattr(self, "training_episode_kills", ()),
            "training_episode_deaths": getattr(self, "training_episode_deaths", ()),
        }

    def winner(self):
        if self.aftermath is not None and self.aftermath.pending_rebirths:
            return None
        living = [
            ship
            for ship in (self.player1, self.player2)
            if ship.currently_alive and ship.current_hp > 0
        ]
        if len(living) == 1:
            return living[0]
        if len(living) == 0:
            return None
        return None


def run(
    screen,
    ship1: SpaceShip,
    ship2: SpaceShip,
    player1_ships=None,
    player2_ships=None,
    audio_service=None,
    menu_sound_manager=None,
    player1_ai=False,
    player2_ai=False,
):
    # Interactive AI/model dependencies are intentionally lazy so headless
    # training workers can import ``BattleSimulation`` without the model stack.
    global BattleAIManager, InferenceModelCache, TrainingModelRepository
    if BattleAIManager is None:
        from src.Battle.battle_ai import BattleAIManager as battle_ai_manager

        BattleAIManager = battle_ai_manager
    if InferenceModelCache is None:
        from src.training.model_loader import InferenceModelCache as model_cache

        InferenceModelCache = model_cache
    if TrainingModelRepository is None:
        from src.training.model_registry import (
            TrainingModelRepository as model_repository,
        )

        TrainingModelRepository = model_repository

    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    simulation = BattleSimulation(
        screen,
        ship1,
        ship2,
        player1_ships,
        player2_ships,
        audio_service=audio_service,
    )
    ai_model_repository = TrainingModelRepository(
        const.DEFAULT_MODELS_PATH,
        const.MODELS_PATH,
    )
    ai_model_cache = InferenceModelCache()
    if player1_ai or player2_ai:
        ai_model_cache.load_initial(ai_model_repository)
    ai_manager = BattleAIManager(
        {1: player1_ai, 2: player2_ai},
        repository=ai_model_repository,
        rng=getattr(simulation, "rng", random),
        model_cache=ai_model_cache,
    )
    ai_manager.bind_round(simulation)
    reset_ai_player_inputs(simulation, ai_manager)
    star_field = DisplayStarField(resources=simulation.resources)

    running = True
    is_paused = False
    resume_countdown_pending = False
    resume_countdown_background = None
    pygame.event.clear(pygame.KEYDOWN)
    pygame.event.clear(pygame.KEYUP)

    state = simulation.state()
    accumulated_key_changes = []
    fixed_step = FixedStepScheduler(const.FPS)

    while running:
        elapsed_seconds = clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    is_paused = not is_paused
                    fixed_step.reset()
                    _sync_interpolation_snapshots(simulation.world)
                    if is_paused:
                        simulation.audio.pause()
                    else:
                        resume_countdown_pending = True
                elif event.key == pygame.K_ESCAPE:
                    was_paused = is_paused
                    frozen_battle_frame = screen.copy()
                    if not was_paused:
                        simulation.audio.pause()
                    if confirm_end_match(screen, menu_sound_manager):
                        simulation.audio.stop_music()
                        running = False
                    elif not was_paused:
                        resume_countdown_pending = True
                        resume_countdown_background = frozen_battle_frame
                    clock.reset()
                elif event.key in simulation.key_states:
                    accumulated_key_changes.append((event.key, True))
            elif event.type == pygame.KEYUP:
                if event.key in simulation.key_states:
                    accumulated_key_changes.append((event.key, False))

        if not running:
            continue

        if is_paused or resume_countdown_pending:
            for key, pressed in filter_ai_key_changes(
                simulation,
                accumulated_key_changes,
                ai_manager,
            ):
                simulation.handle_key_change(key, pressed)
            accumulated_key_changes.clear()
            state = simulation.state()
            interp_t = 1.0
        else:
            physics_steps, interp_t = fixed_step.advance(elapsed_seconds)
            for _ in range(physics_steps):
                filtered_key_changes = filter_ai_key_changes(
                    simulation,
                    accumulated_key_changes,
                    ai_manager,
                )
                state = simulation.step(
                    actions=ai_manager.actions_for_frame(simulation),
                    key_changes=filtered_key_changes,
                )
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
                        preselect_player1=(
                            simulation.player1
                            if simulation.player1.currently_alive
                            else None
                        ),
                        preselect_player2=(
                            simulation.player2
                            if simulation.player2.currently_alive
                            else None
                        ),
                        choose_second_player=simulation.aftermath.choose_second_player,
                        audio_service=simulation.audio,
                        menu_sound_manager=menu_sound_manager,
                        player1_ai=player1_ai,
                        player2_ai=player2_ai,
                    )
                    simulation.select_next_round(selected)
                    ai_manager.bind_round(simulation)
                    reset_ai_player_inputs(simulation, ai_manager)
                    pygame.event.clear(pygame.KEYDOWN)
                    pygame.event.clear(pygame.KEYUP)
                    fixed_step.reset()
                    clock.reset()
                    if not simulation.running:
                        running = False
                        break
                    simulation.audio.pause()
                    resume_countdown_pending = True
                    state = simulation.state()
                    interp_t = 0.0
                    break

        if not running:
            continue

        # Preserve the exact frame behind an Escape confirmation. Redrawing it
        # here can move interpolation forward even though simulation is paused.
        if resume_countdown_background is None:
            draw_battle(
                screen,
                simulation.world,
                simulation.border_rect,
                simulation.border_color,
                star_field,
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
                interp_t=interp_t,
                ai_labels={
                    player: label
                    for player in (1, 2)
                    if (label := ai_manager.label_for_player(player)) is not None
                },
            )

        if resume_countdown_pending:
            from src.Menus import pick_ship

            reset_key_states(simulation.key_states)
            reset_ship_controls(simulation.player1)
            reset_ship_controls(simulation.player2)
            accumulated_key_changes.clear()
            pick_ship.show_battle_countdown(
                screen,
                background=resume_countdown_background or screen.copy(),
                overlay_alpha=128,
            )
            pygame.event.clear(pygame.KEYDOWN)
            pygame.event.clear(pygame.KEYUP)
            fixed_step.reset()
            clock.reset()
            simulation.audio.unpause()
            resume_countdown_pending = False
            resume_countdown_background = None


def _sync_interpolation_snapshots(game_objects):
    for obj in World.coerce(game_objects):
        if hasattr(obj, "position") and hasattr(obj, "previous_position"):
            obj.previous_position = obj.position.copy()
        if hasattr(obj, "heading") and hasattr(obj, "previous_heading"):
            obj.previous_heading = obj.heading


def filter_ai_key_changes(simulation, key_changes, ai_manager):
    """Remove player action key changes for AI-owned sides."""
    return [
        (key, pressed)
        for key, pressed in key_changes or []
        if not _is_ai_action_key(simulation, key, ai_manager)
    ]


def reset_ai_player_inputs(simulation, ai_manager):
    """Clear stale human controls from ships currently owned by AI."""
    for player in (1, 2):
        if not ai_manager.is_ai_player(player):
            continue
        for control in CONTROL_NAMES:
            key = simulation.settings[f"Player {player}: {control}"]
            if key in simulation.key_states:
                simulation.key_states[key] = False
        ship = simulation.player1 if player == 1 else simulation.player2
        reset_ship_controls(ship)


def _is_ai_action_key(simulation, key, ai_manager):
    binding_for_key = getattr(simulation, "_binding_for_key", None)
    if binding_for_key is None:
        return False
    binding = binding_for_key(key)
    return binding is not None and ai_manager.is_ai_player(binding[0])


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
        ship
        for ship in (previous_player1, previous_player2)
        if ship in selected_ships and ship.is_alive()
    }

    persistent_objects = world.objects_excluding_types(
        SpaceShip, Ability, ThrustMarker, BattleEffect
    )
    preserved_abilities = [
        obj
        for obj in world.abilities
        if (obj.parent in preserved_ships and obj.is_alive())
    ]
    world.retain(persistent_objects + preserved_abilities)

    planets = world.planets
    planet = planets[0] if planets else None

    entering_ships = initialize_new_round_ships(
        selected_ships,
        preserved_ships,
        planet,
        rng=rng,
        arena_objects=ship_spawn_obstacles(world),
    )

    player1.opponent = player2
    player2.opponent = player1
    update_preserved_abilities(preserved_abilities, player1, player2, planet)

    from src.Battle.battle_init import apply_vux_starting_conditions

    apply_vux_starting_conditions(
        player1,
        player2,
        preserved_ships,
        rng=rng,
        arena_objects=ship_spawn_obstacles(world, excluding=selected_ships),
    )

    world.add_all(selected_ships)
    return entering_ships


def stop_tracking_projectiles(game_objects):
    for obj in World.coerce(game_objects).abilities:
        if obj.is_alive():
            obj.stop_and_track()


def initialize_new_round_ships(
    selected_ships,
    preserved_ships,
    planet,
    *,
    rng=None,
    arena_objects=(),
):
    rng = rng or random
    new_ships = [ship for ship in selected_ships if ship not in preserved_ships]
    preserved_list = list(preserved_ships)

    if len(new_ships) == 2:
        positions = list(random_ship_positions(rng, arena_objects, new_ships))
    elif len(new_ships) == 1 and preserved_list:
        positions = [
            random_position_away_from(
                preserved_list[0].position,
                rng,
                arena_objects,
                new_ships[0],
            )
        ]
    else:
        positions = []

    for ship, position in zip(new_ships, positions):
        ship.initialize_in_battle(position, rng.randint(0, const.SHIP_DIRECTIONS - 1))
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
        if ability.target is None or not World.is_alive(ability.target):
            ability.target = opponent
        if planet:
            ability.planet = planet


def random_position_away_from(position, rng=None, arena_objects=(), ship=None):
    from src.Battle.battle_init import (
        fallback_ship_positions,
        get_random_position,
        validate_ship_position,
        validate_ship_positions,
    )

    rng = rng or random

    for _ in range(1000):
        candidate = get_random_position(rng)
        if validate_ship_positions(position, candidate) and validate_ship_position(
            candidate, arena_objects, ship
        ):
            return candidate

    # A mocked or pathological RNG must not force an unsafe placement. Search a
    # deterministic lattice before reporting that the arena has no clear spot.
    for candidate in fallback_ship_positions():
        if validate_ship_positions(position, candidate) and validate_ship_position(
            candidate, arena_objects, ship
        ):
            return candidate

    raise RuntimeError("Unable to find a clear ship position")


def random_ship_positions(rng=None, arena_objects=(), ships=()):
    from src.Battle.battle_init import get_valid_ship_positions

    return get_valid_ship_positions(rng or random, arena_objects, ships)


def ship_spawn_obstacles(game_objects, excluding=()):
    """Return live arena bodies that a newly placed ship must avoid."""
    world = World.coerce(game_objects)
    excluded_ids = {id(obj) for obj in excluding}
    candidates = [
        obj
        for obj in world
        if isinstance(obj, (SpaceShip, Planet, Asteroid))
        or (
            isinstance(obj, Ability)
            and getattr(obj, "type", None) in ("projectile", "special_object")
        )
    ]
    return [
        obj
        for obj in candidates
        if (
            id(obj) not in excluded_ids
            and getattr(obj, "currently_alive", True)
            and getattr(obj, "current_hp", 1) > 0
        )
    ]


def reset_key_states(key_states):
    for key in key_states:
        key_states[key] = False


def reset_ship_controls(ship):
    ship.reset_controls()
