"""Phase 6B: computer controllers see enemy objects only once they are a few
frames old."""

import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pygame

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.world import World
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.configuration import GameSettingsRepository
from src.persistence import PersistenceValidationError
from src.resources import HeadlessAssetManager
from src.training.baseline import arena
from src.training.baseline.engine import ACTION_INDEX_BY_MASK
from src.training.contracts import ACTION_INDEX_TABLE, OBSERVATION_FIELD_NAMES
from src.training.model_registry import current_game_settings_metadata
from src.training.observation import build_observation_context, encode_observation
from src.training.process_worker import CoordinatedSimulationWorker
from src.training.session import validate_model_metadata
from src.training.worker_protocol import StartRunCommand

pygame.init()


def _ship(name, **overrides):
    values = {
        "name": name,
        "max_hp": 18,
        "current_hp": 12,
        "rotation": 0,
        "velocity": [0.0, 0.0],
        "trackable": True,
        "input_pressed_frames": {},
        "newly_pressed_controls": set(),
        "limpets_attached": 0,
        "boarded_marines": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _obj(name, **overrides):
    values = {
        "name": name,
        "position": [0.0, 0.0],
        "velocity": [0.0, 0.0],
        "currently_alive": True,
        "current_hp": 1,
        "current_damage": 0,
        "can_expire": False,
        "expiration_timer": 0,
        "type": "projectile",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _field(name):
    return OBSERVATION_FIELD_NAMES.index(name)


def _present(observation, group, slot=0):
    return observation[_field(f"object.{group}.{slot}.present")]


def _delay(frames):
    return mock.patch.object(const, "PERCEPTION_DELAY_FRAMES", frames)


class _Duel:
    """Two ships and a World whose frame the test sets."""

    def __init__(self, mine="Earthling", theirs="Mycon"):
        self.me = _ship(mine, player=1, position=[100.0, 100.0])
        self.enemy = _ship(theirs, player=2, position=[600.0, 100.0])
        self.world = World()
        self.world.add(self.me)
        self.world.add(self.enemy)

    def add(self, obj, frame):
        self.world.frame_id = frame
        self.world.add(obj)
        return obj

    def observe(self, frame, *, viewer=None, context=None):
        self.world.frame_id = frame
        viewer = viewer or self.me
        other = self.enemy if viewer is self.me else self.me
        return encode_observation(
            viewer, other, frame_id=frame, game_objects=self.world, context=context
        )


class EnemyObjectDelayTests(unittest.TestCase):
    def test_enemy_shot_is_absent_for_the_delay_and_present_once_it_reaches_it(self):
        duel = _Duel()
        duel.add(_obj("MyconA1", parent=duel.enemy, player=2, position=[300.0, 100.0]), 10)

        with _delay(5):
            seen = [_present(duel.observe(frame), "enemy_a1") for frame in range(10, 17)]

        self.assertEqual(seen, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def test_ships_own_objects_and_neutral_objects_are_seen_at_once(self):
        duel = _Duel()
        frame = 20
        duel.add(_obj("EarthlingA1", parent=duel.me, player=1, position=[150.0, 100.0]), frame)
        duel.add(
            _obj("Planet", type="space", gravity=1.0, diameter=100, position=[900.0, 900.0]),
            frame,
        )
        duel.add(_obj("Asteroid", type="space", position=[200.0, 300.0]), frame)
        duel.add(_obj("SyreenCrew", parent=duel.enemy, player=2, position=[250.0, 100.0]), frame)
        respawned = duel.add(_ship("Mycon", player=2, position=[700.0, 100.0]), frame)
        duel.enemy = respawned

        with _delay(5):
            observation = duel.observe(frame)

        self.assertEqual(_present(observation, "enemy_ship"), 1.0)
        self.assertEqual(_present(observation, "friendly_a1"), 1.0)
        self.assertEqual(_present(observation, "planet"), 1.0)
        self.assertEqual(_present(observation, "asteroid"), 1.0)
        self.assertEqual(_present(observation, "syreen_crew"), 1.0)

    def test_a_delay_of_zero_gives_todays_observations(self):
        me = _ship("Earthling", player=1, position=[100.0, 100.0])
        enemy = _ship("Mycon", player=2, position=[600.0, 100.0])
        objects = [
            me,
            enemy,
            _obj("MyconA1", parent=enemy, player=2, position=[300.0, 100.0]),
            _obj("EarthlingA1", parent=me, player=1, position=[150.0, 100.0]),
        ]
        # Objects that never entered a World carry no age, as before.
        today = encode_observation(me, enemy, frame_id=3, game_objects=list(objects))
        world = World()
        world.frame_id = 3
        world.add_all(objects)

        with _delay(0):
            delayed = encode_observation(me, enemy, frame_id=3, game_objects=world)

        self.assertEqual(delayed, today)

    def test_a_held_back_object_leaks_into_no_count_or_slot(self):
        duel = _Duel(theirs="Chmmr")
        duel.add(_obj("ChmmrSatellite", parent=duel.enemy, player=2, position=[600.0, 360.0]), 30)
        duel.add(_obj("KzerZaA2", parent=duel.enemy, player=2, position=[500.0, 100.0]), 30)
        duel.add(
            _obj("OrzA3", parent=duel.enemy, player=2, mode="outbound", position=[400.0, 100.0]),
            30,
        )

        with _delay(5):
            young = duel.observe(33)
            old = duel.observe(35)

        empty_slots = [0.0] * (len(young) - _field("object.planet.0.present"))
        self.assertEqual(young[_field("object.planet.0.present"):], empty_slots)
        for count in ("chmmr_satellites", "ur_quan_fighters", "orz_marines_floating"):
            self.assertEqual(young[_field(f"enemy.{count}")], 0.0, count)
            self.assertGreater(old[_field(f"enemy.{count}")], 0.0, count)
        self.assertEqual(_present(old, "enemy_non_a1"), 1.0)

    def test_one_shared_context_filters_each_viewer_separately(self):
        duel = _Duel()
        duel.add(_obj("MyconA1", parent=duel.enemy, player=2, position=[300.0, 100.0]), 50)
        duel.add(_obj("EarthlingA1", parent=duel.me, player=1, position=[400.0, 100.0]), 50)
        duel.world.frame_id = 51
        context = build_observation_context(duel.me, duel.enemy, duel.world)

        with _delay(5):
            mine = duel.observe(51, context=context)
            theirs = duel.observe(51, viewer=duel.enemy, context=context)
            mine_later = duel.observe(55)

        # Each ship sees its own shot at once and the other's only later.
        self.assertEqual(_present(mine, "friendly_a1"), 1.0)
        self.assertEqual(_present(mine, "enemy_a1"), 0.0)
        self.assertEqual(_present(theirs, "friendly_a1"), 1.0)
        self.assertEqual(_present(theirs, "enemy_a1"), 0.0)
        self.assertEqual(_present(mine_later, "enemy_a1"), 1.0)

    def test_a_respawned_enemys_new_objects_wait_for_the_delay(self):
        duel = _Duel(theirs="Chmmr")
        duel.add(_obj("ChmmrSatellite", parent=duel.enemy, player=2, position=[600.0, 360.0]), 0)
        duel.enemy.currently_alive = False
        reborn = duel.add(_ship("Chmmr", player=2, position=[800.0, 100.0]), 100)
        duel.add(_obj("ChmmrSatellite", parent=reborn, player=2, position=[800.0, 360.0]), 100)
        duel.enemy = reborn

        with _delay(5):
            fresh = duel.observe(102)
            settled = duel.observe(105)

        self.assertEqual(_present(fresh, "enemy_ship"), 1.0)
        self.assertEqual(fresh[_field("enemy.chmmr_satellites")], 0.0)
        # The dead Chmmr's satellite is old, so it is still seen.
        self.assertEqual(_present(fresh, "enemy_non_a1"), 1.0)
        self.assertEqual(_present(fresh, "enemy_non_a1", 1), 0.0)
        self.assertEqual(settled[_field("enemy.chmmr_satellites")], 1 / 3)
        self.assertEqual(_present(settled, "enemy_non_a1", 1), 1.0)


def _battle(mine, theirs, gap):
    resources = HeadlessAssetManager()
    audio = NullAudioService()
    simulation = BattleSimulation(
        None,
        create_ship(mine, 1, resources=resources, audio_service=audio),
        create_ship(theirs, 2, resources=resources, audio_service=audio),
        audio_service=audio,
        rng=random.Random(3),
        include_stars=False,
        resources=resources,
    )
    simulation.world.objects[:] = [
        obj for obj in simulation.world.objects if type(obj).__name__ != "Asteroid"
    ]
    me, enemy = simulation.player1, simulation.player2
    me.position = [2000.0, 2000.0]
    enemy.position = [2000.0, 2000.0 + gap]
    for ship in (me, enemy):
        ship.velocity = [0.0, 0.0]
        ship.heading = 0
        ship.rotation = 0.0
    return simulation


_IDLE = ACTION_INDEX_TABLE[0]
_FIRE = ACTION_INDEX_TABLE[ACTION_INDEX_BY_MASK[8]]
_POINT_DEFENCE = ACTION_INDEX_TABLE[ACTION_INDEX_BY_MASK[16]]


class GameRulesSeeTheLiveWorldTests(unittest.TestCase):
    def _young_shot_fate(self, mine, gap, my_action):
        """Fire a Supox shot at ``mine``; return whether the ship's own
        observation showed it, and whether a laser shot it down, in the
        frames before the shot reaches the delay."""

        simulation = _battle(mine, "Supox", gap)
        simulation.step(actions={1: _IDLE, 2: _IDLE})
        simulation.step(actions={1: _IDLE, 2: _FIRE})
        shots = [obj for obj in simulation.world.objects if obj.name == "SupoxA1"]
        self.assertEqual(len(shots), 1)
        shot = shots[0]
        seen = shot_down = False
        # Four frames: younger than the default delay of 5.
        for _ in range(4):
            observation = encode_observation(
                simulation.player1,
                simulation.player2,
                frame_id=simulation.frame_id,
                game_objects=simulation.world,
            )
            seen = seen or _present(observation, "enemy_a1") == 1.0
            simulation.step(actions={1: my_action, 2: _IDLE})
            shot_down = shot_down or any(
                getattr(obj, "target", None) is shot for obj in simulation.world.objects
            )
        return seen, shot_down and not shot.currently_alive

    def test_chmmr_satellites_shoot_a_projectile_younger_than_the_delay(self):
        with _delay(5):
            seen, shot_down = self._young_shot_fate("Chmmr", 500.0, _IDLE)

        self.assertFalse(seen)
        self.assertTrue(shot_down)

    def test_held_earthling_point_defence_shoots_one_down(self):
        with _delay(5):
            seen, shot_down = self._young_shot_fate("Earthling", 300.0, _POINT_DEFENCE)

        self.assertFalse(seen)
        self.assertTrue(shot_down)

    def test_without_a_delay_the_ship_sees_the_shot_at_once(self):
        with _delay(0):
            seen, _ = self._young_shot_fate("Earthling", 300.0, _POINT_DEFENCE)

        self.assertTrue(seen)


class PerceptionDelaySettingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "game.json"
        self.repository = GameSettingsRepository(self.path, const.DEFAULT_KEYS)

    def test_default_is_five_frames(self):
        self.assertEqual(const.DEFAULT_GAMEPLAY["perception_delay_frames"], 5)
        self.assertEqual(self.repository.default().perception_delay_frames, 5)

    def test_values_from_zero_to_twenty_four_are_saved_and_loaded(self):
        for frames in (0, 24):
            settings = self.repository.codec.decode({"perception_delay_frames": frames})
            self.repository.save(settings)
            self.assertEqual(self.repository.load().perception_delay_frames, frames)

    def test_values_outside_the_range_are_rejected(self):
        for frames in (-1, 25):
            # A stored value out of range falls back to the default...
            loaded = self.repository.codec.decode({"perception_delay_frames": frames})
            self.assertEqual(loaded.perception_delay_frames, 5)
            # ...and is never saved.
            with self.assertRaises(PersistenceValidationError):
                self.repository.save(
                    dataclasses.replace(loaded, perception_delay_frames=frames)
                )

    def test_a_settings_file_without_it_loads_the_default(self):
        self.path.write_text(json.dumps({"asteroid_count": 7}), encoding="utf-8")

        settings = self.repository.load()

        self.assertEqual(settings.asteroid_count, 7)
        self.assertEqual(settings.perception_delay_frames, 5)

    def test_applying_settings_sets_the_runtime_delay(self):
        settings = self.repository.codec.decode({"perception_delay_frames": 8})
        with _delay(5):
            with mock.patch.object(const, "_recompute_direction_constants"):
                with mock.patch.multiple(
                    const,
                    ASTEROID_COUNT=const.ASTEROID_COUNT,
                    DIRECTIONS_MULTIPLIER=const.DIRECTIONS_MULTIPLIER,
                    INPUT_REPEAT_DELAY_FRAMES=const.INPUT_REPEAT_DELAY_FRAMES,
                ):
                    const.apply_game_settings(settings)
                    self.assertEqual(const.PERCEPTION_DELAY_FRAMES, 8)


class PerceptionDelayModelTests(unittest.TestCase):
    def test_model_metadata_records_the_delay(self):
        with _delay(3):
            self.assertEqual(current_game_settings_metadata()["perception_delay_frames"], 3)

    def test_a_model_without_it_was_trained_at_zero(self):
        with _delay(5):
            report = validate_model_metadata(
                {"game_settings": {"ship_directions": const.SHIP_DIRECTIONS}}
            )

        self.assertIn("perception delay differs: model 0, current 5", report.warnings)

    def test_a_matching_delay_gives_no_warning(self):
        with _delay(0):
            report = validate_model_metadata(
                {"game_settings": {"ship_directions": const.SHIP_DIRECTIONS}}
            )

        self.assertFalse(
            [warning for warning in report.warnings if "perception delay" in warning]
        )


class PerceptionDelayProcessTests(unittest.TestCase):
    def test_coordinated_workers_take_the_delay_from_the_start_command(self):
        worker = CoordinatedSimulationWorker()
        command = StartRunCommand(
            worker_id=1,
            record_id=2,
            base_seed=3,
            video_fps_multiplier=const.VIDEO_FPS_MULTIPLIER,
            perception_delay_frames=9,
        )

        with _delay(5):
            worker.handle(command)
            self.assertEqual(const.PERCEPTION_DELAY_FRAMES, 9)

    def test_arena_defaults_to_five_and_takes_the_option(self):
        self.assertEqual(arena.parse_arguments([]).perception_delay, 5)
        self.assertEqual(
            arena.parse_arguments(["--perception-delay", "0"]).perception_delay, 0
        )
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                arena.parse_arguments(["--perception-delay", "25"])

    def test_arena_settings_apply_the_delay(self):
        with _delay(5), mock.patch.object(const, "_recompute_direction_constants"):
            with mock.patch.multiple(
                const,
                ASTEROID_COUNT=const.ASTEROID_COUNT,
                DIRECTIONS_MULTIPLIER=const.DIRECTIONS_MULTIPLIER,
                INPUT_REPEAT_DELAY_FRAMES=const.INPUT_REPEAT_DELAY_FRAMES,
            ):
                arena.apply_arena_settings(const.SHIP_DIRECTIONS, 3)
                self.assertEqual(const.PERCEPTION_DELAY_FRAMES, 3)


if __name__ == "__main__":
    unittest.main()
