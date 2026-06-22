import os
from pathlib import Path
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.battle_entry import (
    BLACK,
    EntryTrailStyle,
    RED,
    YELLOW,
    entry_complete,
    entry_duration_frames,
    finish_entry,
    silhouette_color,
    silhouette_lines,
    silhouette_positions,
    start_entry,
    visible_silhouettes,
)
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability


class Ship:
    def __init__(self, position, rotation):
        self.position = list(position)
        self.previous_position = list(position)
        self.rotation = rotation
        self.velocity = [0.0, 0.0]
        self.currently_alive = True
        self.current_hp = 1
        self.trackable = True
        self.size = [80, 60]
        self.processed_frames = []
        self.update_count = 0

    def process_controls(self, frame_id):
        self.processed_frames.append(frame_id)
        return []

    def update(self):
        self.previous_position = self.position.copy()
        self.position[0] += self.velocity[0] * const.SPEED_SCALE
        self.position[1] += self.velocity[1] * const.SPEED_SCALE
        self.update_count += 1
        return True


class BattleEntryAnimationTests(unittest.TestCase):
    def test_standard_trail_runs_from_farthest_to_nearest_behind_heading(self):
        ship = Ship([1000, 1000], 90)
        positions = silhouette_positions(ship)

        self.assertEqual(len(positions), 12)
        self.assertEqual(
            positions[0],
            (
                (
                    1000
                    - (const.ENTRY_TRAIL_SILHOUETTES - 1)
                    * const.ENTRY_TRAIL_SPACING
                )
                % const.ARENA_SIZE,
                1000,
            ),
        )
        self.assertEqual(positions[-1], (1000, 1000))

    def test_style_can_render_four_close_diagonal_trails(self):
        ship = Ship([1000, 1000], 0)
        style = EntryTrailStyle(
            angles=(45, 135, 225, 315),
            spacing=max(ship.size) + 5,
        )
        lines = silhouette_lines(ship, style)

        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line) == 12 for line in lines))
        self.assertTrue(all(line[-1] == (1000, 1000) for line in lines))
        for line in lines:
            for first, second in zip(line, line[1:]):
                dx = (second[0] - first[0] + const.ARENA_SIZE / 2) \
                    % const.ARENA_SIZE - const.ARENA_SIZE / 2
                dy = (second[1] - first[1] + const.ARENA_SIZE / 2) \
                    % const.ARENA_SIZE - const.ARENA_SIZE / 2
                self.assertAlmostEqual(
                    (dx ** 2 + dy ** 2) ** 0.5,
                    style.spacing,
                )

        first_points = [line[0] for line in lines]
        self.assertLess(first_points[0][0], 1000)
        self.assertGreater(first_points[0][1], 1000)
        self.assertLess(first_points[1][0], 1000)
        self.assertLess(first_points[1][1], 1000)
        self.assertGreater(first_points[2][0], 1000)
        self.assertLess(first_points[2][1], 1000)
        self.assertGreater(first_points[3][0], 1000)
        self.assertGreater(first_points[3][1], 1000)

    def test_trail_wraps_around_the_arena(self):
        ship = Ship([10, 10], 90)

        final_position = silhouette_positions(ship)[-1]

        self.assertEqual(
            final_position,
            (10, 10),
        )

    def test_color_fades_yellow_to_red_to_black_then_disappears(self):
        midpoint = const.ENTRY_TRAIL_FADE_FRAMES // 2

        self.assertEqual(silhouette_color(0), YELLOW)
        self.assertEqual(silhouette_color(midpoint), RED)
        self.assertEqual(
            silhouette_color(const.ENTRY_TRAIL_FADE_FRAMES), BLACK
        )
        self.assertIsNone(
            silhouette_color(const.ENTRY_TRAIL_FADE_FRAMES + 1)
        )

    def test_silhouettes_appear_from_farthest_to_nearest(self):
        ship = Ship([1000, 1000], 0)
        existing_ship = Ship([2000, 2000], 0)
        animation = start_entry(
            (ship,), existing_ship, ship, frame_id=0
        )

        self.assertEqual(len(visible_silhouettes(animation, ship, 0)), 1)
        self.assertIs(animation.camera_targets[0], existing_ship)
        self.assertEqual(animation.camera_targets[1].position, (1000, 1000))

        existing_ship.position[0] = 2100
        ship.position[0] = 1100
        self.assertEqual(animation.camera_targets[0].position, [2100, 2000])
        self.assertEqual(animation.camera_targets[1].position, (1000, 1000))

        visible = visible_silhouettes(
            animation, ship, const.ENTRY_TRAIL_STAGGER_FRAMES
        )
        arrival_positions = silhouette_positions(ship, (1000, 1000))
        self.assertEqual(len(visible), 2)
        self.assertEqual(visible[0][0], arrival_positions[0])
        self.assertEqual(visible[1][0], arrival_positions[1])

    def test_only_marked_entries_use_four_trails(self):
        standard = Ship([1000, 1000], 0)
        reborn_pkunk = Ship([2000, 2000], 0)
        animation = start_entry(
            (standard, reborn_pkunk),
            standard,
            reborn_pkunk,
            frame_id=0,
            trail_styles={
                reborn_pkunk: EntryTrailStyle(
                    angles=(45, 135, 225, 315),
                    spacing=85,
                ),
            },
        )

        self.assertEqual(len(visible_silhouettes(animation, standard, 0)), 1)
        self.assertEqual(
            len(visible_silhouettes(animation, reborn_pkunk, 0)),
            4,
        )

    def test_animation_completes_after_full_trail_duration(self):
        ship = Ship([0, 0], 0)
        animation = start_entry(
            (ship,), ship, object(), frame_id=10
        )

        self.assertFalse(
            entry_complete(animation, 10 + entry_duration_frames() - 1)
        )
        self.assertTrue(
            entry_complete(animation, 10 + entry_duration_frames())
        )

    def test_arrival_silhouette_stays_yellow_until_ship_appears(self):
        ship = Ship([1000, 1000], 0)
        animation = start_entry((ship,), ship, object(), frame_id=10)
        final_start = (
            10
            + (const.ENTRY_TRAIL_SILHOUETTES - 1)
            * const.ENTRY_TRAIL_STAGGER_FRAMES
        )

        at_arrival = visible_silhouettes(animation, ship, final_start)
        before_completion = visible_silhouettes(
            animation,
            ship,
            10 + entry_duration_frames() - 1,
        )

        self.assertEqual(at_arrival[-1], ((1000, 1000), YELLOW))
        self.assertEqual(before_completion[-1], ((1000, 1000), YELLOW))

    def test_entry_temporarily_disables_entrant_targeting(self):
        first = Ship([100, 200], 0)
        second = Ship([1000, 1000], 0)

        entry = start_entry((first, second), first, second, frame_id=0)

        self.assertFalse(first.trackable)
        self.assertFalse(second.trackable)

        finish_entry(entry)

        self.assertTrue(first.trackable)
        self.assertTrue(second.trackable)

    def test_two_ship_entry_advances_all_non_entrant_objects(self):
        first = Ship([100, 200], 0)
        second = Ship([1000, 1000], 0)
        for ship in (first, second):
            ship.currently_alive = True
            ship.current_hp = 1

        asteroid = Asteroid.__new__(Asteroid)
        asteroid.update = mock.Mock(return_value=True)
        ability = Ability.__new__(Ability)
        ability.type = "projectile"
        ability.player = 1
        ability.update = mock.Mock(return_value=True)

        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.running = True
        simulation.frame_id = 0
        simulation.needs_selection = False
        simulation.aftermath = None
        simulation.player1 = first
        simulation.player2 = second
        simulation.game_objects = [first, second, asteroid, ability]
        simulation.key_states = {}
        simulation.audio = None
        simulation.resources = None
        simulation.entry = start_entry(
            (first, second), first, second, frame_id=0
        )

        with mock.patch("src.Battle.battle.handle_collisions"):
            simulation.step()

        asteroid.update.assert_called_once_with()
        ability.update.assert_called_once_with()
        self.assertEqual(first.update_count, 0)
        self.assertEqual(second.update_count, 0)

    def test_simulation_advances_existing_ship_and_excludes_entrant(self):
        existing_ship = Ship([100, 200], 0)
        existing_ship.velocity = [10, -5]
        existing_ship.currently_alive = True
        existing_ship.current_hp = 1
        entering_ship = Ship([1000, 1000], 0)
        entering_ship.currently_alive = True
        entering_ship.current_hp = 1

        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.running = True
        simulation.frame_id = 0
        simulation.needs_selection = False
        simulation.aftermath = None
        simulation.player1 = existing_ship
        simulation.player2 = entering_ship
        simulation.game_objects = [existing_ship, entering_ship]
        simulation.key_states = {}
        simulation.audio = None
        simulation.resources = None
        simulation.rng = None
        simulation.sound_enabled = False
        simulation.entry = start_entry(
            (entering_ship,),
            existing_ship,
            entering_ship,
            frame_id=0,
        )

        with mock.patch(
            "src.Battle.battle.handle_collisions"
        ) as handle_collisions:
            state = simulation.step()

        self.assertIs(state["entry"], simulation.entry)
        self.assertEqual(
            existing_ship.position,
            [
                100 + 10 * const.SPEED_SCALE,
                200 - 5 * const.SPEED_SCALE,
            ],
        )
        self.assertEqual(entering_ship.position, [1000, 1000])
        self.assertEqual(existing_ship.processed_frames, [1])
        self.assertEqual(entering_ship.processed_frames, [])
        self.assertEqual(
            handle_collisions.call_args.kwargs["excluded_objects"],
            (entering_ship,),
        )


if __name__ == "__main__":
    unittest.main()
