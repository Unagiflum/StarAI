import math
import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

import src.const as const
from src.configuration import GameSettings
from src.Objects.Ships.registry import create_ability, create_ship
from src.resources import AssetManager


class TurnCreditTests(unittest.TestCase):
    def setUp(self):
        self.original_settings = GameSettings(
            const.DEFAULT_KEYS,
            asteroid_count=const.ASTEROID_COUNT,
            ship_directions=const.SHIP_DIRECTIONS,
            repeat_key_delay=const.INPUT_REPEAT_DELAY_FRAMES,
        )
        const.apply_game_settings(
            GameSettings(
                const.DEFAULT_KEYS,
                asteroid_count=const.ASTEROID_COUNT,
                ship_directions=64,
                repeat_key_delay=const.INPUT_REPEAT_DELAY_FRAMES,
            )
        )
        self.resources = AssetManager()

    def tearDown(self):
        const.apply_game_settings(self.original_settings)

    def make_ship(self, name="Earthling"):
        ship = create_ship(name, 1, resources=self.resources)
        ship.initialize_in_battle([500.0, 500.0], 0)
        ship.in_battle = True
        return ship

    def test_fresh_press_turns_one_fine_step(self):
        ship = self.make_ship()

        ship.set_control_state("turn_right", True, frame_id=0)
        ship.process_controls(frame_id=0)

        self.assertEqual(ship.heading, 1)
        self.assertEqual(ship.rotation, const.TURN_ANGLE)
        self.assertEqual(ship.turn_credits, 3)

    def test_held_repeat_paces_full_credits_at_configured_turn_rate(self):
        ship = self.make_ship()
        ship.turn_wait = 9
        ship.set_control_state("turn_right", True, frame_id=0)

        headings = []
        for frame_id in range(10):
            ship.process_controls(frame_id=frame_id)
            headings.append(ship.heading)

        # The immediate press is followed by paced repeats. Four fine steps
        # are spread over ten frames instead of being spent in one burst when
        # the repeat delay expires.
        self.assertEqual(headings, [1, 1, 1, 2, 2, 3, 3, 3, 4, 4])
        self.assertEqual(ship.heading, 4)

    def test_fractional_progress_preserves_configured_average_rate(self):
        ship = self.make_ship()
        ship.turn_wait = 4
        ship.turn_credits = 0
        ship.turn_credit_progress = 0
        ship._turn_credit_cooldown = None

        earned = []
        for _ in range(5):
            previous = ship.turn_credits
            ship.update_timers()
            earned.append(ship.turn_credits - previous)

        self.assertEqual(earned, [0, 1, 1, 1, 1])
        self.assertEqual(ship.turn_credits, const.DIRECTIONS_MULTIPLIER)

    def test_multi_step_turn_stops_at_first_blocked_heading(self):
        ship = self.make_ship()

        with mock.patch.object(
            ship,
            "rotation_would_overlap",
            side_effect=(False, True),
        ):
            turned = ship.turn_right(const.DIRECTIONS_MULTIPLIER)

        self.assertEqual(turned, 1)
        self.assertEqual(ship.heading, 1)
        self.assertEqual(ship.turn_credits, 3)

    def test_tracking_projectile_spends_partial_credits_smoothly(self):
        ship = self.make_ship()
        target = create_ship("Shofixti", 2, resources=self.resources)
        target.initialize_in_battle([700.0, 500.0], 0)
        ship.opponent = target
        target.opponent = ship
        missile = create_ability("EarthlingA1", ship)

        self.assertEqual(missile.turn_credits, 0)
        for _ in range(4):
            missile.update_heading()

        self.assertAlmostEqual(missile.rotation, 22.5)
        self.assertAlmostEqual(math.hypot(*missile.velocity), missile.speed)

    def test_slylandro_animation_cache_is_independent_of_heading_count(self):
        ship = self.make_ship("Slylandro")

        self.assertEqual(
            len(ship.base_sprites),
            const.ASSET_SPRITE_DIRECTIONS * const.VIDEO_FPS_MULTIPLIER,
        )
        self.assertEqual(
            len(self.resources.ship("Slylandro").sprites),
            const.TOTAL_SPRITE_DIRECTIONS,
        )

    def test_orz_turret_keeps_16_facings_relative_to_fine_ship_heading(self):
        ship = self.make_ship("Orz")
        ship.heading = 1
        ship.rotation = ship.heading * const.TURN_ANGLE

        ship.turret.turn(1)

        self.assertEqual(ship.turret.relative_heading, 1)
        self.assertEqual(
            ship.turret_heading,
            ship.heading + const.DIRECTIONS_MULTIPLIER,
        )

        for _ in range(const.ASSET_SPRITE_DIRECTIONS - 1):
            ship.turret.turn(1)

        self.assertEqual(ship.turret.relative_heading, 0)
        self.assertEqual(ship.turret_heading, ship.heading)

    def test_slylandro_lightning_uses_only_16_quantized_directions(self):
        ship = self.make_ship("Slylandro")
        target = self.make_ship("Earthling")
        ship.opponent = target
        target.opponent = ship
        target.trackable = False
        bolt = ship.plan_action1().spawned_objects[0]
        bolt.rng = mock.Mock()
        bolt.rng.choice.side_effect = lambda values: values[0]
        bolt.rng.randrange.side_effect = [1, 15]
        bolt.rng.uniform.return_value = 20

        bolt._generate_bolt(2)

        bolt.rng.randrange.assert_has_calls([mock.call(16), mock.call(16)])
        for start, end in bolt.collision_segments():
            dx, dy = end[0] - start[0], end[1] - start[1]
            angle = math.degrees(math.atan2(dx, -dy)) % 360
            quantized = angle / (360 / const.ASSET_SPRITE_DIRECTIONS)
            self.assertAlmostEqual(quantized, round(quantized))


if __name__ == "__main__":
    unittest.main()
