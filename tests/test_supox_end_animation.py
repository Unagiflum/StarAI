import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

from src.Battle import collision_responses
from src.Battle.effects import BattleEffect
from src.Objects.Ships.Chmmr.A3.ChmmrSatellite import ChmmrSatellite
from src.Objects.Ships.registry import create_ability, create_ship
from src.audio import NullAudioService


class SupoxEndAnimationTests(unittest.TestCase):
    """The Supox shot shows its end animation only when it hits something."""

    def setUp(self):
        self.supox = create_ship("Supox", 1, audio_service=NullAudioService())
        self.chmmr = create_ship("Chmmr", 2, audio_service=NullAudioService())
        self.supox.initialize_in_battle([1000.0, 1300.0], 0)
        self.chmmr.initialize_in_battle([1000.0, 1000.0], 0)
        self.supox.opponent = self.chmmr
        self.chmmr.opponent = self.supox

    def shot(self, name="SupoxA1", parent=None):
        shot = create_ability(name, parent or self.supox)
        shot.position = [1200.0, 1200.0]
        shot.previous_position = shot.position.copy()
        shot.velocity = [0.0, 0.0]
        self.assertTrue(shot.death_animation)
        return shot

    def end_animations(self, action):
        with (
            mock.patch.object(
                BattleEffect, "from_animation", return_value=object()
            ) as from_animation,
            mock.patch.object(BattleEffect, "from_blast", return_value=object()),
            mock.patch.object(BattleEffect, "play_boom"),
        ):
            action()
        return [call.args[1] for call in from_animation.call_args_list]

    def test_only_the_supox_shot_is_marked(self):
        self.assertTrue(self.shot().end_anim_on_hit_only)
        spathi = create_ship("Spathi", 1, audio_service=NullAudioService())
        self.assertFalse(create_ability("SpathiA1", spathi).end_anim_on_hit_only)

    def test_animation_plays_when_the_shot_hits_a_satellite(self):
        shot = self.shot()
        satellite = ChmmrSatellite(self.chmmr)
        satellite.position = shot.position.copy()
        satellite.previous_position = satellite.position.copy()

        animations = self.end_animations(
            lambda: collision_responses.resolve_projectile_projectile_collision(
                satellite, shot, []
            )
        )

        self.assertEqual(satellite.current_hp, 9)
        self.assertFalse(shot.currently_alive)
        self.assertIn(shot.death_animation, animations)

    def test_no_animation_when_a_laser_shoots_the_shot_down(self):
        shot = self.shot()

        animations = self.end_animations(
            lambda: collision_responses.apply_projectile_laser_impact(
                shot, [], [0.0, -1.0], 1, shot.position.copy()
            )
        )

        self.assertFalse(shot.currently_alive)
        self.assertEqual(animations, [])

    def test_no_animation_when_area_damage_destroys_the_shot(self):
        shot = self.shot()

        animations = self.end_animations(
            lambda: collision_responses.apply_projectile_area_damage(
                self.chmmr, shot, [], [1.0, 0.0], 1.0, 1
            )
        )

        self.assertFalse(shot.currently_alive)
        self.assertEqual(animations, [])

    def test_other_shots_still_animate_when_shot_down(self):
        earthling = create_ship("Earthling", 1, audio_service=NullAudioService())
        earthling.initialize_in_battle([1000.0, 1300.0], 0)
        shot = self.shot("EarthlingA1", earthling)

        animations = self.end_animations(
            lambda: collision_responses.apply_projectile_laser_impact(
                shot, [], [0.0, -1.0], shot.current_hp, shot.position.copy()
            )
        )

        self.assertIn(shot.death_animation, animations)


if __name__ == "__main__":
    unittest.main()
