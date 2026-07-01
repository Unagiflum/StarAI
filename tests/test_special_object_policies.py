import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.collision_capabilities import (
    ProjectileContactPolicy,
    SameTypeContactPolicy,
)
from src.Objects.Ships.ability import (
    Ability,
    SPECIAL_OBJECT_AREA_IMMUNITIES,
)
from src.Objects.Ships.registry import create_ability, create_ship


class SpecialObjectPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False

    @classmethod
    def tearDownClass(cls):
        Ability.sound_enabled = cls.sound_enabled

    def make_ability(self, ship_name, ability_name, player=1):
        parent = create_ship(ship_name, player)
        opponent = create_ship("Earthling", 2 if player == 1 else 1)
        parent.initialize_in_battle([500, 500], 0)
        opponent.initialize_in_battle([900, 900], 0)
        parent.opponent = opponent
        opponent.opponent = parent
        return create_ability(ability_name, parent)

    def test_laser_and_area_capabilities_match_special_object_roles(self):
        for ship_name, ability_name, blocks_lasers in (
            ("Orz", "OrzA3", True),
            ("Chenjesu", "ChenjesuA2", True),
            ("KzerZa", "KzerZaA2", False),
            ("Vux", "VuxA2", False),
            ("Syreen", "SyreenCrew", False),
        ):
            with self.subTest(ability=ability_name):
                ability = self.make_ability(ship_name, ability_name)
                self.assertTrue(ability.laser_target_capabilities.vulnerable)
                self.assertEqual(
                    ability.laser_target_capabilities.blocks_lasers,
                    blocks_lasers,
                )
                self.assertEqual(
                    ability.area_damage_capabilities.immune_to_sources,
                    SPECIAL_OBJECT_AREA_IMMUNITIES,
                )

    def test_projectile_contact_policies_match_special_object_roles(self):
        for ship_name, ability_name, expected_policy in (
            (
                "Orz",
                "OrzA3",
                ProjectileContactPolicy.TAKE_DAMAGE_AND_DESTROY_PROJECTILE,
            ),
            (
                "Chenjesu",
                "ChenjesuA2",
                ProjectileContactPolicy.TAKE_DAMAGE_AND_DESTROY_PROJECTILE,
            ),
            ("KzerZa", "KzerZaA2", ProjectileContactPolicy.FRAGILE),
            ("Vux", "VuxA2", ProjectileContactPolicy.FRAGILE),
            ("Syreen", "SyreenCrew", ProjectileContactPolicy.FRAGILE),
            ("Melnorme", "MelnormeA2", ProjectileContactPolicy.TAKE_DAMAGE),
        ):
            with self.subTest(ability=ability_name):
                ability = self.make_ability(ship_name, ability_name)
                capabilities = ability.special_object_collision_capabilities
                self.assertIs(capabilities.projectile_contact_policy, expected_policy)

    def test_same_type_policies_distinguish_orz_and_chenjesu(self):
        marine = self.make_ability("Orz", "OrzA3")
        cloud = self.make_ability("Chenjesu", "ChenjesuA2")

        self.assertIs(
            marine.special_object_collision_capabilities.same_type_contact_policy,
            SameTypeContactPolicy.IGNORE,
        )
        self.assertIs(
            cloud.special_object_collision_capabilities.same_type_contact_policy,
            SameTypeContactPolicy.BOUNCE,
        )

    def test_fragile_role_uses_physical_collision_capability(self):
        for ship_name, ability_name in (
            ("KzerZa", "KzerZaA2"),
            ("Vux", "VuxA2"),
            ("Syreen", "SyreenCrew"),
        ):
            with self.subTest(ability=ability_name):
                ability = self.make_ability(ship_name, ability_name)
                self.assertTrue(ability.physical_collision_capabilities.is_fragile)


if __name__ == "__main__":
    unittest.main()
