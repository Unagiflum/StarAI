import math

from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.Chenjesu.A1.ChenjesuA1 import ChenjesuA1
from src.Objects.Ships.Chenjesu.A2.ChenjesuA2 import ChenjesuA2
from src.Objects.Ships.space_ship import SpaceShip


class Chenjesu(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.active_a1 = None

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.active_a1 = None

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)
        if self.active_a1 is not None and self.active_a1.is_alive():
            return ActionPlan.invalid(1)

        projectile = ChenjesuA1(self)

        def track_projectile():
            self.active_a1 = projectile

        return self.prepare_action_plan(
            1,
            projectile,
            side_effects=(track_projectile,),
        )

    def perform_action1_release(self):
        projectile = self.active_a1
        if projectile is not None:
            projectile.fragment()

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)

        definition = ABILITY_DEFINITIONS["ChenjesuA2"]
        active_count = sum(
            1
            for obj in self.friendly_objects
            if obj.name == "ChenjesuA2" and obj.is_alive()
        )
        if active_count >= definition.max_count:
            return ActionPlan.invalid(2)

        cloud = ChenjesuA2(self)

        def apply_launch_recoil():
            recoil_speed = cloud.mass * cloud.speed / self.mass
            launch_angle = math.radians(cloud.rotation)
            self.add_impulse(
                -math.sin(launch_angle) * recoil_speed,
                math.cos(launch_angle) * recoil_speed,
            )

        return self.prepare_action_plan(
            2,
            cloud,
            side_effects=(apply_launch_recoil,),
        )
