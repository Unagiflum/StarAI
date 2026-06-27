from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.KzerZa.A1.KzerZaA1 import KzerZaA1
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2


class KzerZa(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.fighter_launch_count = 0

    action_factories = {1: KzerZaA1}

    def plan_action2(self):
        if not self.can_action2() or self.current_hp <= 1:
            return ActionPlan.invalid(2)

        fighter_count = min(2, self.current_hp - 1)
        first_index = self.fighter_launch_count
        special_objects = [
            KzerZaA2(self, launch_angle, first_index + offset)
            for offset, launch_angle in enumerate((135, 225)[:fighter_count])
        ]

        def commit_launch_indices():
            self.fighter_launch_count += fighter_count

        return self.prepare_action_plan(
            2,
            special_objects,
            crew_change=-fighter_count,
            side_effects=(commit_launch_indices,),
        )
