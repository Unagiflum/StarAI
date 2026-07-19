from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.KzerZa.A1.KzerZaA1 import KzerZaA1
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class KzerZa(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.fighter_launch_count = 0
        self.fighter_formation_direction = None

    action_factories = {1: KzerZaA1}

    def plan_action2(self):
        if not self.can_action2() or self.current_hp <= 1:
            return ActionPlan.invalid(2)

        fighter_count = min(2, self.current_hp - 1)
        if self.fighter_formation_direction is None:
            self.fighter_formation_direction = self.rng.choice((-1, 1))
        first_index = self.fighter_launch_count
        definition = ABILITY_DEFINITIONS["KzerZaA2"]
        launch_points = list(
            zip(definition.gun_locations, definition.gun_directions)
        )
        if self.fighter_formation_direction < 0:
            launch_points.reverse()
        special_objects = [
            KzerZaA2(
                self,
                direction,
                first_index + offset,
                location,
                formation_direction=self.fighter_formation_direction,
            )
            for offset, (location, direction) in enumerate(
                launch_points[:fighter_count]
            )
        ]

        def commit_launch_indices():
            self.fighter_launch_count += fighter_count

        return self.prepare_action_plan(
            2,
            special_objects,
            crew_change=-fighter_count,
            side_effects=(commit_launch_indices,),
        )
