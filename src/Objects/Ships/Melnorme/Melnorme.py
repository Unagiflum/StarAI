from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Melnorme.A1.MelnormeA1 import MelnormeA1
from src.Objects.Ships.Melnorme.A2.MelnormeA2 import MelnormeA2


class Melnorme(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.held_a1 = None

    def plan_action1(self):
        if self.held_a1 is not None and self.held_a1.is_alive():
            return ActionPlan.invalid(1)
        self.held_a1 = None
        if not self.can_action1():
            return ActionPlan.invalid(1)
        projectile = MelnormeA1(self)
        return self.prepare_action_plan(1, projectile)

    def perform_action1_release(self):
        if self.held_a1 is not None:
            self.held_a1.release()
        self.held_a1 = None

    def plan_action2(self):
        return self.validate_action(2, MelnormeA2)

    def energy_regeneration_enabled(self):
        return not self.action1_active
