from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.KohrAh.A1.KohrAhA1 import KohrAhA1
from src.Objects.Ships.KohrAh.A2.KohrAhA2 import KohrAhA2


class KohrAh(SpaceShip):

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        ship_data = SHIPS_DATA[ship_name]
        self.SAW_COUNT = ship_data.get("SAW_COUNT", 8)
        self.GAS_COUNT = ship_data.get("GAS_COUNT", 16)
        self.angle_increment = 360 / self.GAS_COUNT
        self.active_projectiles = []
        self.last_action1_state = False

    def plan_action1(self):
        # Update active_projectiles from friendly_objects
        live_projectiles = [
            obj
            for obj in self.friendly_objects
            if isinstance(obj, KohrAhA1) and obj.currently_alive and obj.current_hp > 0
        ]
        self.active_projectiles = live_projectiles

        button_pressed = self.action1_active and not self.last_action1_state
        self.last_action1_state = self.action1_active

        if not button_pressed or not self.can_action1():
            return ActionPlan.invalid(1)

        retained = list(live_projectiles)
        oldest = None
        if len(retained) >= self.SAW_COUNT:
            oldest = retained.pop(0)
        saw = KohrAhA1(self)

        def commit_projectile_limit():
            if oldest is not None:
                oldest.currently_alive = False
            self.active_projectiles = retained + [saw]

        return self.prepare_action_plan(
            1,
            saw,
            side_effects=(commit_projectile_limit,),
        )

    def perform_action1_release(self):
        self.last_action1_state = False
        self.active_projectiles = [
            obj
            for obj in self.friendly_objects
            if isinstance(obj, KohrAhA1) and obj.currently_alive and obj.current_hp > 0
        ]
        for proj in self.active_projectiles:
            proj.stop_and_track()
        return None

    def plan_action2(self):
        return self.validate_action(
            2,
            lambda ship: [
                KohrAhA2(ship, index * ship.angle_increment)
                for index in range(ship.GAS_COUNT)
            ],
        )
