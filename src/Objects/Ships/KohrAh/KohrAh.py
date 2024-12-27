from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.KohrAh.A1.KohrAhA1 import KohrAhA1
from src.Objects.Ships.KohrAh.A2.KohrAhA2 import KohrAhA2
import src.const as const


class KohrAh(SpaceShip):

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]
        self.SAW_COUNT = ship_data.get("SAW_COUNT", 8)
        self.GAS_COUNT = ship_data.get("GAS_COUNT", 16)
        self.angle_increment = 360 / self.GAS_COUNT
        self.active_projectiles = []
        self.last_action1_state = False

    def perform_action1(self):
        # Update active_projectiles from friendly_objects
        self.active_projectiles = [obj for obj in self.friendly_objects if isinstance(obj, KohrAhA1)]

        button_pressed = self.action1_active and not self.last_action1_state
        button_released = not self.action1_active and self.last_action1_state
        self.last_action1_state = self.action1_active

        if button_pressed and self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            # Remove oldest projectile if at max
            if len(self.active_projectiles) >= self.SAW_COUNT:
                oldest = self.active_projectiles.pop(0)
                oldest.currently_alive = False

            # Create new projectile
            ability_obj = KohrAhA1(self)

            if ability_obj.launch_sound:
                ability_obj.launch_sound.play()

            self.active_projectiles.append(ability_obj)
            return ability_obj

        elif button_released:
            # Stop movement of active projectiles
            for proj in self.active_projectiles:
                if proj.currently_alive:
                    proj.is_moving = False

        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            projectiles = []

            for ii in range(self.GAS_COUNT):
                angle_offset = ii * self.angle_increment
                ability_obj = KohrAhA2(self, angle_offset)
                projectiles.append(ability_obj)

            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return projectiles
        return None

    def perform_action3(self):
        return None, False
