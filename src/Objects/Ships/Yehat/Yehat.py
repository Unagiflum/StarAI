from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Yehat.A1.YehatA1 import YehatA1
from src.Objects.Ships.Yehat.A2.YehatA2 import YehatA2


class Yehat(SpaceShip):
    def plan_action1(self):
        side_offset = self.size[0] / 3
        return self.validate_action(
            1,
            lambda ship: [
                YehatA1(ship, offset) for offset in (-side_offset, side_offset)
            ],
        )

    def plan_action2(self):
        if not self.can_action2():
            return self.validate_action(2)

        shield = YehatA2(self)
        return self.prepare_action_plan(
            2,
            shield,
            side_effects=(shield.activate,),
        )

    def set_sprite(self):
        if self.damage_shield_is_active():
            return self._active_damage_shield.sprites[self.heading]
        return super().set_sprite()
