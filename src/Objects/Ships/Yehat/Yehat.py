from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Yehat.A1.YehatA1 import YehatA1


class Yehat(SpaceShip):
    def perform_action1(self):
        side_offset = self.size[0] / 2
        return self.execute_action(
            1,
            lambda ship: [
                YehatA1(ship, offset) for offset in (-side_offset, side_offset)
            ],
        )
