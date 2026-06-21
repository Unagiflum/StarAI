from src.Objects.Ships.space_ship import SpaceShip

class Orz(SpaceShip):
    def plan_action3(self):
        return self.validate_action(3)

    def handles_combined_action(self):
        return True
