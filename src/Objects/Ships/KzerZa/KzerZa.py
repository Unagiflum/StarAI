# KzerZa.py
from src.Objects.Ships.SpaceShip import SpaceShip
import src.Objects.Ships.KzerZa.KzerZaA1 as A1
import src.Objects.Ships.KzerZa.KzerZaA2 as A2

class KzerZa(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
