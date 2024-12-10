# Shofixti.py
from src.Ships.GameObject import SpaceShip
import src.Ships.Shofixti.ShofixtiA1 as A1
import src.Ships.Shofixti.ShofixtiA2 as A2

class Shofixti(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
