from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import pygame
import math
import src.const as const

class ArilouA2(Ability):
    def __init__(self, parent):
        super().__init__("ArilouA2", parent)
        ability_data = ABILITIES_DATA["ArilouA2"]
