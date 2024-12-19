from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const

class EarthlingA1(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("EarthlingA1", parent)
