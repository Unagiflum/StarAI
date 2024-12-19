from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const

class DruugeA1(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("DruugeA1", parent)
