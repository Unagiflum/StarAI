from src.Objects.Ships.Projectile import Projectile

class ThraddashA2(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("ThraddashA2", parent)
