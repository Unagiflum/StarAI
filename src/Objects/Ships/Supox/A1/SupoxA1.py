from src.Objects.Ships.Projectile import Projectile

class SupoxA1(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("SupoxA1", parent)
