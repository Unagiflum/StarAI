from src.Objects.Ships.ability import Ability


class YehatA2(Ability):
    """Timed Yehat damage shield and alternate-ship visual owner."""

    blocks_damage = True

    def __init__(self, parent):
        super().__init__("YehatA2", parent)
        self.can_move = False
        self.can_die = False
        self.can_expire = True
        self.can_collide = False
        self._active = False

    def activate(self):
        previous = self.parent._active_damage_shield
        if previous is not None and previous is not self:
            previous.currently_alive = False
            previous._active = False
        self.parent._active_damage_shield = self
        self._active = True

    def deactivate(self):
        self.currently_alive = False
        self._active = False
        if self.parent._active_damage_shield is self:
            self.parent._active_damage_shield = None

    def update(self):
        if not self.currently_alive:
            return False
        self.previous_position = self.position.copy()
        self.position = self.parent.position.copy()
        self.heading = self.parent.heading
        self.rotation = self.parent.rotation
        self.expiration_timer -= 1
        if self.expiration_timer <= 0:
            self.deactivate()
            return False
        return True

    def is_alive(self):
        return self.currently_alive

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        # The parent ship draws these sprites so its position and draw order are
        # unchanged. Its original masks remain authoritative for collisions.
        return None
