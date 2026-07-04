from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
import src.const as const


class UtwigA2(Ability):
    """Timed Utwig damage shield. Absorbs damage to grant energy."""

    blocks_damage = True

    def __init__(self, parent):
        super().__init__("UtwigA2", parent)
        self.can_move = False
        self.can_die = False
        self.can_expire = True
        self.can_collide = False
        self._active = False
        self.recharge_on_planet = ABILITY_DEFINITIONS["UtwigA2"].recharge_on_planet

        self.gain_sound = None
        if self.audio_service:
            self.gain_sound = self.audio_service.load_effect(
                const.source_path("Objects/Ships/Utwig/A2/") / "UtwigA2Gain.wav",
                const.SOUND_EFFECT_VOLUME,
            )

    def activate(self):
        previous = self.parent._active_damage_shield
        if previous is not None and previous is not self:
            previous.currently_alive = False
            previous._active = False
        self.parent._active_damage_shield = self
        self._active = True
        # UQM sets the counter to 12 and decrements it later in the activation
        # frame, so the next preprocess observes 11.
        self.parent._shield_drain_timer = self.parent.SHIELD_COUNTER_RESET - 1
        # UQM's shield has no activation cooldown after it is released.
        self.parent.action2_timer = 0

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
        return True

    def is_alive(self):
        return self.currently_alive

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        return None

    def absorb_damage(self, damage):
        if damage <= 0:
            return
        self.parent.queue_shield_energy(damage, self.gain_sound)
