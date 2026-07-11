from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Utwig.A1.UtwigA1 import UtwigA1
from src.Objects.Ships.Utwig.A2.UtwigA2 import UtwigA2
import pygame


class Utwig(SpaceShip):
    SHIELD_DRAIN_INTERVAL = 6
    SHIELD_COUNTER_RESET = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shield_drain_timer = 0
        self._pending_shield_energy = 0
        self._pending_shield_gain_sound = None
        self._shield_silhouettes = []
        for s in self.sprites:
            m = pygame.mask.from_surface(s, threshold=80)
            self._shield_silhouettes.append(
                m.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
            )

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self._shield_drain_timer = 0
        self._pending_shield_energy = 0
        self._pending_shield_gain_sound = None

    def control_ready(self, control_name, frame_id):
        if control_name == "action2":
            # The Utwig shield is a held state, not a repeating key action.
            return True
        return super().control_ready(control_name, frame_id)

    def process_controls(self, frame_id=None):
        self._apply_pending_shield_energy()

        shield = getattr(self, "_active_damage_shield", None)
        shield_active = self.damage_shield_is_active()
        if not self.action2_active:
            if shield_active:
                shield.deactivate()
            self._shield_drain_timer = 0
            self.action2_timer = 0
        elif shield_active:
            counter = self._shield_drain_timer
            if counter % self.SHIELD_DRAIN_INTERVAL == 0:
                if not self.change_energy(-self.a2_cost):
                    # UQM leaves the shield up after the midpoint drain fails;
                    # it switches off if the next cycle cannot be started.
                    if counter == 0:
                        shield.deactivate()
                        self.action2_timer = 0
                elif counter == 0:
                    counter = self.SHIELD_COUNTER_RESET

            if self.damage_shield_is_active():
                self._shield_drain_timer = max(0, counter - 1)
            else:
                self._shield_drain_timer = 0

        return super().process_controls(frame_id)

    def queue_shield_energy(self, amount, gain_sound=None):
        self._pending_shield_energy += max(0, amount)
        if gain_sound is not None:
            self._pending_shield_gain_sound = gain_sound

    def _apply_pending_shield_energy(self):
        amount = getattr(self, "_pending_shield_energy", 0)
        if amount <= 0:
            return
        self._pending_shield_energy = 0
        self.change_energy(amount)
        sound = getattr(self, "_pending_shield_gain_sound", None)
        self._pending_shield_gain_sound = None
        if sound:
            sound.play()

    def plan_action1(self):
        return self.validate_action(
            1,
            lambda ship: UtwigA1.create_parallel_projectiles(ship),
        )

    def can_action1(self):
        if self.damage_shield_is_active():
            return False
        if self.action2_active and self.current_energy >= self.a2_cost:
            return False
        return super().can_action1()

    def can_action2(self):
        return not self.damage_shield_is_active() and super().can_action2()

    def plan_action2(self):
        if self.damage_shield_is_active() or not self.can_action2():
            return self.validate_action(2)

        shield = UtwigA2(self)
        return self.prepare_action_plan(
            2,
            shield,
            side_effects=(shield.activate,),
        )

    def take_damage(self, damage, *, shieldable=True, non_lethal=False, source=None):
        damage = max(0, damage)
        if damage <= 0 or self.current_hp <= 0:
            return 0
        if shieldable and self.damage_shield_is_active():
            shield = getattr(self, "_active_damage_shield", None)
            if shield and hasattr(shield, "absorb_damage"):
                shield.absorb_damage(damage)
            return 0

        return super().take_damage(
            damage,
            shieldable=shieldable,
            non_lethal=non_lethal,
            source=source,
        )

    def take_planet_impact_damage(self, damage, *, source=None):
        """Block planet damage without charging when the shield disables it."""
        shield = getattr(self, "_active_damage_shield", None)
        if (
            self.damage_shield_is_active()
            and shield is not None
            and not getattr(shield, "recharge_on_planet", True)
        ):
            return super().take_damage(damage, source=source)
        return self.take_damage(damage, source=source)

    def set_sprite(self, interp_t=0.0):
        sprite = super().set_sprite(interp_t)
        if self.damage_shield_is_active():
            shield = self._active_damage_shield
            timer = shield.expiration_timer - interp_t
            progress = max(0.0, timer) / shield._duration
            # Yellow (255, 255, 0) to Red (255, 0, 0)
            green_val = int(255 * progress)
            color = (255, green_val, 0, 255)

            from src.Battle.interpolation import interpolated_sprite_index

            sprite_idx = interpolated_sprite_index(self, interp_t)
            tinted = self._shield_silhouettes[sprite_idx].copy()
            tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
            return tinted

        return sprite
