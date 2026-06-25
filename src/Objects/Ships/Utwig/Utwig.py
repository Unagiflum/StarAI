from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Utwig.A1.UtwigA1 import UtwigA1
from src.Objects.Ships.Utwig.A2.UtwigA2 import UtwigA2
import pygame


class Utwig(SpaceShip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shield_silhouettes = []
        for s in self.sprites:
            m = pygame.mask.from_surface(s, threshold=80)
            self._shield_silhouettes.append(
                m.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
            )

    def plan_action1(self):
        return self.validate_action(
            1,
            lambda ship: UtwigA1.create_parallel_projectiles(ship),
        )

    def plan_action2(self):
        if not self.can_action2():
            return self.validate_action(2)

        shield = UtwigA2(self)
        return self.prepare_action_plan(
            2,
            shield,
            side_effects=(shield.activate,),
        )

    def take_damage(self, damage, *, shieldable=True):
        damage = max(0, damage)
        if damage <= 0 or self.current_hp <= 0:
            return 0
        if shieldable and self.damage_shield_is_active():
            shield = getattr(self, "_active_damage_shield", None)
            if shield and hasattr(shield, "absorb_damage"):
                shield.absorb_damage(damage)
            return 0
        
        return super().take_damage(damage, shieldable=shieldable)

    def set_sprite(self):
        sprite = super().set_sprite()
        if self.damage_shield_is_active():
            shield = self._active_damage_shield
            progress = shield.expiration_timer / shield.life_time
            # Yellow (255, 255, 0) to Red (255, 0, 0)
            green_val = int(255 * progress)
            color = (255, green_val, 0, 255)
            
            tinted = self._shield_silhouettes[self.heading].copy()
            tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
            return tinted
            
        return sprite
