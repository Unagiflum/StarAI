import math
import src.const as const
from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from src.collision_capabilities import SpecialObjectCollisionCapabilities
from src.toroidal import wrapped_delta
class SyreenCrew(Ability):
    def __init__(self, parent, position):
        super().__init__("SyreenCrew", parent)
        self.position = position
        definition = ABILITIES_DATA["SyreenCrew"]
        self.expiration_timer = definition.life_time
        self.speed = definition.speed
        self.turn_wait = 0  # Can change direction instantly
        
        # The prompt says: drawn as an anti-aliased green circle with diameter 4.
        # Wait, the sprites are usually loaded from file_path, but they don't have a sprite.
        # However, the user said "drawn as an anti-aliased green circle with diameter 4".
        # This implies it might need a custom draw method or we just create a surface?
        # Let's look at how drawing is handled later.
        radius = definition.radius if definition.radius is not None else 2
        self.size = [radius * 2, radius * 2]
        self.damages = list(definition.damage)
        self.current_damage = definition.damage[0]
        self.hp_array = list(definition.start_hp)
        self.current_hp = definition.start_hp[0]
        
        if getattr(self.parent, "planet", None):
            self.set_planet(self.parent.planet)
        
        self.special_object_collision_capabilities = SpecialObjectCollisionCapabilities(
            collides_with_planets=True,
            collides_with_asteroids=True,
            damages_asteroids=False,
            collides_with_projectiles=True,
            damages_projectiles=False,
            collides_with_enemy_ships=True,
            collides_with_friendly_ships=True,
            collides_with_fighters=True,
            bounces_off_same_type=True
        )

    def handle_ship_contact(self, ship, normal):
        # Either ship, if it collides with the crew, will recover them and gain hit points.
        # "Either ship can recover the crew even if at full health, but doing so will not increase crew at that point."
        if ship.current_hp < ship.max_hp:
            ship.current_hp += 1
        self.current_hp = 0
        self.currently_alive = False
        
        from src.audio import active_audio_service
        audio = active_audio_service()
        if audio:
            audio.play_effect(
                const.source_path("Objects/Ships/Syreen/A2/SyreenA2Pickup.wav"),
                const.SOUND_EFFECT_VOLUME,
            )
            
        return True

    def can_recover_with_parent(self):
        return True

    def recover_with_parent(self):
        self.handle_ship_contact(self.parent, None)

    def handle_projectile_contact(self, projectile):
        if getattr(projectile, "projectile_name", None) == self.projectile_name:
            return False
        # Destroyed on contact with projectiles (except bounces off same type handled by responses)
        self.current_hp = 0
        self.currently_alive = False
        return True

    def update_physics(self):
        # Slowly track toward the Syreen ship with speed 5
        # Move at quantized angles with increments of 45 degrees
        # Change direction and speed instantly.
        
        if self.parent and self.parent.currently_alive:
            dx, dy = wrapped_delta(self.position, self.parent.position)
            
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360
                
            # Quantize to 45 degree increments
            target_direction = round(target_angle / 45.0)
            target_angle = (target_direction * 45.0) % 360
            
            angle_rad = math.radians(target_angle)
            self.velocity = [
                math.sin(angle_rad) * self.speed,
                -math.cos(angle_rad) * self.speed,
            ]
            
        super().update_physics()

    def update(self):
        if self.current_hp <= 0:
            self.currently_alive = False
            return False
        return super().update()

    def get_sprite(self, interp_t=0.0):
        if not hasattr(self, "_cached_sprite"):
            import pygame
            radius = int(self.size[0] / 2)
            diameter = radius * 2
            self._cached_sprite = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            pygame.draw.circle(self._cached_sprite, (0, 255, 0), (radius, radius), radius)
        return self._cached_sprite

    def get_collision_mask(self):
        if not hasattr(self, "_cached_mask"):
            import pygame
            self._cached_mask = pygame.mask.from_surface(self.get_sprite())
        return self._cached_mask
