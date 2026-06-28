import math
import src.const as const
from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from src.collision_capabilities import SpecialObjectCollisionCapabilities

class SyreenCrew(Ability):
    def __init__(self, parent, position):
        super().__init__("SyreenCrew", parent)
        self.position = position
        self.expiration_timer = 384
        self.speed = 5
        self.turn_wait = 0  # Can change direction instantly
        
        # The prompt says: drawn as an anti-aliased green circle with diameter 4.
        # Wait, the sprites are usually loaded from file_path, but they don't have a sprite.
        # However, the user said "drawn as an anti-aliased green circle with diameter 4".
        # This implies it might need a custom draw method or we just create a surface?
        # Let's look at how drawing is handled later.
        definition = ABILITIES_DATA["SyreenCrew"]
        radius = definition.radius if definition.radius is not None else 2
        self.size = [radius * 2, radius * 2]
        self.damages = [0]
        self.current_damage = 0
        self.hp_array = [1]
        self.current_hp = 1
        
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
            dx = self.parent.position[0] - self.position[0]
            dy = self.parent.position[1] - self.position[1]
            
            # Wrap around arena
            if dx > const.ARENA_SIZE / 2: dx -= const.ARENA_SIZE
            elif dx < -const.ARENA_SIZE / 2: dx += const.ARENA_SIZE
            if dy > const.ARENA_SIZE / 2: dy -= const.ARENA_SIZE
            elif dy < -const.ARENA_SIZE / 2: dy += const.ARENA_SIZE
            
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
            
        # Apply planet gravity
        if self.parent and getattr(self.parent, "planet", None):
            planet = self.parent.planet
            p_dx = planet.position[0] - self.position[0]
            p_dy = planet.position[1] - self.position[1]
            
            if p_dx > const.ARENA_SIZE / 2: p_dx -= const.ARENA_SIZE
            elif p_dx < -const.ARENA_SIZE / 2: p_dx += const.ARENA_SIZE
            if p_dy > const.ARENA_SIZE / 2: p_dy -= const.ARENA_SIZE
            elif p_dy < -const.ARENA_SIZE / 2: p_dy += const.ARENA_SIZE
            
            distance = math.hypot(p_dx, p_dy)
            if planet.diameter / 2 <= distance <= const.GRAVITY_RANGE:
                gravity_force = const.GRAVITY_MULTIPLIER * planet.gravity
                if not hasattr(self, "gravity_velocity"):
                    self.gravity_velocity = [0.0, 0.0]
                self.gravity_velocity[0] += gravity_force * p_dx / distance
                self.gravity_velocity[1] += gravity_force * p_dy / distance
                
        if hasattr(self, "gravity_velocity"):
            self.velocity[0] += self.gravity_velocity[0]
            self.velocity[1] += self.gravity_velocity[1]
        
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
