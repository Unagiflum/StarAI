from src.Objects.object import PlayerObject
import src.const as const
import math
import pygame
import random
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    FighterCollisionCapabilities,
    LaserTargetCapabilities,
)
from src.Objects.Ships.catalog import ABILITIES_DATA, ABILITY_DEFINITIONS
from src.resources import default_assets
from src.audio import compatibility_audio_service
from src.toroidal import nearest_position, wrapped_delta


def wrapped_endpoint(start, end):
    return nearest_position(end, start)

class Ability(PlayerObject):
    sound_enabled = True

    def __init__(self, ability_name, parent):
        ability_definition = ABILITY_DEFINITIONS[ability_name]

        # Initialize with temporary size, will be set after sprite loading
        super().__init__(
            name=ability_name,
            sprite_location=const.source_path(ability_definition.file_path),
            sprite_scale=ability_definition.sprite_scale,
            size=[0, 0],
            player=parent.player
        )
        self.resources = getattr(parent, "resources", default_assets())
        self.rng = getattr(parent, "rng", random)
        self.audio_service = getattr(parent, "audio_service", None)
        if self.audio_service is None:
            self.audio_service = compatibility_audio_service(
                self.sound_enabled, self.resources
            )
        assets = self.resources.ability(ability_name)
        self.sizes = assets.sizes
        self.size = list(self.sizes[0]) if self.sizes else [0, 0]

        self.sprites = assets.sprites
        self.masks = assets.masks
        self.death_animation = assets.end_animation
        ability_definition = ABILITY_DEFINITIONS[ability_name]
        self.launch_sound = None
        if ability_definition.has_sound:
            self.launch_sound = self.audio_service.load_effect(
                const.source_path(ability_definition.file_path) / f"{ability_name}.wav",
                const.SOUND_EFFECT_VOLUME,
            )

        # Rest of initialization code
        self.parent = parent
        self.opponent = self.parent.opponent
        self.planet = self.parent.planet
        self.projectile_name = ability_name
        self.target = None

        # Basic properties
        self.type = ability_definition.ability_type
        self.collision_capabilities = CollisionCapabilities({
            'projectile': CollisionRole.PROJECTILE,
            'fighter': CollisionRole.FIGHTER,
            'laser': CollisionRole.LASER,
        }.get(self.type, CollisionRole.NONE))
        self.laser_target_capabilities = LaserTargetCapabilities(
            targetable=self.type in ("projectile", "fighter"),
            vulnerable=ability_definition.laser_vulnerable,
        )
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=self.type == "area",
            targetable=self.type != "laser",
        )
        self.area_damage_pending = False
        self.fighter_collision_capabilities = (
            FighterCollisionCapabilities(
                collides_with_planets=ability_definition.collide_planets,
                collides_with_asteroids=ability_definition.collide_asteroids,
                damages_asteroids=ability_definition.damage_asteroids,
                collides_with_projectiles=ability_definition.collide_projectiles,
                damages_projectiles=ability_definition.damage_projectiles,
                collides_with_enemy_ships=ability_definition.collide_enemy_ships,
                collides_with_friendly_ships=ability_definition.collide_friendly_ships,
                collides_with_fighters=ability_definition.collide_fighters,
            )
            if self.type == "fighter"
            else None
        )
        self.start_hp = ability_definition.start_hp[0]
        self.current_hp = self.start_hp
        self.damages = ability_definition.damage
        self.current_damage = self.damages[0]
        self.tracking = ability_definition.tracking
        self.parent_vel = ability_definition.parent_vel
        self.speed = ability_definition.speed * const.PROJ_SPEED_SCALE
        self.life_time = ability_definition.life_time
        self.turn_wait = ability_definition.turn_wait
        self.inertia = ability_definition.inertia
        self.hit_parent = ability_definition.hit_parent
        self.hit_self = ability_definition.hit_self
        self.has_left_parent = False
        self.omnidirectional = ability_definition.omnidirectional
        self.end_anim_count = ability_definition.end_anim

        # Animation properties
        self.frames = ability_definition.frames
        self.frame_delay = ability_definition.frame_delay
        self.current_frame = 0
        self.frame_timer = self.frame_delay

        # Store HP array for evolution
        self.hp_array = ability_definition.start_hp

        # State flags
        self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)
        self.can_move = True
        self.can_die = True
        self.can_expire = True

        if self.type in ('laser', 'projectile', 'fighter'):
            self.can_collide = True
        else:
            self.can_collide = False

        self.expiration_timer = int(self.life_time * const.PROJ_LIFE_SCALE)

    @classmethod
    def preload_resources(cls, ability_name, resources=None):
        """Warm the provider's cache without constructing an ability."""
        resources = resources or default_assets()
        return resources.ability(ability_name)

    def damage_at_distance(self, distance):
        """Return area damage at a radial distance from this ability."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define radial area damage"
        )

    def area_damage_for_target(self, target, distance):
        """Return damage for one eligible area-damage target."""
        return self.damage_at_distance(distance)

    def on_area_damage_hit(self, target, damage):
        """Handle ability-owned state after area damage is applied."""

    def update_heading(self):

        if self.omnidirectional:
            self.heading = 0
        else:
            direction_step = 360 / const.SHIP_DIRECTIONS
            self.heading = int((self.rotation % 360) / direction_step) % const.SHIP_DIRECTIONS

        if self.tracking and self._live_trackable_opponent():
            # Find opponent
            dx, dy = wrapped_delta(self.position, self.opponent.position)

            # Calculate target angle
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360

            # Quantize to nearest available direction
            direction_step = const.TURN_ANGLE
            current_angle = self.rotation
            target_direction = round(target_angle / direction_step)
            target_angle = (target_direction * direction_step) % 360

            # Find shortest turning direction
            angle_diff = target_angle - current_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            # Turn if timer allows
            if self.turn_timer <= 0:
                if self.opponent.trackable:
                    if abs(angle_diff) >= direction_step:
                        self.rotation = (current_angle + (direction_step if angle_diff > 0 else -direction_step)) % 360
                        self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)
                    else:
                        self.rotation = target_angle
            else:
                self.turn_timer -= 1
            angle_rad = math.radians(self.rotation)
            self.velocity = [math.sin(angle_rad) * self.speed, -math.cos(angle_rad) * self.speed]


    def update_physics(self):
        self.update_heading()
        self.apply_speed_limit()
        if self.inertia:
            self.apply_verlet()
        else:
            self.position[0] = (self.position[0] + self.velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE
            self.position[1] = (self.position[1] + self.velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.update_physics()
        self.expiration_timer -= 1

        # Handle frame animation if projectile evolves
        if self.frames > 1:
            if self.frame_timer <= 0:
                if self.current_frame < self.frames - 1:
                    self.current_frame += 1
                    # Update properties for new frame
                    self.size = list(self.sizes[self.current_frame])
                    self.current_damage = self.damages[min(self.current_frame, len(self.damages) - 1)]
                    if len(self.hp_array) > 1:
                        self.current_hp = min(self.current_hp, self.hp_array[self.current_frame])
                    self.frame_timer = self.frame_delay
            else:
                self.frame_timer -= 1

        if self.current_hp <= 0:
            self.currently_alive = False
            return False
        if self.type == 'laser':
            return self.expiration_timer >= 0 and self.current_hp > 0
        return self.expiration_timer > 0 and self.current_hp > 0

    def on_collide(self, target):
        if not self.hit_parent and target == self.parent:
            return False

        if target.current_hp is not None:
            take_damage = getattr(target, "take_damage", None)
            if take_damage is not None:
                take_damage(self.current_damage)
            else:
                target.current_hp = max(
                    0, target.current_hp - self.current_damage
                )

        return True

    def on_ship_impact(self, ship):
        """Apply projectile-specific behavior after damaging a ship."""
        return None

    def begin_planet_avoidance(self, planet, outward_normal):
        """React to fighter separation from a planet."""
        return None

    def can_recover_with_parent(self):
        """Return whether this fighter may currently recover into its parent."""
        return False

    def recover_with_parent(self):
        """Apply fighter-specific recovery behavior."""
        return None

    def stop_and_track(self):
        """Apply any ability-specific round-transition tracking behavior."""
        return None

    def on_opponent_lost(self, opponent):
        """Let an ability react without imposing one fighter behavior."""
        if self.opponent is opponent:
            self.opponent = None

    def _live_trackable_opponent(self):
        if (
            self.opponent is not None
            and self.opponent.currently_alive
            and self.opponent.current_hp > 0
            and self.opponent.trackable
        ):
            return self.opponent
        return None

    def calculate_end_position(self):
        """Update laser geometry when an ability supplies a directed beam."""
        return None

    def is_alive(self):
        return self.currently_alive and self.current_hp > 0


    def set_hp(self, new_hp):
        """Override hp setting to handle evolution and death"""
        if new_hp <= 0:
            self.current_hp = 0
            self.currently_alive = False
            return

        new_hp = min(new_hp, self.current_hp)
        if len(self.hp_array) > 1:
            damage_taken = self.current_hp - new_hp
            if damage_taken > 0:
                frame_advance = damage_taken * self.frame_delay
                self.frame_timer -= frame_advance

        self.current_hp = new_hp

    def get_sprite(self):
        if self.frames > 1:
            return self.sprites[0][self.current_frame]
        return self.sprites[self.heading]

    def draw(self, screen, scale_factor, translation):
        sprite = self.get_sprite()
        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (-scaled_rect.width <= pos_x <= const.SCREEN_HEIGHT + scaled_rect.width and
                        -scaled_rect.height <= pos_y <= const.SCREEN_HEIGHT + scaled_rect.height):
                    screen.blit(scaled_sprite, (
                        const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))

    def get_collision_mask(self):
        if not self.masks:
            return None
        if self.frames > 1:
            return self.masks[self.current_frame]
        return self.masks[self.heading]
