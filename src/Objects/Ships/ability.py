from src.Objects.object import PlayerObject
import src.const as const
import math
import pygame
import random
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    SpecialObjectCollisionCapabilities,
    LaserTargetCapabilities,
    PhysicalCollisionCapabilities,
)
from src.Objects.Ships.catalog import ABILITIES_DATA, ABILITY_DEFINITIONS
from src.resources import default_assets
from src.turn_credits import (
    accrue_turn_credits,
    available_turn_credits,
    initialize_turn_credits,
    spend_turn_credits,
)
from src.audio import compatibility_audio_service
from src.toroidal import nearest_position, wrapped_delta
from src.Objects.Ships.launch_geometry import gun_world_position, place_projectile_at_gun
from src.training import event_ledger


SPECIAL_OBJECT_AREA_IMMUNITIES = frozenset({"SlylandroA2", "SyreenA2"})


def wrapped_endpoint(start, end):
    return nearest_position(end, start)


def outward_visual_laser_start(start, end, distance):
    """Move a beam's drawn origin toward its end by a visual-only distance."""
    dx, dy = wrapped_delta(start, end)
    length = math.hypot(dx, dy)
    if length == 0:
        return list(start)

    distance = min(length, max(0.0, distance))
    return [
        start[0] + dx / length * distance,
        start[1] + dy / length * distance,
    ]


class Ability(PlayerObject):
    sound_enabled = True
    survives_parent_cleanup = False

    def __init__(self, ability_name, parent):
        ability_definition = ABILITY_DEFINITIONS[ability_name]

        # Initialize with temporary size, will be set after sprite loading
        super().__init__(
            name=ability_name,
            sprite_location=const.source_path(ability_definition.file_path),
            sprite_scale=ability_definition.sprite_scale,
            size=[0, 0],
            player=parent.player,
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
        self.anchor_offsets = assets.anchor_offsets
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
        event_ledger.inherit_credit(self, parent)
        self.opponent = self.parent.opponent
        self.planet = self.parent.planet
        self.projectile_name = ability_name
        self.target = None

        # Basic properties
        self.type = ability_definition.ability_type
        self.collision_capabilities = CollisionCapabilities(
            {
                "projectile": CollisionRole.PROJECTILE,
                "special_object": CollisionRole.SPECIAL_OBJECT,
                "laser": CollisionRole.LASER,
                "area": CollisionRole.AREA,
            }.get(self.type, CollisionRole.NONE)
        )
        
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(
            is_solid=False, is_projectile=True
        )
        
        self.laser_target_capabilities = LaserTargetCapabilities(
            targetable=self.type in ("projectile", "special_object"),
            vulnerable=ability_definition.laser_vulnerable,
            blocks_lasers=ability_definition.blocks_lasers,
        )
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=self.type == "area",
            targetable=self.type != "laser",
        )
        self.area_damage_pending = False
        self.special_object_collision_capabilities = SpecialObjectCollisionCapabilities(
            collides_with_planets=ability_definition.collide_planets,
            collides_with_asteroids=ability_definition.collide_asteroids,
            damages_asteroids=ability_definition.damage_asteroids,
            collides_with_projectiles=ability_definition.collide_projectiles,
            damages_projectiles=ability_definition.damage_projectiles,
            collides_with_enemy_ships=ability_definition.collide_enemy_ships,
            collides_with_friendly_ships=ability_definition.collide_friendly_ships,
            collides_with_fighters=ability_definition.collide_fighters,
        )
        self.is_psychic = getattr(ability_definition, 'is_psychic', False)
        self.ignores_shields = getattr(ability_definition, 'ignores_shields', False)
        self.start_hp = ability_definition.start_hp[0]
        self.current_hp = self.start_hp
        self.damages = ability_definition.damage
        self.current_damage = self.damages[0]
        self.tracking = ability_definition.tracking
        self.parent_vel = ability_definition.parent_vel
        self.speed = ability_definition.speed
        self.life_time = ability_definition.life_time
        self.turn_wait = ability_definition.turn_wait
        self.inertia = ability_definition.inertia
        self.hit_parent = ability_definition.hit_parent
        self.hit_self = ability_definition.hit_self
        self.hit_team = getattr(ability_definition, 'hit_team', False)
        self.omnidirectional = ability_definition.omnidirectional
        self.end_anim_count = ability_definition.end_anim

        # Animation properties
        self.frames = ability_definition.frames
        self.frame_delay = max(1, round(ability_definition.frame_delay))
        self.current_frame = 0
        # Abilities spawned by controls are updated once before their first draw.
        # The extra tick preserves the configured gameplay duration while letting
        # the first video frame start at the first source sprite.
        self.frame_timer = self.frame_delay + 1

        # Store HP array for evolution
        self.hp_array = ability_definition.start_hp
        # State flags
        # Generic tracking cadence is governed by turn credits. Specialized
        # abilities may still reuse turn_timer for their own behavior.
        self.turn_timer = 0
        initialize_turn_credits(self, full=False)
        self.can_move = True
        self.can_die = True
        self.can_expire = True

        if self.type in ("laser", "projectile", "special_object"):
            self.can_collide = True
        else:
            self.can_collide = False

        self._duration = round(self.life_time)
        self.expiration_timer = self._duration

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

    def configured_gun(self, index=0):
        definition = ABILITY_DEFINITIONS[self.name]
        locations = definition.gun_locations or ()
        if index >= len(locations):
            raise ValueError(f"Ability '{self.name}' has no gun location {index}")
        directions = definition.gun_directions or ()
        direction = directions[index] if index < len(directions) else None
        return locations[index], direction

    def configured_gun_position(self, index=0, *, rotation=None, position=None):
        location, _ = self.configured_gun(index)
        return gun_world_position(
            self.parent, location, rotation=rotation, position=position
        )

    def configure_laser_colors(self, colors):
        """Select the next configured color for a sequence of laser shots."""
        if not colors:
            raise ValueError(f"Ability '{self.name}' has no configured laser colors")
        cycles = getattr(self.parent, "_laser_color_cycles", None)
        if cycles is None:
            cycles = {}
            self.parent._laser_color_cycles = cycles
        index = cycles.get(self.name, 0)
        self.LASER_COLORS = colors
        self.LASER_COLOR = colors[index % len(colors)]
        cycles[self.name] = index + 1
        return self.LASER_COLOR

    def launch_from_gun(
        self,
        index=0,
        *,
        gun_location=None,
        relative_direction=None,
        gap_multiplier=None,
        inherit_parent_velocity=True,
        gun_rotation=None,
        launch_direction=None,
    ):
        configured_location, configured_direction = self.configured_gun(index)
        location = configured_location if gun_location is None else gun_location
        direction = (
            configured_direction
            if relative_direction is None
            else relative_direction
        )
        if direction is None:
            raise ValueError(f"Ability '{self.name}' gun {index} has no direction")
        return place_projectile_at_gun(
            self,
            location,
            direction,
            gap_multiplier=gap_multiplier,
            inherit_parent_velocity=inherit_parent_velocity,
            gun_rotation=gun_rotation,
            launch_direction=launch_direction,
        )

    def area_damage_for_target(self, target, distance):
        """Return damage for one eligible area-damage target."""
        return self.damage_at_distance(distance)

    def maximum_area_damage_radius(self):
        """Return a finite spatial-query radius, or ``None`` to force a scan.

        Radial abilities conventionally expose ``range``.  Mask-shaped area
        abilities override this method with their current mask radius.
        """
        effect_range = getattr(self, "range", None)
        if (
            isinstance(effect_range, (int, float))
            and math.isfinite(effect_range)
            and effect_range >= 0
        ):
            return float(effect_range)
        return None

    def on_area_damage_hit(self, target, damage):
        """Handle ability-owned state after area damage is applied."""

    def update_heading(self):

        if self.omnidirectional:
            self.heading = 0
        else:
            direction_step = 360 / const.SHIP_DIRECTIONS
            self.heading = (
                int((self.rotation % 360) / direction_step) % const.SHIP_DIRECTIONS
            )

        if self.tracking and hasattr(self, "turn_credit_capacity"):
            accrue_turn_credits(self, self.turn_wait)

        if self.tracking and self._live_trackable_opponent():
            # Find opponent
            dx, dy = wrapped_delta(self.position, self.opponent.position)

            # Calculate target angle
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360

            # Quantize to nearest available direction
            direction_step = const.TURN_ANGLE
            current_angle = self.rotation % 360
            target_direction = round(target_angle / direction_step)
            target_angle = (target_direction * direction_step) % 360

            # Find shortest turning direction
            angle_diff = target_angle - current_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            available_credits = available_turn_credits(self)
            if self.opponent.trackable and available_credits > 0:
                steps_to_target = round(abs(angle_diff) / direction_step)
                steps = min(
                    steps_to_target,
                    available_credits,
                    const.DIRECTIONS_MULTIPLIER,
                )
                if steps:
                    if math.isclose(abs(angle_diff), 180.0, abs_tol=1e-9):
                        direction = self.rng.choice((-1, 1))
                    else:
                        direction = 1 if angle_diff > 0 else -1
                    self.rotation = (
                        current_angle + direction * steps * direction_step
                    ) % 360
                    spend_turn_credits(self, steps)

            self.heading = (
                round(self.rotation / direction_step) % const.SHIP_DIRECTIONS
            )
            angle_rad = math.radians(self.rotation)
            self.velocity = [
                math.sin(angle_rad) * self.speed,
                -math.cos(angle_rad) * self.speed,
            ]

    def update_physics(self):
        self.update_heading()
        super().update_physics()
        self.apply_speed_limit()

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.previous_heading = getattr(self, "heading", 0)
        self.update_physics()
        self.expiration_timer -= 1

        # Handle frame animation if projectile evolves
        if self.frames > 1:
            self.frame_timer -= 1
            if self.frame_timer <= 0:
                if self.current_frame < self.frames - 1:
                    self.current_frame += 1
                    # Update properties for new frame
                    self.size = list(self.sizes[self.current_frame])
                    self.current_damage = self.damages[
                        min(self.current_frame, len(self.damages) - 1)
                    ]
                    if len(self.hp_array) > 1:
                        self.current_hp = min(
                            self.current_hp, self.hp_array[self.current_frame]
                        )
                self.frame_timer = self.frame_delay

        if self.current_hp <= 0:
            self.currently_alive = False
            return False
        if self.type == "laser":
            return self.expiration_timer >= 0 and self.current_hp > 0
        return self.expiration_timer > 0 and self.current_hp > 0

    def on_ship_impact(self, ship):
        """Apply projectile-specific behavior after damaging a ship."""
        return None

    def begin_planet_avoidance(self, planet, outward_normal):
        """React to special_object separation from a planet."""
        return None

    def handle_ship_contact(self, ship, normal=None):
        """Handle a special_object-specific ship collision instead of the default hit."""
        return False

    def handle_asteroid_contact(self, asteroid, normal=None):
        """Handle a special_object-specific asteroid collision instead of the default hit."""
        return False

    def handle_projectile_contact(self, projectile):
        """Handle damage from a projectile without the default special_object removal."""
        return False

    def should_collide_with_projectile_like(self, other):
        """Allow a specialized ability to veto projectile-like contacts."""
        return True

    def on_destroyed(self):
        """React once when collision handling destroys this ability."""
        return None

    def can_recover_with_parent(self):
        """Return whether this special_object may currently recover into its parent."""
        return False

    def recover_with_parent(self):
        """Apply special_object-specific recovery behavior."""
        return None

    def stop_and_track(self):
        """Apply any ability-specific round-transition tracking behavior."""
        return None

    def on_opponent_lost(self, opponent):
        """Let an ability react without imposing one special_object behavior."""
        if self.opponent is opponent:
            self.opponent = None

    def on_opponent_restored(self, opponent):
        """Reacquire an opponent that returned before the round ended."""
        if self.opponent is None:
            self.opponent = opponent

    def on_parent_removed(self):
        """Detach independent objects or silently remove ship-owned objects."""
        if self.survives_parent_cleanup:
            self.parent = None
            return
        self.current_hp = 0
        self.currently_alive = False

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

    def get_sprite(self, interp_t=0.0):
        if self.omnidirectional:
            if self.frames > 1:
                assets = self.resources.ability(self.name)
                video_multiplier = const.VIDEO_FPS_MULTIPLIER
                
                if video_multiplier > 1 and getattr(assets, "interpolated_sprites", None):
                    fraction = (self.frame_delay - self.frame_timer + interp_t) / self.frame_delay
                    sub_frame = min(
                        video_multiplier - 1,
                        max(0, int(fraction * video_multiplier)),
                    )
                    draw_idx = self.current_frame * video_multiplier + sub_frame
                    draw_idx = min(draw_idx, len(assets.interpolated_sprites[0]) - 1)
                    return assets.interpolated_sprites[0][draw_idx]
                return self.sprites[0][self.current_frame]
            return self.sprites[0]

        if self.frames > 1:
            from src.Battle.interpolation import interpolated_sprite_index

            return self.sprites[self.current_frame][
                interpolated_sprite_index(self, interp_t)
            ]
        from src.Battle.interpolation import interpolated_sprite_index

        return self.sprites[interpolated_sprite_index(self, interp_t)]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        sprite = self.get_sprite(interp_t)
        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (
                    -scaled_rect.width
                    <= pos_x
                    <= const.SCREEN_HEIGHT + scaled_rect.width
                    and -scaled_rect.height
                    <= pos_y
                    <= const.SCREEN_HEIGHT + scaled_rect.height
                ):
                    screen.blit(
                        scaled_sprite,
                        (
                            const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                            pos_y - scaled_rect.height // 2,
                        ),
                    )

    def get_collision_mask(self):
        if not self.masks:
            return None
        if self.omnidirectional:
            return self.masks[self.current_frame if self.frames > 1 else 0]
        if self.frames > 1:
            return self.masks[self.current_frame][
                const.heading_to_sprite_index(self.heading)
            ]
        return self.masks[const.heading_to_sprite_index(self.heading)]

    @staticmethod
    def draw_aa_laser(screen, color, start_pos, end_pos, width):
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return

        nx = dx / length
        ny = dy / length

        # Perpendicular vector
        px = -ny
        py = nx

        half_width = width / 2.0
        # Pygame rasterizes the filled polygon and AA edge lines separately.
        # Give the fill a quarter-pixel overlap under each edge so fractional beam
        # angles cannot leave isolated background pixels between them.
        fill_half_width = half_width + 0.25

        # Nominal edge points determine the beam's anti-aliased outline.
        p1 = (start_pos[0] + px * half_width, start_pos[1] + py * half_width)
        p2 = (start_pos[0] - px * half_width, start_pos[1] - py * half_width)

        # Rectangle end points, tangent to the round cap.
        p3 = (end_pos[0] - px * half_width, end_pos[1] - py * half_width)
        p4 = (end_pos[0] + px * half_width, end_pos[1] + py * half_width)

        fill_p1 = (
            start_pos[0] + px * fill_half_width,
            start_pos[1] + py * fill_half_width,
        )
        fill_p2 = (
            start_pos[0] - px * fill_half_width,
            start_pos[1] - py * fill_half_width,
        )
        fill_p3 = (
            end_pos[0] - px * fill_half_width,
            end_pos[1] - py * fill_half_width,
        )
        fill_p4 = (
            end_pos[0] + px * fill_half_width,
            end_pos[1] + py * fill_half_width,
        )

        # Draw the cap first so the body covers its inward half without leaving a seam.
        pygame.draw.aacircle(screen, color, end_pos, half_width)
        pygame.draw.polygon(screen, color, (fill_p1, fill_p2, fill_p3, fill_p4))

        # Full-color AA edges provide coverage blending without a dark alpha fringe.
        pygame.draw.aaline(screen, color, p1, p4)
        pygame.draw.aaline(screen, color, p2, p3)
