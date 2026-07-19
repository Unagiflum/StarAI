from dataclasses import replace
import math

import pygame

import src.const as const
from src.collision_capabilities import (
    ProjectileContactPolicy,
    SameTypeContactPolicy,
)
from src.Objects.object import ThrustMarker
from src.Objects.Ships.ability import Ability, SPECIAL_OBJECT_AREA_IMMUNITIES
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta
from src.training import event_ledger


class OrzA3(Ability):
    """Orz Marine that boards an enemy, fights, and returns as crew."""

    LAUNCHING = "launching"
    OUTBOUND = "outbound"
    BOARDED = "boarded"
    RETURNING = "returning"

    MAX_MARINES = (
        ABILITY_DEFINITIONS["OrzA3"].max_marines
        if ABILITY_DEFINITIONS["OrzA3"].max_marines is not None
        else 8
    )
    BOARDING_WAIT = const.cooldown_frames(12)
    DEATH_ROLL_LIMIT = 16
    KILL_ROLL_LIMIT = 144
    SHIELD_BOUNCE_FRAMES = 5
    COLLIDING_SPECIAL_OBJECTS = {
        "ChenjesuA2",
        "KzerZaA2",
        "SyreenCrew",
        "VuxA2",
    }

    def __init__(self, parent):
        super().__init__("OrzA3", parent)
        self.expiration_timer = float("inf")
        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            immune_to_sources=SPECIAL_OBJECT_AREA_IMMUNITIES,
        )
        self.special_object_collision_capabilities = replace(
            self.special_object_collision_capabilities,
            collides_with_fighters=True,
            destroys_fragile=True,
            projectile_contact_policy=(
                ProjectileContactPolicy.TAKE_DAMAGE_AND_DESTROY_PROJECTILE
            ),
            same_type_contact_policy=SameTypeContactPolicy.IGNORE,
        )

        ability_def = ABILITY_DEFINITIONS["OrzA3"]
        self.launch_time = (
            ability_def.launch_time if ability_def.launch_time is not None else 0
        )
        self.mode = self.LAUNCHING if self.launch_time > 0 else self.OUTBOUND
        self.launch_timer = self.launch_time

        self.boarded_ship = None
        self.boarding_timer = self.BOARDING_WAIT
        self.shield_bounce_timer = 0

        ability_def = ABILITY_DEFINITIONS["OrzA3"]
        self.spiral_distance = (
            ability_def.spiral_distance
            if ability_def.spiral_distance is not None
            else 500
        )
        self.max_thrust = (
            ability_def.max_thrust if ability_def.max_thrust else self.speed
        )
        self.thrust_increment = (
            ability_def.thrust_increment if ability_def.thrust_increment else 8
        )
        self.thrust_wait = ability_def.thrust_wait if ability_def.thrust_wait else 1
        self.look_ahead = (
            ability_def.look_ahead if ability_def.look_ahead is not None else 15
        )
        self.thrust_timer = 0
        self.steering_tie_direction = self.rng.choice((-1, 1))

        self.spawned_objects = []
        self._death_sound_played = False
        self._crew_loss_recorded = False
        self._crew_recovered = False
        self._load_marine_sounds()
        self._load_flight_sprites()
        self._place_at_parent_rear()

    @property
    def is_boarded(self):
        return self.mode == self.BOARDED

    @property
    def is_returning(self):
        return self.mode == self.RETURNING

    @property
    def hud_sprite(self):
        return self.red_flight_sprite

    def get_sprite(self, interp_t=0.0):
        if self.is_returning:
            return self.green_flight_sprite
        return self.red_flight_sprite

    def get_collision_mask(self):
        if self.is_returning:
            return self.green_flight_mask
        return self.red_flight_mask

    def _load_marine_sounds(self):
        directory = const.source_path("Objects/Ships/Orz/A3")
        self.launch_sound = self.audio_service.load_effect(
            directory / "OrzA3Launch.wav", const.SOUND_EFFECT_VOLUME
        )
        self.alarm_sound = self.audio_service.load_effect(
            directory / "OrzA3Alarm.wav", const.SOUND_EFFECT_VOLUME
        )
        self.die_sound = self.audio_service.load_effect(
            directory / "OrzA3Die.wav", const.SOUND_EFFECT_VOLUME
        )
        self.zap_sound = self.audio_service.load_effect(
            directory / "OrzA3Zap.wav", const.SOUND_EFFECT_VOLUME
        )

    def _load_flight_sprites(self):
        directory = const.source_path("Objects/Ships/Orz/A3")
        self.red_flight_sprite = self.resources.image(directory / "OrzA3Red.png").image
        self.green_flight_sprite = self.resources.image(
            directory / "OrzA3Green.png"
        ).image
        self.red_flight_mask = pygame.mask.from_surface(self.red_flight_sprite)
        self.green_flight_mask = pygame.mask.from_surface(self.green_flight_sprite)
        self.size = list(self.red_flight_sprite.get_size())

    def _place_at_parent_rear(self):
        self.launch_from_gun(inherit_parent_velocity=False)
        # UQM allocates the marine as a zeroed element and does not assign a
        # launch velocity. Its tracking preprocess starts on the next update.
        self.velocity = [0.0, 0.0]

    def update(self):
        if not self.currently_alive:
            return False
        if self.current_hp <= 0:
            self._die()
            return False

        self.previous_position = self.position.copy()
        if self.mode == self.BOARDED:
            return self._update_boarded()

        if self.mode == self.LAUNCHING:
            self.update_physics()
            self.launch_timer -= 1
            if self.launch_timer <= 0:
                self.mode = self.OUTBOUND
            return self.currently_alive

        if self.shield_bounce_timer > 0:
            self.shield_bounce_timer -= 1
            self.update_physics()
            return True

        destination = self._flight_destination()

        avoid = False
        target = self._live_trackable_opponent() if self.mode == self.OUTBOUND else None

        margin = self.size[1]
        t_planet = self.predict_planet_collision(frames=90, margin=margin)

        if t_planet is not None:
            t_target = None
            if target:
                t_target = self.predict_target_interception(target, frames=90)

            if t_target is None or t_planet < t_target:
                avoid = True

        if avoid:
            self._apply_avoidance_thrust()
        elif destination is not None:
            self._move_toward(destination)
        else:
            self.velocity = [0.0, 0.0]

        self.update_physics()

        return self.currently_alive

    def _update_boarded(self):
        ship = self.boarded_ship
        if ship is None or not ship.currently_alive or ship.current_hp <= 0:
            self._leave_ship()
            return True

        self.position = ship.position.copy()
        self.velocity = [0.0, 0.0]
        if getattr(ship, "ability_actions_paused", False):
            return True
        self.boarding_timer -= 1
        if self.boarding_timer > 0:
            return True

        self.boarding_timer = self.BOARDING_WAIT
        roll = self.rng.randrange(256)
        if roll < self.DEATH_ROLL_LIMIT:
            self._die()
            return False
        if roll < self.KILL_ROLL_LIMIT:
            damage = ship.take_damage(1, shieldable=False, source=self)
            if damage and self.zap_sound:
                self.zap_sound.play()
            if ship.current_hp <= 0:
                self._leave_ship()
        return self.currently_alive

    def _flight_destination(self):
        if self.mode == self.OUTBOUND:
            target = self._live_trackable_opponent()
            if target is not None:
                intercept_frame = self.predict_target_interception(
                    target, frames=self.look_ahead
                )
                t_traj = target.predict_unhindered_trajectory(frames=self.look_ahead)

                if intercept_frame is None or intercept_frame >= self.look_ahead:
                    if len(t_traj) > 0:
                        return t_traj[-1]
                    return target.position
                else:
                    if intercept_frame < len(t_traj):
                        return t_traj[intercept_frame]
                    elif len(t_traj) > 0:
                        return t_traj[-1]
                    return target.position
            self._begin_return()
        return self.parent.position if self._parent_alive() else None

    def _move_toward(self, destination):
        dx, dy = wrapped_delta(self.position, destination)
        distance = math.hypot(dx, dy)
        if distance <= 0:
            return

        desired_angle = math.degrees(math.atan2(dx / distance, -dy / distance)) % 360
        angle_step = 360 / const.SHIP_DIRECTIONS
        self.rotation = round(desired_angle / angle_step) * angle_step % 360

        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed > 0:
            current_heading = (
                math.degrees(math.atan2(self.velocity[0], -self.velocity[1])) % 360
            )
        else:
            current_heading = self.rotation

        if speed > 0 and distance <= self.spiral_distance:
            min_diff_to_90 = min(
                abs(abs((k * angle_step - current_heading + 180) % 360 - 180) - 90)
                for k in range(const.SHIP_DIRECTIONS)
            )
            current_diff_to_90 = abs(
                abs((self.rotation - current_heading + 180) % 360 - 180) - 90
            )

            if math.isclose(current_diff_to_90, min_diff_to_90, abs_tol=1e-5):
                anti_heading = (current_heading + 180) % 360
                adj1 = (self.rotation + angle_step) % 360
                adj2 = (self.rotation - angle_step) % 360

                diff1 = abs((adj1 - anti_heading + 180) % 360 - 180)
                diff2 = abs((adj2 - anti_heading + 180) % 360 - 180)

                if math.isclose(diff1, diff2):
                    self.rotation = (
                        adj1 if self.steering_tie_direction > 0 else adj2
                    )
                elif diff1 < diff2:
                    self.rotation = adj1
                else:
                    self.rotation = adj2

        in_gravity = False
        if self.planet:
            p_dx, p_dy = wrapped_delta(self.position, self.planet.position)
            in_gravity = math.hypot(p_dx, p_dy) < const.GRAVITY_RANGE

        top_speed = const.SPEED_LIMIT if in_gravity else self.max_thrust
        angle_diff = abs((current_heading - desired_angle + 180) % 360 - 180)
        threshold = 180 / const.SHIP_DIRECTIONS

        if angle_diff <= threshold and speed >= top_speed - 0.001:
            return

        if self.thrust_timer <= 0:
            marker = self.apply_thrust(self.max_thrust, self.thrust_increment, 0, True)
            if marker:
                self.spawned_objects.append(marker)
            self.thrust_timer = self.thrust_wait
        else:
            self.thrust_timer -= 1

    def predict_target_interception(self, target, frames=90):
        m_pos = list(self.position)
        m_vel = list(self.velocity)
        t_traj = target.predict_unhindered_trajectory(frames=frames)

        for f in range(frames):
            if f >= len(t_traj):
                break
            t_pos = t_traj[f]
            dx, dy = wrapped_delta(m_pos, t_pos)
            dist = math.hypot(dx, dy)
            if dist < (self.size[0] + target.size[0]) / 2:
                return f

            if dist > 0:
                dir_x = dx / dist
                dir_y = dy / dist
                m_vel[0] += dir_x * self.thrust_increment
                m_vel[1] += dir_y * self.thrust_increment

                speed = math.hypot(m_vel[0], m_vel[1])
                if speed > self.max_thrust:
                    m_vel[0] = m_vel[0] * self.max_thrust / speed
                    m_vel[1] = m_vel[1] * self.max_thrust / speed

            if self.planet:
                px, py = wrapped_delta(m_pos, self.planet.position)
                p_dist = math.hypot(px, py)
                if p_dist < const.GRAVITY_RANGE and p_dist > self.planet.diameter / 2:
                    gf = const.GRAVITY_MULTIPLIER * self.planet.gravity
                    if p_dist > 0:
                        m_vel[0] += gf * px / p_dist
                        m_vel[1] += gf * py / p_dist

            speed = math.hypot(m_vel[0], m_vel[1])
            if speed > const.SPEED_LIMIT:
                m_vel[0] = m_vel[0] * const.SPEED_LIMIT / speed
                m_vel[1] = m_vel[1] * const.SPEED_LIMIT / speed

            m_pos[0] = (m_pos[0] + m_vel[0] * const.SPEED_SCALE) % const.ARENA_SIZE
            m_pos[1] = (m_pos[1] + m_vel[1] * const.SPEED_SCALE) % const.ARENA_SIZE

        return None

    def _apply_avoidance_thrust(self):
        if not self.planet:
            return
        dx, dy = wrapped_delta(self.position, self.planet.position)
        dist = math.hypot(dx, dy)
        if dist > 0:
            speed = math.hypot(self.velocity[0], self.velocity[1])
            if speed < self.thrust_increment / 2:
                thrust_dir_x = -dx / dist
                thrust_dir_y = -dy / dist
            else:
                vx, vy = self.velocity
                dot_product = -vy * dx + vx * dy
                if math.isclose(dot_product, 0.0, abs_tol=1e-9):
                    turn_first_way = self.steering_tie_direction < 0
                else:
                    turn_first_way = dot_product < 0
                if turn_first_way:
                    thrust_dir_x = -vy / speed
                    thrust_dir_y = vx / speed
                else:
                    thrust_dir_x = vy / speed
                    thrust_dir_y = -vx / speed

            desired_angle = math.degrees(math.atan2(thrust_dir_x, -thrust_dir_y)) % 360
            angle_step = 360 / const.SHIP_DIRECTIONS
            self.rotation = round(desired_angle / angle_step) * angle_step % 360
            if self.thrust_timer <= 0:
                marker = self.apply_thrust(
                    self.max_thrust, self.thrust_increment, 0, True
                )
                if marker:
                    self.spawned_objects.append(marker)
                self.thrust_timer = self.thrust_wait
            else:
                self.thrust_timer -= 1

    def _coast(self):
        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE

    def _spawn_thrust_marker(self):
        angle = math.radians(self.rotation)
        offset = self.size[1] / 2 + 6
        self.spawned_objects.append(
            ThrustMarker(
                self.position[0] - math.sin(angle) * offset,
                self.position[1] + math.cos(angle) * offset,
            )
        )

    def drain_spawned_objects(self):
        result = self.spawned_objects
        self.spawned_objects = []
        return result

    def should_collide_with_ship(self, ship):
        if self.mode in (self.OUTBOUND, self.LAUNCHING):
            return ship is self.opponent
        if self.mode == self.RETURNING:
            return ship is self.parent
        return False

    def should_collide_with_projectile_like(self, other):
        if getattr(other, "type", None) == "special_object":
            return getattr(other, "name", None) in self.COLLIDING_SPECIAL_OBJECTS
        return True

    def handle_ship_contact(self, ship, normal=None):
        if (
            self.mode not in (self.OUTBOUND, self.LAUNCHING)
            or ship.player == self.player
            or ship.current_hp <= 0
        ):
            return False

        if ship.damage_shield_is_active():
            if normal:
                dot = self.velocity[0] * normal[0] + self.velocity[1] * normal[1]
                self.velocity[0] -= 2 * dot * normal[0]
                self.velocity[1] -= 2 * dot * normal[1]
            else:
                self.velocity[0] *= -1
                self.velocity[1] *= -1
            self.position = self.previous_position.copy()
            self.shield_bounce_timer = self.SHIELD_BOUNCE_FRAMES
            return True

        ship.take_damage(1, shieldable=False, source=self)
        if ship.current_hp <= 0:
            self._begin_return()
            return True

        self.mode = self.BOARDED
        self.boarded_ship = ship
        self.target = ship
        self.position = ship.position.copy()
        self.velocity = [0.0, 0.0]
        self.can_collide = False
        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            targetable=False,
        )
        if self not in ship.boarded_marines:
            ship.boarded_marines.append(self)
            event_ledger.record_debuff_applied(
                ship,
                event_ledger.DEBUFF_BOARDING_MARINE,
                actor=self.parent,
                source=self,
            )
        if self.alarm_sound:
            self.alarm_sound.play()
        return True

    def handle_asteroid_contact(self, asteroid, normal=None):
        if normal:
            dot = self.velocity[0] * normal[0] + self.velocity[1] * normal[1]
            self.velocity[0] -= 2 * dot * normal[0]
            self.velocity[1] -= 2 * dot * normal[1]
        else:
            self.velocity[0] *= -1
            self.velocity[1] *= -1
        self.position = self.previous_position.copy()
        self.shield_bounce_timer = self.SHIELD_BOUNCE_FRAMES
        return True

    def handle_projectile_contact(self, projectile):
        if getattr(projectile, "name", None) == "ChenjesuA2":
            return True
        if getattr(projectile, "name", None) == "KzerZaA2":
            projectile.set_hp(0)
            return True
        self.current_hp = max(0, self.current_hp - projectile.current_damage)
        if hasattr(projectile, "set_hp"):
            projectile.set_hp(0)
        else:
            projectile.current_hp = 0
        return True



    def can_recover_with_parent(self):
        return self.mode == self.RETURNING

    def recover_with_parent(self):
        if not self.currently_alive or not self._parent_alive():
            return
        self.parent.current_hp = min(self.parent.max_hp, self.parent.current_hp + 1)
        self._crew_recovered = True
        self._detach_from_ship()
        self.current_hp = 0
        self.currently_alive = False

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        if not self.is_boarded:
            super().draw(screen, scale_factor, translation, interp_t=interp_t)

    def on_opponent_lost(self, opponent):
        super().on_opponent_lost(opponent)
        if self.mode != self.BOARDED:
            self._begin_return()

    def on_host_self_destruct(self):
        self._die()

    def on_destroyed(self):
        self._record_crew_loss()
        self._detach_from_ship()
        self._play_death_sound()

    def _leave_ship(self):
        self._detach_from_ship()
        self._begin_return()
        self.can_collide = True
        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            targetable=True,
        )
        self.current_hp = max(1, self.current_hp)

    def _begin_return(self):
        self.mode = self.RETURNING

    def _detach_from_ship(self):
        if self.boarded_ship is not None:
            try:
                self.boarded_ship.boarded_marines.remove(self)
            except ValueError:
                pass
        self.boarded_ship = None
        self.target = None

    def _die(self):
        if not self.currently_alive:
            return
        self._detach_from_ship()
        self._record_crew_loss()
        self.current_hp = 0
        self.currently_alive = False
        self._play_death_sound()

    def _play_death_sound(self):
        if not self._death_sound_played and self.die_sound:
            self.die_sound.play()
        self._death_sound_played = True

    def _record_crew_loss(self, *, actor=None, source=None):
        if self._crew_loss_recorded or self._crew_recovered:
            return
        self._crew_loss_recorded = True
        destroying_source = source or getattr(self, "_training_destroying_source", None)
        event_ledger.record_launched_crew_lost(
            self,
            actor=actor,
            source=destroying_source if destroying_source is not None else self,
        )

    def _parent_alive(self):
        return self.parent.currently_alive and self.parent.current_hp > 0
