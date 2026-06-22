import math

import src.const as const
from src.collision_capabilities import AreaDamageCapabilities
from src.Objects.Ships.ability import Ability
from src.toroidal import wrapped_delta


class OrzA3(Ability):
    """Orz Marine that boards an enemy, fights, and returns as crew."""

    OUTBOUND = "outbound"
    BOARDED = "boarded"
    RETURNING = "returning"

    MAX_MARINES = 8
    BOARDING_WAIT = int(12 * const.ACTION_WAIT_SCALE)
    DEATH_ROLL_LIMIT = 16
    KILL_ROLL_LIMIT = 144

    def __init__(self, parent):
        super().__init__("OrzA3", parent)
        self.expiration_timer = float("inf")
        self.mode = self.OUTBOUND
        self.boarded_ship = None
        self.boarding_timer = self.BOARDING_WAIT
        self._death_sound_played = False
        self._load_marine_sounds()
        self.hud_sprite = self.resources.image(
            const.source_path("Objects/Ships/Orz/A3/OrzA3.png"),
            size=(12, 12),
        ).image
        self._place_at_parent_rear()

    @property
    def is_boarded(self):
        return self.mode == self.BOARDED

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

    def _place_at_parent_rear(self):
        self.rotation = (self.parent.rotation + 180) % 360
        angle = math.radians(self.rotation)
        distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2
        self.position = [
            (
                self.parent.position[0] + math.sin(angle) * distance
            ) % const.ARENA_SIZE,
            (
                self.parent.position[1] - math.cos(angle) * distance
            ) % const.ARENA_SIZE,
        ]
        self.previous_position = self.position.copy()
        self._set_velocity_toward_angle(self.rotation)

    def update(self):
        if not self.currently_alive:
            return False
        if not self._parent_alive():
            self._die()
            return False
        if self.current_hp <= 0:
            self._die()
            return False

        self.previous_position = self.position.copy()
        if self.mode == self.BOARDED:
            return self._update_boarded()

        destination = self._flight_destination()
        if destination is not None:
            self._move_toward(destination)
        else:
            self.velocity = [0.0, 0.0]
        return self.currently_alive

    def _update_boarded(self):
        ship = self.boarded_ship
        if ship is None or not ship.currently_alive or ship.current_hp <= 0:
            self._leave_ship()
            return True

        self.position = ship.position.copy()
        self.velocity = [0.0, 0.0]
        self.boarding_timer -= 1
        if self.boarding_timer > 0:
            return True

        self.boarding_timer = self.BOARDING_WAIT
        roll = self.rng.randrange(256)
        if roll < self.DEATH_ROLL_LIMIT:
            self._die()
            return False
        if roll < self.KILL_ROLL_LIMIT:
            ship.take_damage(1, shieldable=False)
            if ship.current_hp <= 0:
                self._leave_ship()
        return self.currently_alive

    def _flight_destination(self):
        if self.mode == self.OUTBOUND:
            target = self._live_trackable_opponent()
            if target is not None:
                return target.position
            self.mode = self.RETURNING
        return self.parent.position if self._parent_alive() else None

    def _move_toward(self, destination):
        dx, dy = wrapped_delta(self.position, destination)
        distance = math.hypot(dx, dy)
        if distance <= 0:
            self.velocity = [0.0, 0.0]
            return
        speed = min(self.speed, distance / const.SPEED_SCALE)
        self.velocity = [dx / distance * speed, dy / distance * speed]
        self.rotation = math.degrees(math.atan2(dx, -dy)) % 360
        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE

    def _set_velocity_toward_angle(self, angle_degrees):
        angle = math.radians(angle_degrees)
        self.velocity = [math.sin(angle) * self.speed, -math.cos(angle) * self.speed]

    def handle_ship_contact(self, ship):
        if (
            self.mode != self.OUTBOUND
            or ship.player == self.player
            or ship.current_hp <= 0
        ):
            return False

        ship.take_damage(1, shieldable=False)
        if ship.current_hp <= 0:
            self.mode = self.RETURNING
            return True

        self.mode = self.BOARDED
        self.boarded_ship = ship
        self.target = ship
        self.position = ship.position.copy()
        self.velocity = [0.0, 0.0]
        self.can_collide = False
        self.area_damage_capabilities = AreaDamageCapabilities(
            targetable=False
        )
        if self not in ship.boarded_marines:
            ship.boarded_marines.append(self)
        if self.alarm_sound:
            self.alarm_sound.play()
        return True

    def handle_projectile_contact(self, projectile):
        self.current_hp = max(0, self.current_hp - projectile.current_damage)
        return True

    def begin_planet_avoidance(self, planet, outward_normal):
        self._die()

    def can_recover_with_parent(self):
        return self.mode == self.RETURNING

    def recover_with_parent(self):
        if not self.currently_alive or not self._parent_alive():
            return
        self.parent.current_hp = min(self.parent.max_hp, self.parent.current_hp + 1)
        self._detach_from_ship()
        self.current_hp = 0
        self.currently_alive = False

    def draw(self, screen, scale_factor, translation):
        if not self.is_boarded:
            super().draw(screen, scale_factor, translation)

    def on_opponent_lost(self, opponent):
        super().on_opponent_lost(opponent)
        if self.mode != self.BOARDED:
            self.mode = self.RETURNING

    def on_destroyed(self):
        self._detach_from_ship()
        self._play_death_sound()

    def _leave_ship(self):
        self._detach_from_ship()
        self.mode = self.RETURNING
        self.can_collide = True
        self.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        self.current_hp = max(1, self.current_hp)

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
        self.current_hp = 0
        self.currently_alive = False
        self._play_death_sound()

    def _play_death_sound(self):
        if not self._death_sound_played and self.die_sound:
            self.die_sound.play()
        self._death_sound_played = True

    def _parent_alive(self):
        return self.parent.currently_alive and self.parent.current_hp > 0
