"""Frame-by-frame flight of one ship, as the battle simulation moves it.

The baseline rules that look ahead (the planet and incoming shots) fly
the ship forward with this model: one heading step every ``turn_wait + 1``
frames, one thrust increment every ``thrust_wait + 1`` frames with the
simulation's speed caps, the planet's pull inside its well, a Verlet step
and the speed limit. Positions and headings are in the frame of the ship at
the start: ``x`` forward, ``y`` to the right, bearings clockwise from the
starting nose. Ships without inertia feel no gravity and stop whenever
they do not thrust.
"""

from __future__ import annotations

import math

import src.const as const
from src.training.baseline.ship_specs import PLANET_GRAVITY, PLANET_RADIUS, ShipSpec


def planet_gravity(x: float, y: float, distance: float, pull: float) -> tuple[float, float]:
    """Pull toward the origin on a body at ``(x, y)``, ``distance`` from it."""

    if pull <= 0.0 or distance < PLANET_RADIUS or distance > const.GRAVITY_RANGE:
        return 0.0, 0.0
    return -pull * x / distance, -pull * y / distance


def thrusted_velocity(
    vx: float,
    vy: float,
    dx: float,
    dy: float,
    increment: float,
    max_speed: float,
    *,
    inside_well: bool,
) -> tuple[float, float]:
    """Velocity after one push of ``increment`` along ``(dx, dy)``, as
    ``apply_thrust`` caps it."""

    nx, ny = vx + dx * increment, vy + dy * increment
    speed = math.hypot(nx, ny)
    scale = 1.0
    if speed > max_speed and not inside_well:
        scale = max_speed / speed
    if speed > const.MAX_GRAV_WHIP:
        scale = const.MAX_GRAV_WHIP / speed
    change_x, change_y = nx * scale - vx, ny * scale - vy
    change = math.hypot(change_x, change_y)
    if change > increment > 0.0:
        change_x *= increment / change
        change_y *= increment / change
    return vx + change_x, vy + change_y


def turn_step_degrees() -> float:
    """Degrees one heading step turns the nose."""

    return float(const.TURN_ANGLE) * const.DIRECTIONS_MULTIPLIER


class ShipFlight:
    """One ship, stepped a frame at a time.

    ``planet`` is the planet's centre in the starting frame, or ``None``
    when it is too far to matter. With ``drifting`` a ship without inertia
    keeps its motion while it does not thrust rather than stopping, which
    projects the path it is already flying.
    """

    __slots__ = (
        "spec",
        "x",
        "y",
        "vx",
        "vy",
        "nose",
        "frame",
        "planet_distance",
        "_has_planet",
        "_planet_x",
        "_planet_y",
        "_pull",
        "_gx",
        "_gy",
        "_turn_step",
        "_turn_period",
        "_thrust_period",
        "_next_turn",
        "_next_thrust",
        "_drifting",
    )

    def __init__(
        self,
        spec: ShipSpec,
        position: tuple[float, float],
        velocity: tuple[float, float],
        *,
        planet: tuple[float, float] | None = None,
        drifting: bool = False,
    ):
        self.spec = spec
        self.x, self.y = float(position[0]), float(position[1])
        self.vx, self.vy = float(velocity[0]), float(velocity[1])
        self.nose = 0.0
        self.frame = 0
        self._has_planet = planet is not None
        if planet is None:
            self._planet_x, self._planet_y, self._pull = 0.0, 0.0, 0.0
        else:
            self._planet_x, self._planet_y = float(planet[0]), float(planet[1])
            self._pull = (
                const.GRAVITY_MULTIPLIER * PLANET_GRAVITY if spec.inertial else 0.0
            )
        self._turn_step = turn_step_degrees()
        self._turn_period = spec.turn_wait + 1
        self._thrust_period = spec.thrust_wait + 1
        self._next_turn = 0
        self._next_thrust = 0
        self._drifting = bool(drifting)
        self._update_gravity_source()
        self._gx, self._gy = self._gravity()

    @property
    def inside_well(self) -> bool:
        return self.planet_distance <= const.GRAVITY_RANGE

    def _update_gravity_source(self) -> None:
        self.planet_distance = (
            math.hypot(self.x - self._planet_x, self.y - self._planet_y)
            if self._has_planet
            else math.inf
        )

    def _gravity(self) -> tuple[float, float]:
        if self._pull <= 0.0:
            return 0.0, 0.0
        return planet_gravity(
            self.x - self._planet_x,
            self.y - self._planet_y,
            self.planet_distance,
            self._pull,
        )

    def steering_toward(self, bearing: float) -> float:
        """Degrees from the nose to ``bearing``, wrapped into (-180, 180]."""

        error = (float(bearing) - self.nose + 180.0) % 360.0 - 180.0
        return 180.0 if error == -180.0 else error

    def kick(self, angle: float, increment: float, max_speed: float) -> None:
        """An impulse ``angle`` degrees off the nose, capped as thrust is
        (the Druuge cannon's recoil)."""

        if not self.spec.inertial:
            return
        radians = math.radians(self.nose + float(angle))
        self.vx, self.vy = thrusted_velocity(
            self.vx,
            self.vy,
            math.cos(radians),
            math.sin(radians),
            float(increment),
            float(max_speed),
            inside_well=self.inside_well,
        )

    def step(self, turn_error: float | None, thrust: bool) -> None:
        """Advance one frame, turning toward ``turn_error`` degrees off the
        nose (none when ``None``) and thrusting when ``thrust``."""

        frame = self.frame
        spec = self.spec
        if (
            turn_error is not None
            and frame >= self._next_turn
            and abs(turn_error) > self._turn_step / 2.0
        ):
            nose = self.nose + math.copysign(self._turn_step, turn_error)
            self.nose = (nose + 180.0) % 360.0 - 180.0
            self._next_turn = frame + self._turn_period
        if thrust and frame >= self._next_thrust:
            self._next_thrust = frame + self._thrust_period
            radians = math.radians(self.nose)
            dx, dy = math.cos(radians), math.sin(radians)
            if spec.inertial:
                self.vx, self.vy = thrusted_velocity(
                    self.vx,
                    self.vy,
                    dx,
                    dy,
                    spec.thrust_increment,
                    spec.max_thrust,
                    inside_well=self.inside_well,
                )
            else:
                self.vx, self.vy = dx * spec.max_thrust, dy * spec.max_thrust
        elif not spec.inertial and not thrust and not self._drifting:
            self.vx, self.vy = 0.0, 0.0
        gx, gy = self._gx, self._gy
        self.x += self.vx + 0.5 * gx
        self.y += self.vy + 0.5 * gy
        self._update_gravity_source()
        nx, ny = self._gravity()
        self.vx += 0.5 * (gx + nx)
        self.vy += 0.5 * (gy + ny)
        self._gx, self._gy = nx, ny
        speed = math.hypot(self.vx, self.vy)
        if speed > const.SPEED_LIMIT:
            self.vx *= const.SPEED_LIMIT / speed
            self.vy *= const.SPEED_LIMIT / speed
        self.frame = frame + 1
