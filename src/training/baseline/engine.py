"""Rule engine shared by every baseline strategy.

A strategy is an ordered list of rules. Each rule may set some of the four
controls: turning, thrust, weapon 1 and weapon 2. The first rule to set a
control claims it, and later rules cannot change a claimed control. A rule
whose condition is false claims nothing. Controls nobody claims stay off.
"""

from __future__ import annotations

import math
from typing import Callable

import src.const as const
from src.training.baseline.observation_view import (
    ObservationView,
    ObservedObject,
    bearing_of,
    wrap_degrees,
)
from src.training.baseline.ship_specs import (
    COLLISION_MARGIN,
    FORM_BASE,
    FORM_BLAZER,
    FORM_XFORM,
    FORM_YWING,
    PLANET_RADIUS,
    ShipSpec,
    WeaponSpec,
    ship_spec,
    weapon_spec,
)
from src.training.contracts import ACTION_INDEX_TABLE

ACTION_INDEX_BY_MASK = {action.mask: index for index, action in enumerate(ACTION_INDEX_TABLE)}

TURN_LEFT = -1
TURN_NONE = 0
TURN_RIGHT = 1

# Heading offsets used by the shared behaviors, in degrees.
APPROACH_ANGLE_OFFSET = 45.0
AVOID_ANGLE_OFFSET = 135.0
PERPENDICULAR_AWAY_OFFSET = 110.0
PLANET_SIDESTEP_OFFSET = 45.0
# Fraction of a range below which [Approach_Range] backs off.
APPROACH_RANGE_INNER_FRACTION = 0.7
PLANET_COLLISION_SECONDS = 2.0
PLANET_COURSE_LATERAL_FRACTION = 0.2
PLANET_AVOID_LOOKAHEAD = 1.5
# Sideways drift, as a fraction of top speed, that counts as orbiting.
ORBIT_DRIFT_THRESHOLD = 0.15


class ControlPlan:
    """First-claim-wins control values for one decision."""

    __slots__ = ("turn", "thrust", "a1", "a2")

    def __init__(self) -> None:
        self.turn: int | None = None
        self.thrust: bool | None = None
        self.a1: bool | None = None
        self.a2: bool | None = None

    def claim_turn(self, value: int) -> bool:
        if self.turn is not None:
            return False
        self.turn = int(value)
        return True

    def claim_thrust(self, value: bool) -> bool:
        if self.thrust is not None:
            return False
        self.thrust = bool(value)
        return True

    def claim_a1(self, value: bool) -> bool:
        if self.a1 is not None:
            return False
        self.a1 = bool(value)
        return True

    def claim_a2(self, value: bool) -> bool:
        if self.a2 is not None:
            return False
        self.a2 = bool(value)
        return True

    @property
    def steering_claimed(self) -> bool:
        return self.turn is not None

    def mask(self) -> int:
        mask = 0
        if self.thrust:
            mask |= 1
        if self.turn == TURN_LEFT:
            mask |= 2
        elif self.turn == TURN_RIGHT:
            mask |= 4
        if self.a1:
            mask |= 8
        if self.a2:
            mask |= 16
        return mask

    def action_index(self) -> int:
        return ACTION_INDEX_BY_MASK[self.mask()]


class BaselineMemory:
    """State a strategy keeps between decisions within one episode."""

    def __init__(self) -> None:
        turn_angle = float(const.TURN_ANGLE)
        # Degrees one decision turns: a fresh press, and a continued hold.
        # Both start as estimates and are refined from observed headings.
        self.tap_degrees = max(turn_angle, 15.0)
        self.hold_degrees = 3.0 * self.tap_degrees
        self.reset_episode()

    def reset_episode(self) -> None:
        self.previous_heading: float | None = None
        self.previous_turn_state = TURN_NONE
        self.previous_crew: float | None = None
        self.sides: dict[str, int] = {}
        self.touched_sides: set[str] = set()
        self.values: dict[str, object] = {}

    def begin_decision(self, view: ObservationView) -> None:
        turn_state = _held_turn_state(view)
        heading = view.me.heading
        if self.previous_heading is not None and turn_state != TURN_NONE:
            turned = abs(wrap_degrees(heading - self.previous_heading))
            if turned > 0.0:
                if turn_state != self.previous_turn_state:
                    self.tap_degrees = 0.5 * self.tap_degrees + 0.5 * turned
                else:
                    self.hold_degrees = 0.5 * self.hold_degrees + 0.5 * turned
        self.previous_heading = heading
        self.previous_turn_state = turn_state
        self.touched_sides = set()

    def end_decision(self, view: ObservationView) -> None:
        for key in tuple(self.sides):
            if key not in self.touched_sides:
                del self.sides[key]
        self.previous_crew = view.me.crew


def _held_turn_state(view: ObservationView) -> int:
    if view.me.held("left"):
        return TURN_LEFT
    if view.me.held("right"):
        return TURN_RIGHT
    return TURN_NONE


class BaselineContext:
    """One decision's view, plan and shared behaviors for a strategy."""

    def __init__(
        self,
        *,
        ship_name: str,
        view: ObservationView,
        memory: BaselineMemory,
        action_frame_span: int,
    ):
        self.ship_name = ship_name
        self.view = view
        self.memory = memory
        self.plan = ControlPlan()
        self.span_frames = max(1, int(action_frame_span))

    # ----------------------------------------------------------------- facts

    @property
    def me(self):
        return self.view.me

    @property
    def enemy(self) -> ObservedObject | None:
        return self.view.enemy

    @property
    def enemy_block(self):
        return self.view.enemy_block

    @property
    def enemy_name(self) -> str | None:
        return self.view.enemy_ship_type

    @property
    def planet(self) -> ObservedObject | None:
        return self.view.planet

    @property
    def distance(self) -> float:
        enemy = self.enemy
        return enemy.distance if enemy is not None else math.inf

    @property
    def span_seconds(self) -> float:
        return self.span_frames / float(const.FPS)

    @property
    def form(self) -> str:
        if self.ship_name == "Androsynth" and self.me.androsynth_blazer:
            return FORM_BLAZER
        if self.ship_name == "Mmrnmrhm":
            return FORM_YWING if self.me.mmrnmrhm_ywing else FORM_XFORM
        return FORM_BASE

    @property
    def spec(self) -> ShipSpec:
        return ship_spec(self.ship_name, self.form)

    def weapon(self, action: int) -> WeaponSpec:
        return weapon_spec(self.ship_name, int(action), self.form)

    def battery_at_least(self, fraction: float) -> bool:
        return self.me.battery_fraction >= float(fraction)

    def held(self, control: str) -> bool:
        return self.me.held(control)

    def enemy_cloaked(self) -> bool:
        return not self.enemy_block.trackable

    def imminent(self, obj: ObservedObject, seconds: float) -> bool:
        """Whether ``obj`` will touch me within ``seconds`` plus one decision."""

        if obj.distance <= COLLISION_MARGIN:
            return True
        frames, miss = obj.closest_approach()
        window = float(seconds) * const.FPS + self.span_frames
        return 0.0 <= frames <= window and miss <= COLLISION_MARGIN

    def threats(self) -> tuple[ObservedObject, ...]:
        """Enemy objects, plus the enemy hull when it is a blazer."""

        threats = list(self.view.enemy_objects)
        enemy = self.enemy
        if enemy is not None and self.enemy_block.androsynth_blazer:
            threats.append(enemy)
        return tuple(threats)

    def imminent_threat(self, seconds: float) -> ObservedObject | None:
        for threat in self.threats():
            if self.imminent(threat, seconds):
                return threat
        return None

    def crew_dropped(self) -> bool:
        previous = self.memory.previous_crew
        return previous is not None and self.me.crew < previous - 1e-6

    # ---------------------------------------------------------------- aiming

    def aim_bearing(self, action: int) -> float | None:
        """Bearing a shot from ``action`` must travel to meet the enemy."""

        enemy = self.enemy
        if enemy is None:
            return None
        weapon = self.weapon(action)
        if not weapon.leads_target:
            return enemy.bearing
        own_forward = self.me.velocity_forward
        own_right = self.me.velocity_right
        carry = 1.0 - weapon.parent_velocity
        drift_forward = enemy.velocity_forward + carry * own_forward
        drift_right = enemy.velocity_right + carry * own_right
        speed = weapon.speed
        a = drift_forward * drift_forward + drift_right * drift_right - speed * speed
        b = 2.0 * (enemy.forward * drift_forward + enemy.right * drift_right)
        c = enemy.forward * enemy.forward + enemy.right * enemy.right
        frames = _smallest_positive_root(a, b, c)
        if frames is None:
            return enemy.bearing
        return bearing_of(
            enemy.forward + drift_forward * frames,
            enemy.right + drift_right * frames,
        )

    def _gun_offsets(self, action: int) -> tuple[float, ...]:
        if self.ship_name == "Orz" and action == 1:
            return (self.me.orz_turret_bearing,)
        return self.weapon(action).gun_directions

    def aim_error(self, action: int) -> float | None:
        aim = self.aim_bearing(action)
        if aim is None:
            return None
        return min(
            (wrap_degrees(aim - offset) for offset in self._gun_offsets(action)),
            key=abs,
        )

    def pointing(self, action: int, tolerance: float | None = None) -> bool:
        weapon = self.weapon(action)
        if weapon.auto_aims:
            return self.enemy is not None
        error = self.aim_error(action)
        if error is None:
            return False
        if tolerance is None:
            tolerance = max(weapon.pointing_tolerance, self.memory.tap_degrees / 2.0)
        return abs(error) <= float(tolerance)

    def in_range(self, action: int, scale: float = 1.0) -> bool:
        weapon = self.weapon(action)
        return self.enemy is not None and self.distance <= weapon.range * float(scale)

    # -------------------------------------------------------------- steering

    def turn_toward(self, bearing: float) -> int:
        """Turn command that brings the nose to ``bearing`` without overshoot."""

        bearing = wrap_degrees(bearing)
        tolerance = max(float(const.TURN_ANGLE) / 2.0, self.memory.tap_degrees / 2.0)
        if abs(bearing) <= tolerance:
            return TURN_NONE
        direction = TURN_RIGHT if bearing > 0 else TURN_LEFT
        held_direction = _held_turn_state(self.view)
        if held_direction == direction and abs(bearing) < 0.75 * self.memory.hold_degrees:
            # Release now so the next decision is a short, fresh tap.
            return TURN_NONE
        return direction

    def point_at(self, bearing: float | None) -> bool:
        if bearing is None:
            return False
        return self.plan.claim_turn(self.turn_toward(bearing))

    def steer(self, bearing: float | None, thrust: bool = True) -> bool:
        """Claim turning and thrust to head toward ``bearing``."""

        if bearing is None or self.plan.steering_claimed:
            return False
        self.plan.claim_turn(self.turn_toward(bearing))
        if self.ship_name == "Slylandro":
            # Slylandro always moves forward; a fresh thrust press reverses
            # it, which beats turning around when the target is behind.
            thrust = bool(thrust) and abs(wrap_degrees(bearing)) > 135.0
            thrust = thrust and not self.held("thrust")
        self.plan.claim_thrust(bool(thrust))
        return True

    def side(self, key: str, base: float, offset: float) -> float:
        """``base`` plus or minus ``offset``, keeping the side once chosen."""

        self.memory.touched_sides.add(key)
        sign = self.memory.sides.get(key)
        if sign is None:
            plus = abs(wrap_degrees(base + offset))
            minus = abs(wrap_degrees(base - offset))
            sign = 1 if plus <= minus else -1
            self.memory.sides[key] = sign
        return wrap_degrees(base + sign * offset)

    def drift_cancelling_side(
        self,
        key: str,
        target: ObservedObject,
        offset: float,
    ) -> float:
        """Approach ``target`` at ``offset`` on the side that stops orbiting.

        Thrust toward the clockwise side (bearing + offset) lowers the target's
        clockwise drift across my view, so pick the side matching that drift.
        With little drift, keep the side that needs the least turning.
        """

        drift = target.lateral_speed
        if abs(drift) > ORBIT_DRIFT_THRESHOLD * max(1.0, self.spec.max_thrust):
            self.memory.touched_sides.add(key)
            self.memory.sides[key] = 1 if drift > 0.0 else -1
            return wrap_degrees(target.bearing + (offset if drift > 0.0 else -offset))
        return self.side(key, target.bearing, offset)

    def point_weapon(self, action: int) -> bool:
        error = self.aim_error(action)
        if error is None:
            return False
        return self.point_at(error)

    def point_at_opponent(self) -> bool:
        enemy = self.enemy
        return self.point_at(enemy.bearing if enemy is not None else None)

    def press(self, control: str) -> bool:
        """True only when ``control`` was released, so this is a fresh press."""

        return not self.held(control)

    # -------------------------------------------------------- shared behaviors

    def avoid_planet(self) -> bool:
        """Rule 0: steer clear of the planet's gravity well and surface."""

        planet = self.planet
        if planet is None:
            return False
        inside_well = planet.distance <= const.GRAVITY_RANGE
        surface = max(0.0, planet.distance - PLANET_RADIUS)
        on_course = (
            planet.closing_speed > 0.0
            and abs(planet.lateral_speed)
            < PLANET_COURSE_LATERAL_FRACTION * planet.closing_speed
        )
        collision_soon = (
            planet.closing_seconds(surface)
            < PLANET_COLLISION_SECONDS + self.span_seconds
        )
        if not inside_well and not (on_course and collision_soon):
            return False
        heading = self.side("avoid_planet", planet.bearing, PERPENDICULAR_AWAY_OFFSET)
        return self.steer(heading)

    def approach_opponent(self) -> bool:
        enemy = self.enemy
        if enemy is None:
            return False
        return self.steer(enemy.bearing)

    def approach_object(self, target: ObservedObject | None) -> bool:
        if target is None:
            return False
        return self.steer(target.bearing)

    def avoid_opponent(self) -> bool:
        enemy = self.enemy
        if enemy is None or enemy.closing_speed <= 0.0:
            return False
        heading = self.side("avoid_opponent", enemy.bearing, AVOID_ANGLE_OFFSET)
        planet = self.planet
        if (
            planet is not None
            and planet.distance < const.GRAVITY_RANGE * PLANET_AVOID_LOOKAHEAD
            and abs(wrap_degrees(planet.bearing - heading)) < PLANET_SIDESTEP_OFFSET
        ):
            # Sidestep the planet on the side away from the opponent.
            candidates = (
                wrap_degrees(planet.bearing + PLANET_SIDESTEP_OFFSET),
                wrap_degrees(planet.bearing - PLANET_SIDESTEP_OFFSET),
            )
            heading = max(
                candidates,
                key=lambda candidate: abs(wrap_degrees(candidate - enemy.bearing)),
            )
        return self.steer(heading)

    def gravity_whip(self) -> bool:
        planet = self.planet
        if planet is None or planet.distance <= const.GRAVITY_RANGE:
            return False
        heading = self.side("gravity_whip", planet.bearing, APPROACH_ANGLE_OFFSET)
        return self.steer(heading)

    def approach_range(self, distance_limit: float, key: str = "range") -> bool:
        """[Approach_Range]: close in at an angle, back off when too close."""

        enemy = self.enemy
        if enemy is None or distance_limit <= 0.0:
            return False
        if enemy.distance > distance_limit:
            heading = self.drift_cancelling_side(
                f"{key}_far",
                enemy,
                APPROACH_ANGLE_OFFSET,
            )
            return self.steer(heading)
        if enemy.distance < APPROACH_RANGE_INNER_FRACTION * distance_limit:
            heading = self.side(
                f"{key}_near",
                enemy.bearing,
                PERPENDICULAR_AWAY_OFFSET,
            )
            return self.steer(heading)
        return False

    def approach_weapon_range(self, action: int) -> bool:
        return self.approach_range(self.weapon(action).range, key=f"a{int(action)}_range")

    # --------------------------------------------------------------- weapons

    def a1(self, value: bool) -> bool:
        return self.plan.claim_a1(bool(value))

    def a2(self, value: bool) -> bool:
        return self.plan.claim_a2(bool(value))


StrategyFunction = Callable[[BaselineContext], None]


def _smallest_positive_root(a: float, b: float, c: float) -> float | None:
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        root = -c / b
        return root if root > 0 else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_discriminant = math.sqrt(discriminant)
    roots = sorted(
        (
            (-b - sqrt_discriminant) / (2.0 * a),
            (-b + sqrt_discriminant) / (2.0 * a),
        )
    )
    for root in roots:
        if root > 0.0:
            return root
    return None
