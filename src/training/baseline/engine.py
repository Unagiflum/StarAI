"""Rule engine shared by every baseline strategy.

A strategy is an ordered list of rules. Each rule may set some of the four
controls: turning, thrust, weapon 1 and weapon 2. The first rule to set a
control claims it, and later rules cannot change a claimed control. A rule
whose condition is false claims nothing. Controls nobody claims stay off.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    PLANET_GRAVITY,
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
# [Avoid_Opponent] applies only within this fraction of the arena size.
AVOID_RANGE_FRACTION = 0.3
PERPENDICULAR_AWAY_OFFSET = 110.0
PLANET_SIDESTEP_OFFSET = 45.0
# Fraction of a range below which [Approach_Range] backs off.
APPROACH_RANGE_INNER_FRACTION = 0.7
PLANET_AVOID_LOOKAHEAD = 1.5
# Rule 0, the planet: it steps in only when the ship's coasting path,
# followed this far ahead, brings the hull within the clearance of the
# surface (the hull's own reach plus the margin), and lets go once that path
# clears it by the release margin too.
PLANET_AVOID_HORIZON_SECONDS = 4.0
PLANET_CLEARANCE_MARGIN = 20.0
PLANET_RELEASE_MARGIN = 40.0
# Escape headings tried, in degrees off the planet's bearing: straight across
# the well, where thrust deepens the pass most quickly, and mostly out of it.
PLANET_ESCAPE_OFFSETS = (90.0, 135.0)
# Thrust only while the nose is within this many degrees of the escape
# heading, so a ship still turning away does not push itself further in.
PLANET_ESCAPE_THRUST_ANGLE = 90.0
# Escapes are compared by their closest predicted pass, so when every one
# hits, the shallowest graze wins. An escape, once chosen, is kept unless
# another's pass comes this much less close.
PLANET_SIDE_SWITCH_MARGIN = 30.0
# Sideways drift, as a fraction of top speed, that counts as orbiting.
ORBIT_DRIFT_THRESHOLD = 0.15
# The baseline decides every frame, so a decision lasts one frame.
DECISION_FRAMES = 1
# Speed above top speed, in world units per frame, that counts as overspeed.
OVERSPEED_MARGIN = 0.5
# Above top speed, coast when the wanted heading is within this many
# degrees of the direction of motion. With no drag, a ship pushed past top
# speed keeps it, so beyond this angle it must still thrust to turn.
OVERSPEED_COAST_ANGLE = 45.0
# Ships that coast above top speed rather than thrust. Opt-in: across every
# ship the arena showed no gain, so only the afterburner ship uses it.
COAST_ABOVE_TOP_SPEED_SHIPS = frozenset({"Thraddash"})


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
        # Degrees a turning frame moves the nose: a fresh press, and a
        # continued hold. Both start at one heading step and are refined
        # from observed headings.
        self.tap_degrees = turn_angle
        self.hold_degrees = turn_angle
        self.reset_episode()

    def forget_previous_frame(self) -> None:
        """Skip turn measurement across frames other control flew."""

        self.previous_heading = None
        self.previous_turn_state = TURN_NONE

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
    ):
        self.ship_name = ship_name
        self.view = view
        self.memory = memory
        self.plan = ControlPlan()
        self.span_frames = DECISION_FRAMES
        # Strategies may opt in for one decision; see COAST_ABOVE_TOP_SPEED_SHIPS.
        self.coast_above_top_speed = ship_name in COAST_ABOVE_TOP_SPEED_SHIPS

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

    def overspeed(self) -> bool:
        """Whether thrust would slow me: above top speed, outside a gravity well.

        Above top speed, thrust in any direction pulls the speed back toward
        it, except inside a planet's gravity range.
        """

        if not self.spec.inertial:
            return False
        planet = self.planet
        if planet is not None and planet.distance <= const.GRAVITY_RANGE:
            return False
        return self.me.speed > self.spec.max_thrust + OVERSPEED_MARGIN

    def thrust_helps(self, bearing: float) -> bool:
        """Whether thrust toward ``bearing`` is worth its cost in speed.

        Above top speed, coast while already moving roughly toward the
        heading; thrust is still needed to turn the motion itself.
        """

        if not self.coast_above_top_speed or not self.overspeed():
            return True
        motion = bearing_of(self.me.velocity_forward, self.me.velocity_right)
        return abs(wrap_degrees(float(bearing) - motion)) > OVERSPEED_COAST_ANGLE

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

    def steer(self, bearing: float | None, thrust: bool = True, brake: bool = False) -> bool:
        """Claim turning and thrust to head toward ``bearing``.

        Above top speed thrust only slows the ship, so it stays off unless
        ``brake`` asks for exactly that.
        """

        if bearing is None or self.plan.steering_claimed:
            return False
        self.plan.claim_turn(self.turn_toward(bearing))
        if thrust and not brake and not self.thrust_helps(bearing):
            thrust = False
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
        """Rule 0: keep the hull off the planet's surface.

        Steps in only when the coasting path hits the surface within the
        horizon, so a pass that already misses is left to the strategy.
        Gravity outpulls the weakest engines, so climbing straight out
        rarely works; speed across the line to the planet is what carries
        a ship past it. Each escape heading is flown ahead with the ship's
        own turn and thrust timing, so the side chosen weighs both how far
        the nose must turn and the drift the ship already carries. Thrust
        waits until the nose is within ``PLANET_ESCAPE_THRUST_ANGLE`` of the
        escape heading, rather than pushing the ship in while it turns.
        """

        planet = self.planet
        if planet is None:
            return False
        frames = int(round(PLANET_AVOID_HORIZON_SECONDS * const.FPS))
        if planet.distance - planet.relative_speed * frames > const.GRAVITY_RANGE:
            return False
        spec = self.spec
        position = (-planet.forward, -planet.right)
        velocity = (-planet.velocity_forward, -planet.velocity_right)
        active = "avoid_planet" in self.memory.sides
        coast = simulate_planet_pass(position, velocity, frames, spec)
        release = planet_contact_distance(spec)
        if active:
            release += PLANET_RELEASE_MARGIN
        if coast.closest >= release:
            return False

        # Each plan is one side and one offset, flown as the simulation
        # flies it; keeping the plan keeps the prediction true.
        plans: dict[tuple[int, int], tuple[float, float]] = {}
        for sign in (1, -1):
            for index, offset in enumerate(PLANET_ESCAPE_OFFSETS):
                result = simulate_planet_pass(
                    position, velocity, frames, spec, escape=(sign, offset)
                )
                heading = wrap_degrees(planet.bearing + sign * offset)
                # Ties go to the heading the nose reaches soonest.
                plans[(sign, index)] = (result.closest, -abs(heading))
        chosen = max(plans, key=plans.__getitem__)
        previous = (
            self.memory.sides.get("avoid_planet"),
            self.memory.sides.get("avoid_planet_offset"),
        )
        if (
            previous in plans
            and previous != chosen
            and plans[chosen][0] < plans[previous][0] + PLANET_SIDE_SWITCH_MARGIN
        ):
            chosen = previous
        sign, index = chosen
        for key, value in (("avoid_planet", sign), ("avoid_planet_offset", index)):
            self.memory.touched_sides.add(key)
            self.memory.sides[key] = value
        heading = wrap_degrees(planet.bearing + sign * PLANET_ESCAPE_OFFSETS[index])
        return self.steer(heading, thrust=abs(heading) <= PLANET_ESCAPE_THRUST_ANGLE)

    def approach_opponent(self) -> bool:
        enemy = self.enemy
        if enemy is None:
            return False
        return self.steer(enemy.bearing)

    def intercept_seconds(self, reach: float) -> float:
        """Seconds to bring the opponent within ``reach`` flying at it at top speed.

        Uses the opponent's current motion along the line between us; returns
        infinity when it moves away faster than this ship can follow.
        """

        enemy = self.enemy
        if enemy is None:
            return math.inf
        gap = enemy.distance - float(reach)
        if gap <= 0.0:
            return 0.0
        radians = math.radians(enemy.bearing)
        own_toward = (
            self.me.velocity_forward * math.cos(radians)
            + self.me.velocity_right * math.sin(radians)
        )
        enemy_toward = enemy.closing_speed - own_toward
        achievable = self.spec.max_thrust + enemy_toward
        if achievable <= 0.0:
            return math.inf
        return gap / (achievable * const.FPS)

    def approach_object(self, target: ObservedObject | None) -> bool:
        if target is None:
            return False
        return self.steer(target.bearing)

    def avoid_opponent(self) -> bool:
        """[Avoid_Opponent]: flee a closing opponent at 135 degrees off its bearing.

        Applies only while the opponent closes and is nearer than
        ``AVOID_RANGE_FRACTION`` of the arena. If the escape heading would
        lead within 45 degrees of a nearby planet, fly 45 degrees to the side
        of the planet instead, on whichever side the nose reaches sooner.
        """

        enemy = self.enemy
        if (
            enemy is None
            or enemy.closing_speed <= 0.0
            or enemy.distance >= float(const.ARENA_SIZE) * AVOID_RANGE_FRACTION
        ):
            return False
        heading = self.side("avoid_opponent", enemy.bearing, AVOID_ANGLE_OFFSET)
        planet = self.planet
        if (
            planet is not None
            and planet.distance < const.GRAVITY_RANGE * PLANET_AVOID_LOOKAHEAD
            and abs(wrap_degrees(heading - planet.bearing)) < PLANET_SIDESTEP_OFFSET
        ):
            heading = min(
                (
                    wrap_degrees(planet.bearing + PLANET_SIDESTEP_OFFSET),
                    wrap_degrees(planet.bearing - PLANET_SIDESTEP_OFFSET),
                ),
                key=abs,
            )
        return self.steer(heading)

    def gravity_whip(self) -> bool:
        planet = self.planet
        if planet is None or planet.distance <= const.GRAVITY_RANGE:
            return False
        heading = self.side("gravity_whip", planet.bearing, APPROACH_ANGLE_OFFSET)
        return self.steer(heading)

    def approach_range(
        self,
        distance_limit: float,
        key: str = "range",
        inner_fraction: float = APPROACH_RANGE_INNER_FRACTION,
    ) -> bool:
        """[Approach_Range]: close in at an angle, back off when too close.

        Between ``inner_fraction`` of ``distance_limit`` and the limit
        itself, steering is left to later rules.
        """

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
        if enemy.distance < float(inner_fraction) * distance_limit:
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


@dataclass(frozen=True, slots=True)
class PlanetPass:
    """Predicted pass by the planet: nearest centre distance, and the frame
    the hull first comes within the clearance of the surface."""

    closest: float
    impact_frame: float = math.inf


def planet_contact_distance(spec: ShipSpec) -> float:
    """Centre distance from the planet that rule 0 keeps this hull beyond."""

    return PLANET_RADIUS + spec.planet_reach + PLANET_CLEARANCE_MARGIN


def _planet_gravity(x: float, y: float, distance: float, pull: float) -> tuple[float, float]:
    if pull <= 0.0 or distance < PLANET_RADIUS or distance > const.GRAVITY_RANGE:
        return 0.0, 0.0
    return -pull * x / distance, -pull * y / distance


def simulate_planet_pass(
    position: tuple[float, float],
    velocity: tuple[float, float],
    frames: int,
    spec: ShipSpec,
    escape: tuple[int, float] | None = None,
) -> PlanetPass:
    """Fly the ship past the planet for up to ``frames`` frames.

    ``position`` is the ship's offset from the planet and ``velocity`` its
    motion, both in the ship's frame at the start (forward, right). With no
    ``escape`` the ship coasts. With ``escape = (sign, offset)`` it flies as
    rule 0 does: every frame it aims ``offset`` degrees off the planet's
    bearing on the ``sign`` side, turning one heading step and thrusting one
    increment at the ship's own cadences, and thrusting only while the nose
    is within ``PLANET_ESCAPE_THRUST_ANGLE`` of that heading. This follows
    the simulation's thrust caps and Verlet step; non-inertial ships feel no
    gravity and stop whenever they do not thrust.
    """

    x, y = float(position[0]), float(position[1])
    vx, vy = float(velocity[0]), float(velocity[1])
    inertial = spec.inertial
    pull = const.GRAVITY_MULTIPLIER * PLANET_GRAVITY if inertial else 0.0
    contact = planet_contact_distance(spec)
    turn_step = float(const.TURN_ANGLE) * const.DIRECTIONS_MULTIPLIER
    turn_period = spec.turn_wait + 1
    thrust_period = spec.thrust_wait + 1
    next_turn = 0
    next_thrust = 0
    nose = 0.0
    distance = math.hypot(x, y)
    closest = distance
    impact_frame = math.inf
    gx, gy = _planet_gravity(x, y, distance, pull)
    for frame in range(frames):
        if escape is not None:
            sign, offset = escape
            target = bearing_of(-x, -y) + sign * offset
            error = wrap_degrees(target - nose)
            thrusting = abs(error) <= PLANET_ESCAPE_THRUST_ANGLE
            if frame >= next_turn and abs(error) > turn_step / 2.0:
                nose = wrap_degrees(nose + math.copysign(turn_step, error))
                next_turn = frame + turn_period
            if thrusting and frame >= next_thrust:
                next_thrust = frame + thrust_period
                radians = math.radians(nose)
                dx, dy = math.cos(radians), math.sin(radians)
                if inertial:
                    vx, vy = _thrusted_velocity(
                        vx, vy, dx, dy, spec, inside_well=distance <= const.GRAVITY_RANGE
                    )
                else:
                    vx, vy = dx * spec.max_thrust, dy * spec.max_thrust
            elif not inertial and not thrusting:
                vx, vy = 0.0, 0.0
        x += vx + 0.5 * gx
        y += vy + 0.5 * gy
        distance = math.hypot(x, y)
        nx, ny = _planet_gravity(x, y, distance, pull)
        vx += 0.5 * (gx + nx)
        vy += 0.5 * (gy + ny)
        gx, gy = nx, ny
        speed = math.hypot(vx, vy)
        if speed > const.SPEED_LIMIT:
            vx *= const.SPEED_LIMIT / speed
            vy *= const.SPEED_LIMIT / speed
        closest = min(closest, distance)
        if distance <= contact and math.isinf(impact_frame):
            impact_frame = float(frame + 1)
        if distance <= PLANET_RADIUS:
            # Through the surface: no pass comes any closer.
            break
        if distance > const.GRAVITY_RANGE and x * vx + y * vy > 0.0:
            # Out of the well and leaving: nothing pulls it back.
            break
    return PlanetPass(closest=closest, impact_frame=impact_frame)


def _thrusted_velocity(
    vx: float,
    vy: float,
    dx: float,
    dy: float,
    spec: ShipSpec,
    *,
    inside_well: bool,
) -> tuple[float, float]:
    """Velocity after one thrust increment, as ``apply_thrust`` caps it."""

    increment = spec.thrust_increment
    nx, ny = vx + dx * increment, vy + dy * increment
    speed = math.hypot(nx, ny)
    scale = 1.0
    if speed > spec.max_thrust and not inside_well:
        scale = spec.max_thrust / speed
    if speed > const.MAX_GRAV_WHIP:
        scale = const.MAX_GRAV_WHIP / speed
    change_x, change_y = nx * scale - vx, ny * scale - vy
    change = math.hypot(change_x, change_y)
    if change > increment > 0.0:
        change_x *= increment / change
        change_y *= increment / change
    return vx + change_x, vy + change_y


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
