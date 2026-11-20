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
from src.training.baseline.flight import ShipFlight
from src.training.baseline.threats import (
    THREAT_SIZE,
    FlownPath,
    ThreatTrack,
    arrival_frames,
    fly_move,
    judge_move,
    straight_line_danger,
    threat_tracks,
)
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
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS

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
# Rule 1, incoming shots: how far ahead moves are flown, the headings tried
# (degrees off the nose), and thrust only while the nose is within this many
# degrees of the heading. Threats whose straight path passes farther than
# the hull's reach plus the slack are ignored. Moves are flown until every
# threat has passed or expired, up to the horizon; a few frames more see
# a pass just after.
EVADE_HORIZON_SECONDS = 2.5
EVADE_HORIZON_FRAMES = int(round(EVADE_HORIZON_SECONDS * const.FPS))
EVADE_EXTRA_FRAMES = 4
# A dodge waits while one begun this many frames later still does as well.
EVADE_DELAY_FRAMES = 6
EVADE_HEADINGS = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
EVADE_THRUST_ANGLE = 90.0
EVADE_PATH_SLACK = 250.0
# The rule lets go once the shots it dodged miss by this much.
EVADE_RELEASE_MARGIN = 30.0
# The planet's pull is flown when it is within this much of its well.
EVADE_PLANET_MARGIN = 300.0
# Moves are scored in crew: the hits they take, plus, for a shot fired to
# destroy a threat, this many crew per full battery it costs. Charging for
# the time spent turning as well did worse in the arena; it only breaks ties.
EVADE_SHOT_COST = 0.5
# Kinds of evade plan kept in memory.
EVADE_COAST = 0
EVADE_HEADING = 1
EVADE_SHOOT = 2
# Ships the rule leaves alone. It cannot fly the Slylandro's controls; the
# others did worse with it in the arena: the Supox and Umgah have their own
# dodges on A2, the Syreen must close in to sing, and the Utwig shield turns
# hits into battery.
EVADE_EXCLUDED_SHIPS = frozenset({"Slylandro", "Supox", "Syreen", "Umgah", "Utwig"})
# Ships whose weapon 1 is not a plain shot to aim at a threat: homing,
# charged, parked or self-aiming weapons. The Chenjesu crystal flies until
# released, so it meets a threat ahead like any other shot.
SHOOT_DOWN_EXCLUDED_SHIPS = frozenset(
    {"Androsynth", "Arilou", "Earthling", "KohrAh", "Melnorme", "Mycon", "Umgah"}
)
DRUUGE_RECOIL = (
    float(ABILITY_DEFINITIONS["DruugeA1"].recoil_increment or 0.0),
    float(ABILITY_DEFINITIONS["DruugeA1"].max_recoil or 0.0),
)
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
    def enemy_form(self) -> str:
        name = self.enemy_name
        if name == "Mmrnmrhm":
            return FORM_YWING if self.enemy_block.mmrnmrhm_ywing else FORM_XFORM
        if name == "Androsynth" and self.enemy_block.androsynth_blazer:
            return FORM_BLAZER
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

    def evade_threats(self) -> bool:
        """Rule 1: keep enemy shots in flight off the hull.

        Steps in only when the path the ship is on (no turning, thrust as
        held) is predicted to take a hit within the horizon, and some move
        does better. The moves are coasting, flying one of
        ``EVADE_HEADINGS`` (thrusting once the nose is within
        ``EVADE_THRUST_ANGLE`` of it), and turning to shoot one threat down
        with weapon 1 where the shot can destroy it before it lands; the
        Druuge's shots also kick it backward. Moves are scored by the crew
        they lose, plus for a shot the battery it costs, and among equals the
        one turning the nose for the least time wins. A move, once chosen,
        is kept while nothing does better, and the rule lets go once neither the
        held path nor coasting comes within ``EVADE_RELEASE_MARGIN`` of a hit.
        """

        if self.ship_name in EVADE_EXCLUDED_SHIPS or self.plan.steering_claimed:
            return False
        spec = self.spec
        reach = spec.shot_reach
        tracks = [
            track
            for track in threat_tracks(
                self.view.enemy_objects,
                (self.me.velocity_forward, self.me.velocity_right),
                enemy_name=self.enemy_name,
                enemy_form=self.enemy_form,
            )
            if straight_line_danger(
                track, EVADE_HORIZON_FRAMES, reach + THREAT_SIZE + EVADE_PATH_SLACK
            )
        ]
        if not tracks:
            return False
        frames = min(
            EVADE_HORIZON_FRAMES,
            max(arrival_frames(track, EVADE_HORIZON_FRAMES) for track in tracks)
            + EVADE_EXTRA_FRAMES,
        )
        planet = self.planet
        planet_centre = None
        if planet is not None and planet.distance <= const.GRAVITY_RANGE + EVADE_PLANET_MARGIN:
            planet_centre = (planet.forward, planet.right)
        velocity = (self.me.velocity_forward, self.me.velocity_right)
        held_thrust = self.held("thrust") and self.ship_name != "Slylandro"

        def judge(controller, **kwargs):
            path, _ = fly_move(
                spec,
                velocity,
                frames,
                controller,
                planet=planet_centre,
                recoil=kwargs.pop("recoil", None),
            )
            return path, judge_move(tracks, path, reach, planet=planet_centre, **kwargs)

        _, held = judge(lambda flight: (None, held_thrust, False))
        coast_path, coast = judge(lambda flight: (None, False, False))
        active = "evade" in self.memory.sides
        if held.damage <= 0.0 and coast.damage <= 0.0:
            if not active or min(held.clearance, coast.clearance) >= EVADE_RELEASE_MARGIN:
                return False
        baseline_damage = held.damage

        # Each option: (crew lost, other cost in crew, seconds spent turning,
        # plan, claim). Plans are kept as absolute headings so a kept plan
        # means the same move next frame.
        options: list[
            tuple[float, float, float, tuple[int, int], Callable[[], bool]]
        ] = []
        own_heading = self.me.heading
        options.append((coast.damage, 0.0, 0.0, (EVADE_COAST, 0), self._evade_coast))
        headings = {wrap_degrees(offset) for offset in EVADE_HEADINGS}
        previous_kind = self.memory.sides.get("evade")
        previous_heading = self.memory.sides.get("evade_heading")
        if previous_kind == EVADE_HEADING and previous_heading is not None:
            headings.add(wrap_degrees(float(previous_heading) - own_heading))
        for bearing in sorted(headings):
            _, outcome = judge(_heading_controller(bearing))
            absolute = int(round((own_heading + bearing) % 360.0))
            options.append(
                (
                    outcome.damage,
                    0.0,
                    self._turn_seconds(bearing),
                    (EVADE_HEADING, absolute),
                    lambda bearing=bearing: self.steer(
                        bearing, thrust=abs(bearing) <= EVADE_THRUST_ANGLE
                    ),
                )
            )
        options.extend(self._shoot_down_options(tracks, coast_path, judge))

        best = min(options, key=lambda option: (option[0] + option[1], option[2]))
        best_score = best[0] + best[1]
        if best_score >= baseline_damage and not active:
            return False
        if not active and best[3][0] != EVADE_SHOOT:
            # Leave the ship to its strategy while a dodge begun a little
            # later still does as well.
            hold = lambda flight: (None, held_thrust, False)
            for bearing in sorted(headings, key=abs):
                _, outcome = judge(
                    _delayed(hold, _heading_controller(bearing), EVADE_DELAY_FRAMES)
                )
                if outcome.damage <= best_score:
                    return False
        if active:
            for option in options:
                same_plan = option[3][0] == previous_kind and (
                    previous_kind != EVADE_HEADING
                    or abs(wrap_degrees(option[3][1] - float(previous_heading or 0))) < 1.0
                )
                if same_plan and option[0] + option[1] <= best_score:
                    best = option
                    break
        kind, value = best[3]
        self.memory.touched_sides.update(("evade", "evade_heading"))
        self.memory.sides["evade"] = kind
        self.memory.sides["evade_heading"] = value
        return best[4]()

    def _turn_seconds(self, bearing: float) -> float:
        """Seconds the nose takes to turn to ``bearing``."""

        fine = float(const.TURN_ANGLE)
        step = fine * const.DIRECTIONS_MULTIPLIER
        steps = math.ceil(max(0.0, abs(float(bearing)) - fine / 2.0) / step)
        if steps <= 0:
            return 0.0
        frames = (steps - 1) * (self.spec.turn_wait + 1) + 1
        return frames / float(const.FPS)

    def _evade_coast(self) -> bool:
        self.plan.claim_turn(TURN_NONE)
        self.plan.claim_thrust(False)
        return True

    def _shoot_down_options(self, tracks, coast_path, judge):
        """Options that turn to fire weapon 1 at one threat."""

        if self.ship_name in SHOOT_DOWN_EXCLUDED_SHIPS:
            return []
        weapon = self.weapon(1)
        laser = weapon.kind == "laser" and not weapon.tracking
        if not laser and not (
            weapon.kind == "projectile" and not weapon.tracking and weapon.speed > 0.0
        ):
            return []
        if self.me.battery < self.spec.a1_cost:
            return []
        recoil = DRUUGE_RECOIL if self.ship_name == "Druuge" else None
        ready_frame = int(math.ceil(max(0.0, self.me.timer_seconds("a1_timer")) * const.FPS))
        fine = float(const.TURN_ANGLE)
        step = float(const.TURN_ANGLE) * const.DIRECTIONS_MULTIPLIER
        period = self.spec.turn_wait + 1
        guns = self._gun_offsets(1)
        own_heading = self.me.heading
        options = []
        for index, track in enumerate(tracks):
            if recoil is None and weapon.damage < track.hit_points:
                continue
            aim = _lead_bearing(track, coast_path, weapon, 0)
            if aim is None:
                continue
            gun = min(guns, key=lambda offset: abs(wrap_degrees(aim - offset)))
            error = wrap_degrees(aim - gun)
            turns = int(math.ceil(max(0.0, abs(error) - fine / 2.0) / step))
            fire_frame = max(ready_frame, (turns - 1) * period + 1 if turns else 0)
            if fire_frame >= len(coast_path.xs) - 1:
                continue
            refined = _lead_bearing(track, coast_path, weapon, fire_frame)
            if refined is not None:
                aim = refined
                error = wrap_degrees(aim - gun)
            # The nose can only point along the ship's own heading steps.
            absolute = round((own_heading + error) / fine) * fine
            nose = wrap_degrees(absolute - own_heading)
            direction = nose + gun

            def controller(flight, nose=nose, fire_frame=fire_frame):
                if flight.frame == fire_frame:
                    flight.nose = nose
                    return None, False, True
                if flight.frame < fire_frame:
                    return flight.steering_toward(nose), False, False
                return None, False, False

            _, outcome = judge(
                controller,
                recoil=recoil,
                shot=(fire_frame, weapon, direction, index),
            )
            cost = EVADE_SHOT_COST * self.spec.a1_cost / max(1.0, self.me.max_battery)
            options.append(
                (
                    outcome.damage,
                    cost,
                    self._turn_seconds(nose),
                    (EVADE_SHOOT, int(round(absolute % 360.0))),
                    lambda nose=nose: self._shoot_down(nose),
                )
            )
        return options

    def _shoot_down(self, nose: float) -> bool:
        self.point_at(nose)
        self.plan.claim_thrust(False)
        if abs(nose) <= float(const.TURN_ANGLE) / 2.0 + 1e-6 and self.me.a1_ready:
            self.a1(True)
            if self.ship_name == "Orz":
                # Weapon 1 with weapon 2 launches a marine instead.
                self.a2(False)
        return True

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


def _heading_controller(bearing: float):
    """Fly toward ``bearing`` in the starting frame, thrusting once the nose
    is within ``EVADE_THRUST_ANGLE`` of it."""

    def controller(flight: ShipFlight):
        error = flight.steering_toward(bearing)
        return error, abs(error) <= EVADE_THRUST_ANGLE, False

    return controller


def _delayed(first, then, frames: int):
    """Fly ``first`` for ``frames`` frames, then ``then``."""

    def controller(flight: ShipFlight):
        return (first if flight.frame < frames else then)(flight)

    return controller


def _lead_bearing(
    track: ThreatTrack,
    path: FlownPath,
    weapon: WeaponSpec,
    fire_frame: int,
) -> float | None:
    """Bearing, in the starting frame, a shot fired at ``fire_frame`` from
    ``path`` must travel to meet ``track`` flying straight on."""

    fire_frame = min(fire_frame, len(path.xs) - 1)
    qx = track.x + track.vx * fire_frame - path.xs[fire_frame]
    qy = track.y + track.vy * fire_frame - path.ys[fire_frame]
    if weapon.kind == "laser":
        return bearing_of(qx, qy)
    carry = weapon.parent_velocity
    wx = track.vx - carry * path.vxs[fire_frame]
    wy = track.vy - carry * path.vys[fire_frame]
    speed = weapon.speed
    frames = _smallest_positive_root(
        wx * wx + wy * wy - speed * speed,
        2.0 * (qx * wx + qy * wy),
        qx * qx + qy * qy,
    )
    if frames is None:
        return None
    return bearing_of(qx + wx * frames, qy + wy * frames)


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
    is within ``PLANET_ESCAPE_THRUST_ANGLE`` of that heading. Flight follows
    ``flight.ShipFlight``.
    """

    flight = ShipFlight(
        spec, position, velocity, planet=(0.0, 0.0), drifting=escape is None
    )
    contact = planet_contact_distance(spec)
    closest = flight.planet_distance
    impact_frame = math.inf
    for frame in range(frames):
        if escape is None:
            flight.step(None, False)
        else:
            sign, offset = escape
            target = bearing_of(-flight.x, -flight.y) + sign * offset
            error = flight.steering_toward(target)
            flight.step(error, abs(error) <= PLANET_ESCAPE_THRUST_ANGLE)
        distance = flight.planet_distance
        closest = min(closest, distance)
        if distance <= contact and math.isinf(impact_frame):
            impact_frame = float(frame + 1)
        if distance <= PLANET_RADIUS:
            # Through the surface: no pass comes any closer.
            break
        if (
            distance > const.GRAVITY_RANGE
            and flight.x * flight.vx + flight.y * flight.vy > 0.0
        ):
            # Out of the well and leaving: nothing pulls it back.
            break
    return PlanetPass(closest=closest, impact_frame=impact_frame)


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
