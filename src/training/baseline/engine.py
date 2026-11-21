"""Rule engine shared by every baseline strategy.

A strategy is an ordered list of rules. Each rule may set some of the four
controls: turning, thrust, weapon 1 and weapon 2. The first rule to set a
control claims it, and later rules cannot change a claimed control. A rule
whose condition is false claims nothing. Controls nobody claims stay off.
The one exception is the last rule, keep moving, which for a few ships may
take back a turn the strategy claimed only to aim while the weapon could
not fire anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import src.const as const
from src.training.baseline.behaviours import ShipBehaviour, ship_behaviour
from src.training.baseline.flight import ShipFlight, turn_step_degrees
from src.training.baseline.threats import (
    SHOT_CONTACT,
    THREAT_SIZE,
    FlownPath,
    ThreatTrack,
    arrival_frames,
    fly_move,
    judge_move,
    pass_frames,
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
    TYPICAL_HULL_RADIUS,
    ShipSpec,
    WeaponSpec,
    ship_spec,
    weapon_spec,
)
from src.training.contracts import ACTION_INDEX_TABLE
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS

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
# followed ahead for the ship's horizon, brings the hull within the clearance
# of the surface (the hull's own reach plus the margin), and lets go once
# that path clears it by the release margin too. The horizon is the frames
# the ship takes to turn about and reach top speed, plus the margin frames,
# kept within the bounds in seconds: a ship that turns and accelerates
# quickly needs to see less of its path to get clear.
PLANET_HORIZON_MARGIN_FRAMES = 12
PLANET_HORIZON_MIN_SECONDS = 1.0
PLANET_HORIZON_MAX_SECONDS = 4.0
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
EVADE_SLIDE = 3
EVADE_HOP = 4
EVADE_TURRET = 5
# With weapon 2 held, the Supox turning keys and thrust push it sideways
# and backward: (turn key, thrust key, degrees off the nose pushed).
SUPOX_SLIDES = (
    (TURN_LEFT, False, -90.0),
    (TURN_RIGHT, False, 90.0),
    (TURN_NONE, True, 180.0),
    (TURN_LEFT, True, -135.0),
    (TURN_RIGHT, True, 135.0),
)
# The Umgah's weapon 2 hops it this far straight backward, once every
# period, leaving it at rest. Rule 1 tries runs of these many hops.
UMGAH_HOP_DISTANCE = float(ABILITY_DEFINITIONS["UmgahA2"].backup_speed) * const.SPEED_SCALE
UMGAH_HOP_FRAMES = SHIP_DEFINITIONS["Umgah"].a2_wait + 1
EVADE_HOP_COUNTS = (1, 2)
# Ships whose weapon 1 is held. The Chenjesu crystal and the Kohr-Ah saw
# fly while A1 is held, and burst or park on release; the Melnorme shot
# charges at the nose while held and flies on release. A new crystal or saw
# needs A1 released for a frame first. Rule 1 weighs what the held shot is
# worth against a shoot-down, and a threat my shot in flight (or the
# Melnorme charge) is due to meet first counts as stopped.
HELD_WEAPON_SHIPS = frozenset({"Chenjesu", "KohrAh", "Melnorme"})
# A crystal or saw in flight is worth its damage when its closest pass of the
# enemy hull comes within the hull's reach, falling to nothing at twice that,
# and nothing when the pass is more than HELD_SHOT_HORIZON_FRAMES away or
# behind it. Parked Kohr-Ah saws fly slower than HELD_SHOT_MIN_SPEED.
HELD_SHOT_HORIZON_FRAMES = 72
HELD_SHOT_MIN_SPEED = 20.0
# A Melnorme charge not lined up on the enemy in range is worth this share
# of its damage.
MELNORME_UNAIMED_CHARGE_WEIGHT = 0.5
# My shot meets a threat when their centres pass this close.
HELD_SHOT_INTERCEPT_CONTACT = SHOT_CONTACT + THREAT_SIZE
# The Orz turret: weapon 2 held with a turn key turns the turret, not the
# hull, one sprite facing every ORZ_TURRET_STEP_FRAMES frames. Weapon 1
# cannot fire while weapon 2 is held (both together launch a marine).
ORZ_TURRET_STEP = 360.0 / const.ASSET_SPRITE_DIRECTIONS
ORZ_TURRET_STEP_FRAMES = SHIP_DEFINITIONS["Orz"].a2_wait + 1
# Chmmr satellites, as observed: enemy objects that never expire and orbit
# the Chmmr at its satellite distance, one turn every satellite period frames.
# A satellite laser reaches SATELLITE_LASER_RANGE and the Chmmr laser
# CHMMR_LASER_RANGE; beyond both the ship is out of their reach.
SATELLITE_ORBIT = float(SHIP_DEFINITIONS["Chmmr"].satellite_distance)
SATELLITE_TURN_RADIANS = 2.0 * math.pi / float(SHIP_DEFINITIONS["Chmmr"].satellite_period)
SATELLITE_ORBIT_TOLERANCE = 96.0
# Shots and beams meet a satellite within this distance of its centre.
SATELLITE_HIT_RADIUS = 20.0
# Standoff from a closing opponent (such as a tractoring Chmmr): run once
# it would reach the near edge before the ship could turn away, with this
# much time to spare; the closing acceleration is smoothed with this weight
# on each frame, as the tractor pulls in pulses. Close in from far no faster
# than the approach speed.
STANDOFF_TURN_MARGIN = 0.5
STANDOFF_ACCEL_SMOOTHING = 0.1
STANDOFF_APPROACH_SPEED = 15.0
SATELLITE_LASER_RANGE = float(ABILITY_DEFINITIONS["ChmmrSatelliteLaser"].range)
CHMMR_LASER_RANGE = float(ABILITY_DEFINITIONS["ChmmrA1"].range)
DRUUGE_RECOIL = (
    float(ABILITY_DEFINITIONS["DruugeA1"].recoil_increment or 0.0),
    float(ABILITY_DEFINITIONS["DruugeA1"].max_recoil or 0.0),
)
# Last rule, keep moving: below this fraction of top speed, a ship no rule
# moves builds speed across the line to the opponent, thrusting once the
# nose is within the angle of that heading. Ships that opt out are in
# ``behaviours.SHIP_BEHAVIOURS``.
KEEP_MOVING_SPEED_FRACTION = 0.5
KEEP_MOVING_THRUST_ANGLE = 45.0
# Sideways drift, as a fraction of top speed, that counts as orbiting.
ORBIT_DRIFT_THRESHOLD = 0.15
# The baseline decides every frame, so a decision lasts one frame.
DECISION_FRAMES = 1
# Speed above top speed, in world units per frame, that counts as overspeed.
OVERSPEED_MARGIN = 0.5
# Ilwrath flames, measured on the real collision masks: held flames reach a
# hull about 300 units ahead centre to centre.
ILWRATH_FLAME_REACH = 300.0
# Recharging (``recharge_retreat``): keep the opponent beyond its weapons'
# reach plus a hull and this margin, running until it is this much farther.
RECHARGE_REACH_MARGIN = 150.0
RECHARGE_RELEASE_MARGIN = 200.0
# Above top speed, coast when the wanted heading is within this many
# degrees of the direction of motion. With no drag, a ship pushed past top
# speed keeps it, so beyond this angle it must still thrust to turn.
OVERSPEED_COAST_ANGLE = 45.0


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
        self.behaviour: ShipBehaviour = ship_behaviour(ship_name)
        # Strategies may opt in for one decision.
        self.coast_above_top_speed = self.behaviour.coast_above_top_speed

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

    def enemy_reach(self) -> float:
        """Longest reach of the opponent's direct weapons (not escorts)."""

        name = self.enemy_name
        if name is None:
            return 0.0
        if name == "Ilwrath":
            return ILWRATH_FLAME_REACH
        form = self.enemy_form
        reach = 0.0
        for action in (1, 2):
            weapon = weapon_spec(name, action, form)
            if weapon.kind in ("projectile", "laser", "area"):
                reach = max(reach, weapon.range)
        return reach

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

    # ---------------------------------------------------------- held weapons

    def held_shot(self) -> ObservedObject | None:
        """The shot weapon 1 is holding, for ``HELD_WEAPON_SHIPS`` with A1 held.

        The Melnorme charge rides at the nose, at rest relative to the ship.
        The Chenjesu crystal or Kohr-Ah saw is the fastest of my shots still
        in flight; parked saws and crystal shards don't count. None when
        nothing is held, or A1 is held on a shot already gone.
        """

        if self.ship_name not in HELD_WEAPON_SHIPS or not self.held("a1"):
            return None
        shots = self.view.friendly_a1_objects
        if self.ship_name == "Melnorme":
            return min(
                (shot for shot in shots if shot.relative_speed < 1.0),
                key=lambda shot: shot.distance,
                default=None,
            )
        own_forward = self.me.velocity_forward
        own_right = self.me.velocity_right

        def speed(shot: ObservedObject) -> float:
            return math.hypot(shot.velocity_forward + own_forward, shot.velocity_right + own_right)

        minimum_damage = 0.0
        if self.ship_name == "Chenjesu":
            minimum_damage = float(ABILITY_DEFINITIONS["ChenjesuA1"].fragment_damage)
        flying = [
            shot
            for shot in shots
            if speed(shot) > HELD_SHOT_MIN_SPEED and -shot.crew_effect > minimum_damage
        ]
        return max(flying, key=speed, default=None)

    def held_shot_value(self) -> float | None:
        """Crew the held shot is expected to take from the enemy; None when
        weapon 1 holds nothing.

        A crystal or saw in flight is worth its damage on a predicted hit
        (see ``held_pass_share``); one receding or due to miss is worth
        nothing. The Melnorme charge is worth its current damage when the
        enemy is in range along the nose, else ``MELNORME_UNAIMED_CHARGE_WEIGHT``
        of it.
        """

        if self.ship_name not in HELD_WEAPON_SHIPS or not self.held("a1"):
            return None
        shot = self.held_shot()
        enemy = self.enemy
        if shot is None or enemy is None:
            return 0.0
        damage = max(0.0, -shot.crew_effect)
        if self.ship_name == "Melnorme":
            if self.in_range(1) and self.pointing(1):
                return damage
            return damage * MELNORME_UNAIMED_CHARGE_WEIGHT
        reach = ship_spec(self.enemy_name, self.enemy_form).shot_reach
        return damage * held_pass_share(shot, enemy, reach)

    def held_shot_meets_threat(self) -> bool:
        """Whether the held shot is due to meet an enemy object it destroys
        before that object reaches me."""

        shot = self.held_shot()
        if shot is None:
            return False
        return any(self._shot_stops(shot, threat) for threat in self.view.enemy_objects)

    def _shot_stops(self, shot: ObservedObject, threat: ObservedObject) -> bool:
        if -shot.crew_effect < threat.hit_points:
            return False
        frames, miss = threat.closest_approach()
        limit = frames if 0.0 <= frames and miss <= COLLISION_MARGIN else HELD_SHOT_HORIZON_FRAMES
        return pass_frames(shot, threat, HELD_SHOT_INTERCEPT_CONTACT, limit) is not None

    def stopped_threats(self, tracks) -> set[int]:
        """Indices of ``tracks`` my shots already out will stop first.

        For ``HELD_WEAPON_SHIPS`` only: a shot in flight, or the Melnorme
        charge at the nose, that its damage lets destroy the threat and is
        due to meet it, flying straight on, before the threat reaches me.
        """

        if self.ship_name not in HELD_WEAPON_SHIPS:
            return set()
        shots = list(self.view.friendly_a1_objects)
        stopped = set()
        for index, track in enumerate(tracks):
            threat = track.source
            limit = arrival_frames(track, EVADE_HORIZON_FRAMES)
            for shot in shots:
                if -shot.crew_effect < threat.hit_points:
                    continue
                if pass_frames(shot, threat, HELD_SHOT_INTERCEPT_CONTACT, limit) is not None:
                    stopped.add(index)
                    break
        return stopped

    def intercept(
        self,
        action: int,
        predict: Callable[[int], tuple[float, float]],
    ) -> tuple[float, int] | None:
        """Bearing to fire ``action`` along to meet a target, and the frames.

        ``predict(frames)`` gives the target's offset (forward, right) from
        this ship after ``frames`` frames, both moving as they are. Returns
        None when the weapon can't reach it within its reach.
        """

        weapon = self.weapon(action)
        if weapon.kind == "laser":
            forward, right = predict(0)
            if math.hypot(forward, right) > weapon.range:
                return None
            return bearing_of(forward, right), 0
        if weapon.speed <= 0.0:
            return None
        carry = 1.0 - weapon.parent_velocity
        own_forward = self.me.velocity_forward
        own_right = self.me.velocity_right
        last = int(math.floor(weapon.range / weapon.speed))
        for frames in range(1, last + 1):
            forward, right = predict(frames)
            # The shot moves at its speed relative to where this ship would
            # be had it kept only the share of its motion the shot carries.
            forward += carry * own_forward * frames
            right += carry * own_right * frames
            if math.hypot(forward, right) <= weapon.speed * frames:
                return bearing_of(forward, right), frames
        return None

    def gun_error(self, action: int, bearing: float) -> float:
        """Turn needed to bring the nearest gun of ``action`` to ``bearing``."""

        return min(
            (wrap_degrees(bearing - offset) for offset in self._gun_offsets(action)),
            key=abs,
        )

    def aim_gun(self, action: int, bearing: float, tolerance: float | None = None) -> bool:
        """Turn the nearest gun toward ``bearing``; True once it points there."""

        error = self.gun_error(action, bearing)
        self.point_at(error)
        if tolerance is None:
            tolerance = max(
                self.weapon(action).pointing_tolerance, self.memory.tap_degrees / 2.0
            )
        return abs(error) <= float(tolerance)

    def turn_turret(self, bearing: float) -> bool:
        """Turn the Orz turret a step toward ``bearing`` (from the nose);
        True once it points there, within half a step.

        The turn key serves the turret only with weapon 2 held, which keeps
        the hull from turning and weapon 1 from firing. So the turret turns
        only while no rule turns the hull, fires, or uses weapon 2 this
        frame; a hull held on its heading keeps it, and thrust is unaffected.
        """

        error = wrap_degrees(float(bearing) - self.me.orz_turret_bearing)
        if abs(error) <= ORZ_TURRET_STEP / 2.0 + 1e-6:
            return True
        plan = self.plan
        if plan.turn not in (None, TURN_NONE) or plan.a1 or plan.a2 is not None:
            return False
        plan.turn = TURN_RIGHT if error > 0.0 else TURN_LEFT
        plan.a2 = True
        plan.a1 = False
        return False

    # ---------------------------------------------------------- satellites

    def chmmr_satellites(self) -> tuple[ObservedObject, ...]:
        """The opponent Chmmr's live satellites, or none for other opponents."""

        enemy = self.enemy
        if enemy is None or self.enemy_name != "Chmmr":
            return ()
        found = []
        for obj in self.view.objects("enemy_non_a1"):
            if obj.expires or obj.hit_points <= 0.0:
                continue
            orbit = math.hypot(*self._wrapped_offset(obj, enemy))
            if abs(orbit - SATELLITE_ORBIT) <= SATELLITE_ORBIT_TOLERANCE:
                found.append(obj)
        return tuple(found)

    def _wrapped_offset(
        self, obj: ObservedObject, origin: ObservedObject
    ) -> tuple[float, float]:
        """Shortest offset (forward, right) from ``origin`` to ``obj``.

        Each is observed at its nearest copy on the wrapping arena, which may
        lie across the wrap from the other's, so shift by whole arenas along
        the world axes (turned into this ship's frame) to the shortest one.
        """

        forward = obj.forward - origin.forward
        right = obj.right - origin.right
        heading = math.radians(self.me.heading)
        size = float(const.ARENA_SIZE)
        # World x (screen right) and y (screen down) in (forward, right).
        x_axis = (math.sin(heading) * size, math.cos(heading) * size)
        y_axis = (-math.cos(heading) * size, math.sin(heading) * size)
        best = (forward, right)
        for x_shift in (-1, 0, 1):
            for y_shift in (-1, 0, 1):
                candidate = (
                    forward + x_shift * x_axis[0] + y_shift * y_axis[0],
                    right + x_shift * x_axis[1] + y_shift * y_axis[1],
                )
                if math.hypot(*candidate) < math.hypot(*best):
                    best = candidate
        return best

    def satellite_offset(self, satellite: ObservedObject, frames: int) -> tuple[float, float]:
        """Where ``satellite`` will be, relative to this ship, in ``frames``.

        It keeps circling the Chmmr in the direction it now moves, while the
        Chmmr drifts on as it is moving.
        """

        enemy = self.enemy
        rel_forward, rel_right = self._wrapped_offset(satellite, enemy)
        spin = rel_forward * (satellite.velocity_right - enemy.velocity_right) - rel_right * (
            satellite.velocity_forward - enemy.velocity_forward
        )
        turn = SATELLITE_TURN_RADIANS * frames * (1.0 if spin >= 0.0 else -1.0)
        radius = math.hypot(rel_forward, rel_right)
        angle = math.atan2(rel_right, rel_forward) + turn
        return (
            enemy.forward + enemy.velocity_forward * frames + radius * math.cos(angle),
            enemy.right + enemy.velocity_right * frames + radius * math.sin(angle),
        )

    def satellite_shot(self, action: int) -> tuple[ObservedObject, float] | None:
        """The satellite to shoot with ``action`` and the bearing to fire along.

        Prefers the weakest satellite in reach, then the soonest hit.
        """

        options = []
        for satellite in self.chmmr_satellites():
            solved = self.intercept(
                action, lambda frames, s=satellite: self.satellite_offset(s, frames)
            )
            if solved is not None:
                bearing, frames = solved
                options.append((satellite.hit_points, frames, bearing, satellite))
        if not options:
            return None
        _, _, bearing, satellite = min(options, key=lambda item: item[:2])
        return satellite, bearing

    def satellite_on_beam(
        self, bearing: float, reach: float, half_width: float = SATELLITE_HIT_RADIUS
    ) -> bool:
        """Whether a satellite will lie on a beam fired next frame along
        ``bearing``, within ``reach`` and ``half_width`` of the beam."""

        beam = math.radians(bearing)
        along_x, along_y = math.cos(beam), math.sin(beam)
        for satellite in self.chmmr_satellites():
            forward, right = self.satellite_offset(satellite, 1)
            along = forward * along_x + right * along_y
            across = -forward * along_y + right * along_x
            if 0.0 < along <= reach and abs(across) <= half_width:
                return True
        return False

    def satellites_in_reach_of_me(self) -> bool:
        """Whether a satellite can shoot this ship (and so ignores my shots)."""

        return any(
            satellite.distance <= SATELLITE_LASER_RANGE
            for satellite in self.chmmr_satellites()
        )

    def satellite_ring(self, reach: float, margin: float = 0.0) -> tuple[float, float] | None:
        """Distances from the Chmmr that are safe yet let ``reach`` hit a satellite.

        Inside the inner edge a satellite or the Chmmr laser reaches this
        ship; beyond the outer edge the weapon can't reach the near
        satellite. None when the weapon is too short for any safe spot.
        """

        inner = max(SATELLITE_ORBIT + SATELLITE_LASER_RANGE, CHMMR_LASER_RANGE) + margin
        outer = SATELLITE_ORBIT + float(reach) - margin
        if outer < inner:
            return None
        return inner, outer

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
        frames = self.behaviour.planet_horizon_frames or planet_horizon_frames(self.spec)
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
        thrust = spec.constant_speed or abs(heading) <= PLANET_ESCAPE_THRUST_ANGLE
        return self.steer(heading, thrust=thrust)

    def evade_threats(self) -> bool:
        """Rule 1: keep enemy shots in flight off the hull.

        Steps in only when the path the ship is on (no turning, thrust as
        held) is predicted to take a hit within the horizon, and some move
        does better. The moves are coasting, flying one of
        ``EVADE_HEADINGS`` (thrusting once the nose is within
        ``EVADE_THRUST_ANGLE`` of it), and turning to shoot one threat down
        with weapon 1 where the shot can destroy it before it lands (the Orz
        may turn its turret instead, keeping its course); the Druuge's shots
        also kick it backward. The Supox may also slide with
        weapon 2 held, the Umgah hops backward in place of headings, and the
        Slylandro turns, or turns about, at its constant speed. Moves are
        scored by the crew they lose, plus for a shot the battery it costs,
        and among equals the one turning the nose for the least time wins. A move, once chosen,
        is kept while nothing does better, and the rule lets go once neither the
        held path nor coasting comes within ``EVADE_RELEASE_MARGIN`` of a hit.
        """

        if self.plan.steering_claimed:
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
        stopped = self.stopped_threats(tracks)
        if stopped:
            tracks = [track for index, track in enumerate(tracks) if index not in stopped]
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
        options.append((coast.damage, 0.0, 0.0, (EVADE_COAST, 0), self._evade_coast))
        previous_kind = self.memory.sides.get("evade")
        previous_heading = self.memory.sides.get("evade_heading")
        moves = self._evade_moves(previous_kind, previous_heading, held_thrust, tracks)
        for plan, controller, seconds, claim in moves:
            _, outcome = judge(controller)
            options.append((outcome.damage, 0.0, seconds, plan, claim))
        options.extend(self._shoot_down_options(tracks, coast_path, judge))

        best = min(options, key=lambda option: (option[0] + option[1], option[2]))
        best_score = best[0] + best[1]
        if best_score >= baseline_damage and not active:
            return False
        if not active and best[3][0] not in (EVADE_SHOOT, EVADE_TURRET):
            # Leave the ship to its strategy while a dodge begun a little
            # later still does as well.
            hold = lambda flight: (None, held_thrust, False)
            for _, controller, _, _ in sorted(moves, key=lambda move: move[2]):
                _, outcome = judge(_delayed(hold, controller, EVADE_DELAY_FRAMES))
                if outcome.damage <= best_score:
                    return False
        if active:
            for option in options:
                kind, value = option[3]
                if kind != previous_kind:
                    same_plan = False
                elif kind == EVADE_HEADING:
                    same_plan = abs(wrap_degrees(value - float(previous_heading or 0))) < 1.0
                elif kind in (EVADE_SLIDE, EVADE_HOP):
                    same_plan = value == previous_heading
                else:
                    same_plan = True
                if same_plan and option[0] + option[1] <= best_score:
                    best = option
                    break
        kind, value = best[3]
        self.memory.touched_sides.update(("evade", "evade_heading"))
        self.memory.sides["evade"] = kind
        self.memory.sides["evade_heading"] = value
        return best[4]()

    def _evade_moves(self, previous_kind, previous_heading, held_thrust: bool, tracks=()):
        """Moves rule 1 weighs besides coasting and shooting: each is
        ``(plan, controller, seconds turning, claim)``.

        Ships fly the headings, and with ``run_from_threats`` also the
        heading straight away from the threats; the Supox also slides with
        weapon 2 held, and the Umgah hops backward instead.
        """

        moves = []
        own_heading = self.me.heading
        if self.behaviour.evade_headings:
            headings = {wrap_degrees(offset) for offset in EVADE_HEADINGS}
            if self.behaviour.run_from_threats:
                away = run_bearing(tracks)
                if away is not None:
                    headings.add(away)
            if previous_kind == EVADE_HEADING and previous_heading is not None:
                headings.add(wrap_degrees(float(previous_heading) - own_heading))
            constant_speed = self.spec.constant_speed
            for bearing in sorted(headings):
                absolute = int(round((own_heading + bearing) % 360.0))
                if constant_speed:
                    # Thrust turns the ship about, which ``steer`` presses
                    # only with the heading behind and thrust released.
                    controller = _turn_about_controller(bearing, not self.held("thrust"))
                    thrust = True
                else:
                    controller = _heading_controller(bearing)
                    thrust = abs(bearing) <= EVADE_THRUST_ANGLE
                moves.append(
                    (
                        (EVADE_HEADING, absolute),
                        controller,
                        self.turn_seconds(bearing),
                        lambda bearing=bearing, thrust=thrust: self.steer(
                            bearing, thrust=thrust
                        ),
                    )
                )
        if self.ship_name == "Supox":
            for index, (turn, backward, angle) in enumerate(SUPOX_SLIDES):
                moves.append(
                    (
                        (EVADE_SLIDE, index),
                        _slide_controller(angle),
                        0.0,
                        lambda turn=turn, backward=backward: self._supox_slide(
                            turn, backward
                        ),
                    )
                )
        if self.ship_name == "Umgah":
            ready = int(
                math.ceil(
                    max(
                        self.me.timer_seconds("thrust_timer"),
                        self.me.timer_seconds("a2_timer"),
                        0.0,
                    )
                    * const.FPS
                )
            )
            for count in EVADE_HOP_COUNTS:
                if self.me.battery >= count * self.spec.a2_cost:
                    moves.append(
                        ((EVADE_HOP, count), _hop_controller(count, ready), 0.0, self._hop)
                    )
        return moves

    def _supox_slide(self, turn: int, backward: bool) -> bool:
        self.plan.claim_a2(True)
        self.plan.claim_turn(turn)
        self.plan.claim_thrust(backward)
        return True

    def _hop(self) -> bool:
        self.plan.claim_turn(TURN_NONE)
        self.plan.claim_thrust(False)
        self.plan.claim_a2(True)
        return True

    def turn_seconds(self, bearing: float) -> float:
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

        if not self.behaviour.shoot_down:
            return []
        weapon = self.weapon(1)
        laser = weapon.kind == "laser" and not weapon.tracking
        if not laser and not (
            weapon.kind == "projectile" and not weapon.tracking and weapon.speed > 0.0
        ):
            return []
        held_value = self.held_shot_value()
        ready_frame = int(math.ceil(max(0.0, self.me.timer_seconds("a1_timer")) * const.FPS))
        if self.ship_name == "Melnorme":
            # The shot is the charge, flown on release at its current damage.
            charge = self.held_shot() if held_value is not None else None
            if charge is None:
                return []
            weapon = replace(weapon, damage=max(0.0, -charge.crew_effect))
            ready_frame = 0
        elif self.me.battery < self.spec.a1_cost:
            return []
        elif held_value is not None:
            # A1 is released this frame; the new shot leaves on the next.
            ready_frame = max(ready_frame, 1)
        # Pre-empting a held shot gives up what it is worth.
        forgone = held_value or 0.0
        recoil = DRUUGE_RECOIL if self.ship_name == "Druuge" else None
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
                    cost + forgone,
                    self.turn_seconds(nose),
                    (EVADE_SHOOT, int(round(absolute % 360.0))),
                    lambda nose=nose: self._shoot_down(nose),
                )
            )
        if self.ship_name == "Orz":
            options.extend(
                self._turret_shoot_down_options(tracks, coast_path, judge, ready_frame)
            )
        return options

    def _turret_shoot_down_options(self, tracks, coast_path, judge, ready_frame):
        """Options that turn the Orz turret, not the hull, to fire weapon 1
        at one threat; the hull coasts on, keeping its heading for the
        strategy, so among equal outcomes these beat turning the hull."""

        weapon = self.weapon(1)
        turret = self.me.orz_turret_bearing
        turret_ready = int(
            math.ceil(max(0.0, self.me.timer_seconds("a2_timer")) * const.FPS)
        )
        cost = EVADE_SHOT_COST * self.spec.a1_cost / max(1.0, self.me.max_battery)
        options = []
        for index, track in enumerate(tracks):
            if weapon.damage < track.hit_points:
                continue
            aim = _lead_bearing(track, coast_path, weapon, 0)
            if aim is None:
                continue
            fire_frame = None
            for _ in range(2):
                # Refine once: lead the threat from the frame the shot leaves.
                steps = int(round(wrap_degrees(aim - turret) / ORZ_TURRET_STEP))
                if steps == 0:
                    break
                fire_frame = max(
                    ready_frame,
                    turret_ready + (abs(steps) - 1) * ORZ_TURRET_STEP_FRAMES + 1,
                )
                if fire_frame >= len(coast_path.xs) - 1:
                    fire_frame = None
                    break
                refined = _lead_bearing(track, coast_path, weapon, fire_frame)
                if refined is None:
                    break
                aim = refined
            if fire_frame is None or steps == 0:
                # Already on it: the hull option with the nose kept covers it.
                continue
            direction = wrap_degrees(turret + steps * ORZ_TURRET_STEP)

            def controller(flight, fire_frame=fire_frame):
                return None, False, flight.frame == fire_frame

            _, outcome = judge(
                controller, shot=(fire_frame, weapon, direction, index)
            )
            options.append(
                (
                    outcome.damage,
                    cost,
                    0.0,
                    (EVADE_TURRET, int(round((self.me.heading + direction) % 360.0))),
                    lambda direction=direction: self._turret_shoot_down(direction),
                )
            )
        return options

    def _turret_shoot_down(self, bearing: float) -> bool:
        self.plan.claim_thrust(False)
        if self.turn_turret(bearing):
            self.plan.claim_turn(TURN_NONE)
            if self.me.a1_ready:
                self.a1(True)
                self.a2(False)
        return True

    def _shoot_down(self, nose: float) -> bool:
        self.point_at(nose)
        self.plan.claim_thrust(False)
        aligned = abs(nose) <= float(const.TURN_ANGLE) / 2.0 + 1e-6
        if self.ship_name == "Melnorme":
            # Keep charging until the nose is on, then let the charge fly.
            self.a1(not aligned)
            return True
        if self.ship_name in HELD_WEAPON_SHIPS:
            # A held crystal or saw is let go first; A1 stays released until
            # the nose is on, so nothing else fires it early.
            self.a1(aligned and self.me.a1_ready and not self.held("a1"))
            return True
        if aligned and self.me.a1_ready:
            self.a1(True)
            if self.ship_name == "Orz":
                # Weapon 1 with weapon 2 launches a marine instead.
                self.a2(False)
        return True

    def keep_moving(self) -> bool:
        """Last rule: a ship nothing else moves builds speed.

        Runs after the strategy, and only when no rule claimed thrust and the
        ship is below ``KEEP_MOVING_SPEED_FRACTION`` of top speed. It heads
        across the line to the opponent, on the side the ship already drifts
        toward, so the build-up neither closes in nor backs off. With turning
        unclaimed it steers there. For ships whose behaviour lets it take
        aim (``keep_moving_takes_aim``), with turning
        claimed only to aim, it takes the turn back while weapon 1 could not
        fire this frame anyway; once at speed the ship coasts and the
        strategy aims again.
        """

        enemy = self.enemy
        if (
            enemy is None
            or self.plan.thrust is not None
            or not self.behaviour.keep_moving
        ):
            return False
        spec = self.spec
        if spec.inertial and self.me.speed >= KEEP_MOVING_SPEED_FRACTION * spec.max_thrust:
            return False
        across = math.radians(enemy.bearing + 90.0)
        lateral = self.me.velocity_forward * math.cos(across) + self.me.velocity_right * math.sin(
            across
        )
        key = "keep_moving"
        self.memory.touched_sides.add(key)
        if abs(lateral) > ORBIT_DRIFT_THRESHOLD * max(1.0, spec.max_thrust):
            self.memory.sides[key] = 1 if lateral > 0.0 else -1
        heading = self.side(key, enemy.bearing, 90.0)
        thrust = abs(heading) <= KEEP_MOVING_THRUST_ANGLE
        if self.plan.steering_claimed:
            if not self.behaviour.keep_moving_takes_aim or not self._aim_can_wait():
                return False
            self.plan.turn = None
        return self.steer(heading, thrust=thrust)

    def _aim_can_wait(self) -> bool:
        """Whether weapon 1 could not fire now even if aimed."""

        if self.plan.a1:
            return False
        return (
            not self.in_range(1)
            or not self.me.a1_ready
            or self.me.battery < self.spec.a1_cost
        )

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

    def closing_reach(self, key: str, seconds: float) -> float:
        """Distance the opponent will be at after ``seconds``, closing as
        it now is and speeding up at its smoothed rate (tracked under ``key``)."""

        enemy = self.enemy
        values = self.memory.values
        previous = values.get(f"{key}_closing_speed", enemy.closing_speed)
        values[f"{key}_closing_speed"] = enemy.closing_speed
        accel = (1.0 - STANDOFF_ACCEL_SMOOTHING) * values.get(f"{key}_closing_accel", 0.0)
        accel += STANDOFF_ACCEL_SMOOTHING * (enemy.closing_speed - previous)
        values[f"{key}_closing_accel"] = accel
        frames = seconds * const.FPS
        return enemy.distance - (
            max(0.0, enemy.closing_speed) * frames + 0.5 * max(0.0, accel) * frames * frames
        )

    def keep_standoff(
        self,
        key: str,
        near: float,
        far: float,
        *,
        release: float | None = None,
        approach_speed: float = STANDOFF_APPROACH_SPEED,
    ) -> bool:
        """Hold the opponent between ``near`` and ``far``; True while running.

        Runs straight away from it once it would come within ``near`` before
        the nose could turn away, until it is beyond ``release`` (default
        midway) or clearly falling back. Closes in from beyond ``far`` at an
        angle, thrusting only below ``approach_speed``. Inside the band
        steering is left free, for aiming.
        """

        enemy = self.enemy
        if enemy is None:
            return False
        values = self.memory.values
        away = wrap_degrees(enemy.bearing + 180.0)
        reach = self.closing_reach(key, self.turn_seconds(away) + STANDOFF_TURN_MARGIN)
        release = (near + far) / 2.0 if release is None else float(release)
        running_key = f"{key}_running"
        if enemy.closing_speed > 0.0 and reach <= near:
            values[running_key] = True
        elif enemy.distance > release or (enemy.closing_speed < 0.0 and reach > near):
            values[running_key] = False
        if values.get(running_key, False):
            self.steer(away)
            return True
        if enemy.distance > far:
            heading = self.drift_cancelling_side(f"{key}_far", enemy, APPROACH_ANGLE_OFFSET)
            self.steer(heading, thrust=enemy.closing_speed < approach_speed)
        return False

    def recharging(self) -> bool:
        """Whether the battery is below the ship's ``recharge_battery``."""

        threshold = self.behaviour.recharge_battery
        return threshold is not None and self.me.battery < threshold

    def recharge_retreat(self, key: str = "recharge") -> bool:
        """Keep out of the opponent's weapon reach while recharging.

        Applies while the battery is below the ship's ``recharge_battery``
        (``behaviours.SHIP_BEHAVIOURS``): it runs once the opponent would
        come within its reach plus a hull and ``RECHARGE_REACH_MARGIN``
        before the nose could turn away, as ``keep_standoff`` does, and
        never closes in. True while running.
        """

        if self.enemy is None or not self.recharging():
            self.memory.values[f"{key}_running"] = False
            return False
        near = self.enemy_reach() + TYPICAL_HULL_RADIUS + RECHARGE_REACH_MARGIN
        return self.keep_standoff(
            key, near, math.inf, release=near + RECHARGE_RELEASE_MARGIN
        )

    def hold_band(
        self,
        band: tuple[float, float] | None = None,
        key: str = "band",
    ) -> bool:
        """Hold the opponent within a distance band, by ``keep_standoff``.

        The band is ``band`` or else the ship's ``band``
        (``behaviours.SHIP_BEHAVIOURS``); with neither, it does nothing.
        True while running from a closer.
        """

        band = band or self.behaviour.band
        if band is None:
            return False
        near, far = band
        return self.keep_standoff(key, float(near), float(far))

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


def turn_about_frames(spec: ShipSpec) -> int:
    """Frames the ship takes to turn its nose 180 degrees, as
    ``flight.ShipFlight`` turns it: the first step at once, then one every
    ``turn_wait + 1`` frames."""

    steps = math.ceil(180.0 / turn_step_degrees() - 1e-9)
    return (steps - 1) * (spec.turn_wait + 1) + 1


def top_speed_frames(spec: ShipSpec) -> int:
    """Frames of thrust from rest to top speed, as ``flight.ShipFlight``
    pushes it: 0 for ships without inertia or of constant speed, which fly
    at top speed whenever they move."""

    if not spec.inertial or spec.constant_speed or spec.thrust_increment <= 0.0:
        return 0
    pushes = math.ceil(spec.max_thrust / spec.thrust_increment - 1e-9)
    return (pushes - 1) * (spec.thrust_wait + 1) + 1


def planet_horizon_frames(spec: ShipSpec) -> int:
    """Frames ahead rule 0 follows the ship's path past the planet."""

    frames = turn_about_frames(spec) + top_speed_frames(spec) + PLANET_HORIZON_MARGIN_FRAMES
    low = int(round(PLANET_HORIZON_MIN_SECONDS * const.FPS))
    high = int(round(PLANET_HORIZON_MAX_SECONDS * const.FPS))
    return max(low, min(high, frames))


def held_pass_share(shot: ObservedObject, target: ObservedObject, reach: float) -> float:
    """Share of its damage a shot flying straight on is expected to deal
    ``target``, a hull of ``reach``: all of it when its closest pass comes
    within ``reach``, falling to none at twice that; none when that pass is
    behind it or more than ``HELD_SHOT_HORIZON_FRAMES`` away."""

    offset_forward = shot.forward - target.forward
    offset_right = shot.right - target.right
    relative_forward = shot.velocity_forward - target.velocity_forward
    relative_right = shot.velocity_right - target.velocity_right
    speed_squared = relative_forward * relative_forward + relative_right * relative_right
    if speed_squared <= 1e-9:
        return 0.0
    closest = -(offset_forward * relative_forward + offset_right * relative_right) / speed_squared
    if closest <= 0.0 or closest > HELD_SHOT_HORIZON_FRAMES:
        return 0.0
    miss = math.hypot(
        offset_forward + relative_forward * closest,
        offset_right + relative_right * closest,
    )
    reach = max(1.0, float(reach))
    return max(0.0, min(1.0, (2.0 * reach - miss) / reach))


def run_bearing(tracks) -> float | None:
    """Bearing straight away from ``tracks``, each pulling by the crew it
    costs over its distance; None when they cancel out."""

    forward = right = 0.0
    for track in tracks:
        distance = math.hypot(track.x, track.y)
        if distance <= 0.0:
            continue
        pull = track.weight / distance
        forward -= pull * track.x / distance
        right -= pull * track.y / distance
    if math.hypot(forward, right) <= 1e-9:
        return None
    return wrap_degrees(bearing_of(forward, right))


def _heading_controller(bearing: float):
    """Fly toward ``bearing`` in the starting frame, thrusting once the nose
    is within ``EVADE_THRUST_ANGLE`` of it."""

    def controller(flight: ShipFlight):
        error = flight.steering_toward(bearing)
        return error, abs(error) <= EVADE_THRUST_ANGLE, False

    return controller


def _turn_about_controller(bearing: float, fresh_press: bool):
    """Fly a ship of constant speed toward ``bearing``, first turning it
    about when the heading lies behind and thrust is free to press."""

    def controller(flight: ShipFlight):
        error = flight.steering_toward(bearing)
        turn_about = fresh_press and flight.frame == 0 and abs(error) > 135.0
        return error, turn_about, False

    return controller


def _slide_controller(angle: float):
    """Push ``angle`` degrees off the nose every frame, without turning."""

    def controller(flight: ShipFlight):
        flight.thrust_angle = angle
        return None, True, False

    return controller


def _hop_controller(count: int, ready_frame: int):
    """Hop backward ``count`` times: from the frame the hop is ready, once
    every ``UMGAH_HOP_FRAMES``."""

    def controller(flight: ShipFlight):
        if (
            flight.hops < count
            and flight.frame >= ready_frame
            and (flight.hops == 0 or flight.frame - flight.last_hop_frame >= UMGAH_HOP_FRAMES)
        ):
            return None, False, False, UMGAH_HOP_DISTANCE
        return None, False, False

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
            if spec.constant_speed:
                # Thrust turns it about, which ``steer`` presses only with
                # the heading behind.
                thrust = frame == 0 and abs(error) > 135.0
            else:
                thrust = abs(error) <= PLANET_ESCAPE_THRUST_ANGLE
            flight.step(error, thrust)
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
