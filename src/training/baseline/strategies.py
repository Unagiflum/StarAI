"""Per-ship baseline strategies.

Each function lists one ship's rules in priority order (see ``engine``). The
shared rules run before every strategy: rule 0 avoids the planet, and rule 1
dodges or shoots down enemy shots in flight, where ``EVADE_GATES`` allows.
After the strategy, the last rule builds speed for a ship nothing moves.
"""

from __future__ import annotations

import math
from typing import Callable

import src.const as const
from src.training.baseline.engine import (
    APPROACH_ANGLE_OFFSET,
    AVOID_ANGLE_OFFSET,
    PERPENDICULAR_AWAY_OFFSET,
    ORZ_TURRET_STEP,
    SATELLITE_HIT_RADIUS,
    TURN_LEFT,
    TURN_NONE,
    TURN_RIGHT,
    UMGAH_HOP_DISTANCE,
    UMGAH_HOP_FRAMES,
    BaselineContext,
    StrategyFunction,
    _smallest_positive_root,
)
from src.training.baseline.observation_view import bearing_of, wrap_degrees
from src.training.baseline.ship_specs import (
    FORM_BASE,
    FORM_BLAZER,
    FORM_XFORM,
    FORM_YWING,
    TYPICAL_HULL_RADIUS,
    shofixti_kill_distance,
    weapon_spec,
)
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS

# Collision windows from the specification, in seconds. The engine adds one
# action span to each so a reaction is never a decision late.
ARILOU_TELEPORT_SECONDS = 0.1
SHIELD_SECONDS = 0.2
THRADDASH_AFTERBURNER_SECONDS = 0.5
UMGAH_CONE_THREAT_SECONDS = 1.0
UMGAH_RETRO_SECONDS = 0.2
# Umgah hop tactics: battery kept for dodges, how far beside the opponent
# the hops pass, how well the tail must line up before hopping, the cone's
# useful half-angle, and the reach of hostile shots that keep thrust off.
UMGAH_HOP_RESERVE = 5.0
UMGAH_PASS_CLEARANCE = 80.0
UMGAH_HOP_ALIGN_TOLERANCE = 10.0
UMGAH_CONE_HALF_ANGLE = 40.0
UMGAH_HOSTILE_RANGE = 1200.0
# Ahead and this close, thrust in rather than turn around to hop.
UMGAH_CLOSE_APPROACH_RANGE = 400.0
# Hop runs: each hop moves ``UMGAH_HOP_DISTANCE``, one every
# ``UMGAH_HOP_FRAMES`` (both from the engine). Within the lateral-aim
# distance a run must pass the opponent this far to the side (within the
# lateral limits, beyond which the cone misses or the hulls bump); farther
# off, pointing the tail near it is enough. Enemy objects
# within the corridor half-width of the path block a run, and an opponent
# whose nose points within the facing angle of me is passed on the side
# away from its nose.
UMGAH_LATERAL_AIM_DISTANCE = 1000.0
UMGAH_RUN_CLEARANCE = 90.0
UMGAH_RUN_MIN_LATERAL = 40.0
UMGAH_RUN_MAX_LATERAL = 160.0
UMGAH_CORRIDOR_HALF_WIDTH = 2.0 * TYPICAL_HULL_RADIUS + 10.0
# Hops ahead checked for a clear path, even when hopping only once.
UMGAH_PATH_LOOKAHEAD_HOPS = 4
UMGAH_FACING_ANGLE = 60.0
# Short-reach opponents that must come to the cone; the Umgah waits facing them.
UMGAH_AMBUSH_SHIPS = frozenset({"Ilwrath", "Pkunk", "Shofixti"})
# Distances from the specification, in world units.
LONG_RANGE = 1000.0
FAR_RANGE = 2000.0
# Earthling standoff: hold the opponent between the band's edges, inside nuke
# range; nukes need this much room to turn onto the target and not catch the
# Earthling, and launch within the tolerance as they home. Whip only off a
# planet at least this far around from the opponent. Point defense also
# fires at objects within this fraction of its reach even when not closing.
EARTHLING_BAND_NEAR = 1200.0
EARTHLING_BAND_FAR = 2000.0
EARTHLING_NUKE_MIN_DISTANCE = 400.0
EARTHLING_NUKE_TOLERANCE = 30.0
EARTHLING_WHIP_AWAY_ANGLE = 120.0
EARTHLING_PD_CLOSE_FRACTION = 0.5
DRUUGE_ESCAPE_RANGE = 500.0
# Ships whose fighters, Dogi or limpets the Druuge recoils away from.
DRUUGE_ESCAPE_DANGER_SHIPS = frozenset({"KzerZa", "Chenjesu", "Vux"})
# Ships whose hulls the Kzer-Za fighters should not be wasted on.
KZER_ZA_FIGHTER_COUNTERS = frozenset({"Earthling", "Thraddash"})
TRACTOR_IMMUNE_SHIPS = frozenset({"Arilou", "Slylandro"})
# Pkunk shots live only briefly: fire within this fraction of their reach
# unless the opponent is closing, and wait out recharges just beyond reach.
PKUNK_FIRE_RANGE_FRACTION = 0.7
PKUNK_STATION_RANGE_FACTOR = 1.27
# Recharging, the Pkunk also waits this far beyond the opponent's reach.
# Passes aim this far beside the opponent; within gun range it thrusts only
# while within the tolerance of side-on. An opponent whose nose points
# within the facing angle of the Pkunk is passed away from its nose, and
# hits due within the dodge window are sidestepped.
PKUNK_STATION_MARGIN = 150.0
PKUNK_PASS_OFFSET = 250.0
PKUNK_SIDE_ON_TOLERANCE = 30.0
PKUNK_FACING_ANGLE = 60.0
PKUNK_DODGE_SECONDS = 0.4
# Zoq-Fot-Pik dives for a tongue strike it can make within this many seconds.
ZOQ_FOT_PIK_LUNGE_SECONDS = 1.0
ZOQ_FOT_PIK_TONGUE_WIDTH = float(ABILITY_DEFINITIONS["ZoqFotPikA2"].area_width or 10)
ZOQ_FOT_PIK_TONGUE_SKIPPED_SHIPS = frozenset({"Slylandro"})
# Ilwrath flames, measured on the real collision masks: held flames reach a
# hull about 300 units ahead centre to centre, and a hull about 60 units to
# the side of the nose line. The cloak takes this many frames to fade in,
# and is used only with battery left for this many flames afterwards.
ILWRATH_FLAME_REACH = 300.0
ILWRATH_FLAME_HALF_WIDTH = 60.0
ILWRATH_FADE_FRAMES = float(ABILITY_DEFINITIONS["IlwrathA2"].fade_duration or 5)
ILWRATH_CLOAK_RESERVE = 4.0
# Thraddash: hold a standoff band and fire from it; burn away, tail first,
# from an opponent inside the escape distance or due to arrive within the
# escape time. Slow turners are baited in nearer so they chase into the
# trail. A battery reserve stays for that burn. Escorts closing within the
# swarm range get the tail, and the trail is laid once they are within the
# trail range. Shots from this far off the nose are dodged with a burn.
THRADDASH_STANDOFF_NEAR = 900.0
THRADDASH_STANDOFF_FAR = 1400.0
THRADDASH_ESCAPE_DISTANCE = 700.0
THRADDASH_BAIT_DISTANCE = 500.0
THRADDASH_ESCAPE_SECONDS = 1.5
THRADDASH_BURN_RESERVE = 8.0
THRADDASH_TAIL_TOLERANCE = 30.0
THRADDASH_SWARM_RANGE = 700.0
THRADDASH_TRAIL_RANGE = 350.0
THRADDASH_DODGE_MIN_BEARING = 45.0
# Escape and bait distances stay this far outside the opponent's reach.
THRADDASH_REACH_MARGIN = 100.0
# Dash in on the afterburner when this far beyond the standoff band, with
# the nose within the alignment of the approach heading, up to the dash
# speed, keeping the dash cost above the reserve.
THRADDASH_DASH_MARGIN = 600.0
THRADDASH_DASH_ALIGN = 15.0
THRADDASH_DASH_SPEED = 60.0
THRADDASH_DASH_COST = 6.0
THRADDASH_SLOW_TURN_WAIT = 3
# Arilou against a Chmmr: hold this band, just beyond the Chmmr laser and
# within laser reach of the near satellite, and fire while a satellite lies
# on the beam (half the satellite's width either side of it).
ARILOU_CHMMR_HOLD = 650.0
ARILOU_CHMMR_FAR = 700.0
# The beam stops at a satellite's edge, a little beyond its centre's reach.
ARILOU_BEAM_MARGIN = -15.0
# Against a Chmmr: hold this band outside the satellites' and the Chmmr
# laser's reach, shooting satellites first; inside the escape distance burn
# away, trailing flames between us, until clear of the release distance.
THRADDASH_CHMMR_NEAR = 1000.0
THRADDASH_CHMMR_FAR = 1450.0
THRADDASH_CHMMR_ESCAPE = 900.0
THRADDASH_CHMMR_RELEASE = 1300.0
# The escape starts early enough to turn tail before the Chmmr, closing at
# its current speed and still speeding up as it is, reaches the escape
# distance; with this much time to spare.
THRADDASH_CHMMR_TURN_MARGIN = 0.5
# Never thrust toward the Chmmr while closing faster than this: its tractor
# beam adds speed on top.
THRADDASH_CHMMR_APPROACH_SPEED = 15.0
# The tractor pulls in pulses, so the closing acceleration is smoothed with
# this weight on each new frame.
THRADDASH_CHMMR_ACCEL_SMOOTHING = 0.1
THRADDASH_SLOW_TURNERS = frozenset(
    name
    for name, ship in SHIP_DEFINITIONS.items()
    if int(ship.turn_wait) >= THRADDASH_SLOW_TURN_WAIT
)
# Orz marines only feed a Thraddash trail.
ORZ_MARINE_COUNTERS = frozenset({"Thraddash"})
# Against a Chmmr, marines launch only once this many satellites or fewer
# are left to shoot them down.
ORZ_CHMMR_MARINE_SATELLITES = 1
# Standoff band from a Chmmr, beyond its reach and within turret reach of a
# satellite.
ORZ_CHMMR_NEAR = 900.0
ORZ_CHMMR_FAR = 1250.0
# Opponents whose shots are too tough to shoot down and must be dodged:
# the Orz aims with the hull alone, its turret kept as it is, which the
# arena shows dodges them better.
ORZ_HULL_AIM_OPPONENTS = frozenset({"Chenjesu", "KohrAh"})
# Androsynth against a Chmmr: blaze in from this range once the battery is
# at least this full.
ANDROSYNTH_CHMMR_BLAZE_RANGE = 800.0
ANDROSYNTH_CHMMR_BLAZE_BATTERY = 0.9
# Opponents the Orz kites, as the arena favours: it holds them in a band,
# running hull away from one closing in while the turret fires back.
ORZ_KITE_OPPONENTS = frozenset(
    {"Arilou", "Druuge", "KzerZa", "Orz", "Pkunk", "Shofixti", "Spathi", "Syreen", "Yehat"}
)
ORZ_KITE_NEAR = 700.0
ORZ_KITE_FAR = 1100.0
# Supox: hold a distance band, by default 600-900, or just outside a
# shorter-reaching opponent's weapons. With the gun within the slide aim
# tolerance it holds A2 to slide instead of turning; it dodges hits due
# within the dodge window and slides off the nose line of an opponent
# pointing within the facing tolerance.
SUPOX_BAND_NEAR = 600.0
SUPOX_BAND_FAR = 900.0
SUPOX_BAND_GUN_FRACTION = 0.95
SUPOX_BAND_MIN_WIDTH = 150.0
SUPOX_OUTRANGE_MARGIN = 150.0
SUPOX_SLIDE_AIM_TOLERANCE = 6.0
SUPOX_DODGE_SECONDS = 0.5
SUPOX_FACING_TOLERANCE = 20.0
# Vux: an opponent counts as crippled once limpets leave it slower than the
# Vux and turning no more often than every this many frames. Until then the
# Vux moves away, retreating at this fraction of top speed before turning
# to face the opponent, and heading away again once below the resume
# fraction of that. Limpets keep the laser reserve near laser reach. The
# laser fires whenever the hull crosses the beam within its full reach (the
# arena favoured that over holding fire to 450 units). A crippled opponent
# is approached from beyond this angle off its nose, to the close distance.
VUX_DISABLED_TURN_WAIT = 3.0
VUX_RETREAT_SPEED_FRACTION = 0.8
VUX_RETREAT_RESUME_FRACTION = 0.5
VUX_LASER_RESERVE = 20.0
VUX_LASER_RESERVE_MARGIN = 200.0
VUX_BURST_DISTANCE = weapon_spec("Vux", 1).range + TYPICAL_HULL_RADIUS
VUX_BLIND_ANGLE = 60.0
VUX_CLOSE_DISTANCE = 300.0
VUX_LIMPET_REACH = weapon_spec("Vux", 2).range
VUX_SLINGSHOT_DISTANCE = 2500.0
# Holding the beam: keep facing a target within the beam tolerance plus this
# slack (about one heading step) while the battery lasts above this floor.
VUX_HOLD_BEAM_SLACK = 6.0
VUX_HOLD_BEAM_BATTERY = 4.0
# Syreen song: sing within this distance, where it takes 6 or more crew;
# below the grown crew count, a Syreen that cannot outrun an opponent with
# strong short-range weapons keeps to gun range instead of diving in. The
# arena showed caution costs fights against the Androsynth and Pkunk, so
# they are left out.
SYREEN_SONG_DISTANCE = 300.0
SYREEN_GROWN_CREW = 20.0
SYREEN_CLOSE_DANGER_SHIPS = frozenset(
    {"Ilwrath", "KohrAh", "Orz", "Shofixti", "Umgah", "ZoqFotPik"}
)
SONG_IMMUNE_SHIPS = frozenset(
    name
    for name, ship in SHIP_DEFINITIONS.items()
    if getattr(ship, "immune_to_psychic", False)
)
# Shofixti glory device. A kill blast must still kill if the opponent opens
# the gap for this many frames first, with a margin for the blast centre
# sitting off the hull centre. A hit due within the fatal-hit window counts
# as about to land.
SHOFIXTI_BLAST_LEAD_FRAMES = 2.0
SHOFIXTI_BLAST_MARGIN = 10.0
SHOFIXTI_FATAL_HIT_SECONDS = 0.15
# Shield ships raise their shield against a Shofixti within at least this
# distance, half the blast's reach, where it removes 10 crew.
SHOFIXTI_SHIELD_DISTANCE = float(ABILITY_DEFINITIONS["ShofixtiA2"].range) / 2.0
# Instant-hit weapons give no warning shot. Those that do not track hit only
# when the opponent's nose points within this many degrees of the target.
INSTANT_HIT_AIM_TOLERANCE = 20.0
# Not counted as instant hits: the Shofixti blast is the opponent's own
# last resort, the Slylandro A2 only hits asteroids, and the Syreen song
# never takes the last crew.
NOT_INSTANT_HITS = frozenset({"ShofixtiA2", "SlylandroA2", "SyreenA2"})
# Lasers carried by escorts rather than the opponent's hull.
ESCORT_LASERS = {"Chmmr": "ChmmrSatelliteLaser", "KzerZa": "KzerZaA2Laser"}


def _approach_enemy_object(ctx: BaselineContext, objects) -> bool:
    nearest = min(objects, key=lambda item: item.distance, default=None)
    return ctx.approach_object(nearest)


def _hold_while_closing_on_enemy(ctx: BaselineContext) -> bool:
    """Keep weapon 1 held while my fastest shot still closes on the enemy."""

    enemy = ctx.enemy
    shots = ctx.view.friendly_a1_objects
    if enemy is None or not shots:
        return False
    own_forward = ctx.me.velocity_forward
    own_right = ctx.me.velocity_right
    shot = max(
        shots,
        key=lambda item: (
            (item.velocity_forward + own_forward) ** 2
            + (item.velocity_right + own_right) ** 2
        ),
    )
    offset_forward = shot.forward - enemy.forward
    offset_right = shot.right - enemy.right
    relative_forward = shot.velocity_forward - enemy.velocity_forward
    relative_right = shot.velocity_right - enemy.velocity_right
    return offset_forward * relative_forward + offset_right * relative_right < 0.0


def _shot_in_flight(ctx: BaselineContext, minimum_speed: float) -> bool:
    own_forward = ctx.me.velocity_forward
    own_right = ctx.me.velocity_right
    return any(
        (
            (shot.velocity_forward + own_forward) ** 2
            + (shot.velocity_right + own_right) ** 2
        )
        > minimum_speed * minimum_speed
        for shot in ctx.view.friendly_a1_objects
    )


def androsynth(ctx: BaselineContext) -> None:
    if ctx.chmmr_satellites():
        _androsynth_vs_chmmr(ctx)
        return
    blazer = ctx.me.androsynth_blazer
    battery = ctx.me.battery_fraction
    if blazer and battery > 0.10:
        ctx.approach_opponent()
    if battery < 0.10:
        ctx.avoid_opponent()
    if ctx.distance < LONG_RANGE and battery >= 0.50 and not blazer:
        ctx.a2(True)
    if ctx.distance >= LONG_RANGE:
        ctx.avoid_opponent()
    if battery >= 0.50:
        ctx.a1(True)


def _androsynth_vs_chmmr(ctx: BaselineContext) -> None:
    """Save the battery, then blaze into the Chmmr once and keep ramming.

    The blazer passes through satellites without harming them, and inside
    their ring the lasers take about a crew a frame, so the crew buys one
    ram: from close in, on a full battery, at where the Chmmr will be.
    """

    if ctx.me.androsynth_blazer:
        ctx.steer(_blazer_intercept_bearing(ctx))
        return
    ctx.point_at_opponent()
    if (
        ctx.distance <= ANDROSYNTH_CHMMR_BLAZE_RANGE
        and ctx.me.battery >= ANDROSYNTH_CHMMR_BLAZE_BATTERY * ctx.me.max_battery
    ):
        ctx.a2(True)


def _blazer_intercept_bearing(ctx: BaselineContext) -> float:
    """Bearing on which the blazer, at its constant speed, meets the
    opponent moving as it is; the opponent's bearing if it can't."""

    enemy = ctx.enemy
    speed = ctx.spec.max_thrust
    # The opponent's motion over the arena, in my frame.
    drift_forward = enemy.velocity_forward + ctx.me.velocity_forward
    drift_right = enemy.velocity_right + ctx.me.velocity_right
    frames = _smallest_positive_root(
        drift_forward * drift_forward + drift_right * drift_right - speed * speed,
        2.0 * (enemy.forward * drift_forward + enemy.right * drift_right),
        enemy.forward * enemy.forward + enemy.right * enemy.right,
    )
    if frames is None:
        return enemy.bearing
    return bearing_of(
        enemy.forward + drift_forward * frames, enemy.right + drift_right * frames
    )


def arilou(ctx: BaselineContext) -> None:
    if ctx.imminent_threat(ARILOU_TELEPORT_SECONDS) is not None:
        ctx.a2(True)
    if ctx.chmmr_satellites():
        _arilou_vs_chmmr(ctx)
        return
    ctx.approach_weapon_range(1)
    if ctx.in_range(1) and ctx.me.battery >= ctx.spec.a1_cost + ctx.spec.a2_cost:
        ctx.a1(True)


def _arilou_vs_chmmr(ctx: BaselineContext) -> None:
    """Hold just outside the Chmmr laser; fire as a satellite crosses the beam.

    The laser always aims at the Chmmr and stops at the first satellite in
    its path, so it hits one only while that satellite lies on the line
    within reach. The band is beyond the satellites' reach, and the tractor
    can't hold the Arilou.
    """

    enemy = ctx.enemy
    # Stop closing in early enough to turn the nose away before the Chmmr,
    # closing as fast as it now is, comes within the hold distance.
    away = wrap_degrees(enemy.bearing + 180.0)
    turn_frames = ctx.turn_seconds(away) * const.FPS
    stopping = max(0.0, enemy.closing_speed) * turn_frames
    if enemy.distance - stopping > ARILOU_CHMMR_FAR:
        ctx.steer(enemy.bearing)
    else:
        # The laser aims itself, so keep the nose away from the Chmmr and
        # back off only while it is nearer than the hold distance.
        ctx.steer(
            wrap_degrees(enemy.bearing + 180.0),
            thrust=enemy.distance < ARILOU_CHMMR_HOLD,
        )
    reach = ctx.weapon(1).range - ARILOU_BEAM_MARGIN
    blocked = ctx.satellite_on_beam(enemy.bearing, reach)
    if blocked and ctx.me.battery >= ctx.spec.a1_cost + ctx.spec.a2_cost:
        ctx.a1(True)


def chenjesu(ctx: BaselineContext) -> None:
    if ctx.me.battery >= ctx.spec.a2_cost and ctx.me.chenjesu_dogis < 4:
        ctx.a2(True)
    ctx.avoid_opponent()
    ctx.point_weapon(1)
    if ctx.held("a1"):
        # The crystal bursts on release: hold it while it still closes.
        ctx.a1(_hold_while_closing_on_enemy(ctx))
    elif ctx.pointing(1):
        ctx.a1(True)


def chmmr(ctx: BaselineContext) -> None:
    ctx.approach_opponent()
    if (
        ctx.distance > LONG_RANGE
        and ctx.battery_at_least(0.30)
        and ctx.enemy_name not in TRACTOR_IMMUNE_SHIPS
        and not ctx.enemy_cloaked()
    ):
        ctx.a2(True)
    if ctx.pointing(1) and ctx.in_range(1):
        ctx.a1(True)


def druuge(ctx: BaselineContext) -> None:
    dangers = (
        [
            danger
            for danger in ctx.view.objects("enemy_non_a1")
            if danger.distance < DRUUGE_ESCAPE_RANGE
        ]
        if ctx.enemy_name in DRUUGE_ESCAPE_DANGER_SHIPS
        else []
    )
    if any(danger.closing_speed > 0.0 for danger in dangers) and all(
        # Recoil pushes the ship backward, so every danger must sit ahead.
        abs(danger.bearing) <= 90.0
        for danger in dangers
    ):
        ctx.a1(True)
    targets = [ctx.enemy] if ctx.enemy is not None else []
    if ctx.enemy_name == "Chenjesu":
        targets.extend(ctx.view.objects("enemy_non_a1"))
    target = min(targets, key=lambda item: item.distance, default=None)
    if target is not None:
        if target is ctx.enemy:
            ctx.point_weapon(1)
            aimed = ctx.pointing(1)
        else:
            ctx.point_at(target.bearing)
            tolerance = max(
                ctx.weapon(1).pointing_tolerance,
                ctx.memory.tap_degrees / 2.0,
            )
            aimed = abs(wrap_degrees(target.bearing)) <= tolerance
        if aimed and target.distance <= ctx.weapon(1).range:
            ctx.a1(True)
    if ctx.me.battery < ctx.spec.a1_cost:
        ctx.a2(True)


def _earthling_point_defense(ctx: BaselineContext) -> None:
    """Fire point defense only at something it can reach and that matters.

    It spares the Earthling's own nukes, but every shot costs battery the
    next nuke needs, so leave alone objects already flying away.
    """

    reach = ctx.weapon(2).range + TYPICAL_HULL_RADIUS
    enemy = ctx.enemy
    if enemy is not None and enemy.distance <= reach:
        ctx.a2(True)
        return
    for obj in ctx.view.enemy_objects:
        if obj.distance <= reach and (
            obj.closing_speed > 0.0 or obj.distance <= reach * EARTHLING_PD_CLOSE_FRACTION
        ):
            ctx.a2(True)
            return


def _earthling_whip(ctx: BaselineContext) -> bool:
    """Gravity-whip only when the planet lies away from the opponent.

    Pass the planet on the side that leaves heading further from the
    opponent, so the speed gained carries the ship away rather than at it.
    """

    planet = ctx.planet
    enemy = ctx.enemy
    if planet is None or enemy is None or planet.distance <= const.GRAVITY_RANGE:
        return False
    if ctx.me.speed >= ctx.spec.max_thrust:
        return False
    if abs(wrap_degrees(planet.bearing - enemy.bearing)) < EARTHLING_WHIP_AWAY_ANGLE:
        return False
    heading = max(
        (
            wrap_degrees(planet.bearing + APPROACH_ANGLE_OFFSET),
            wrap_degrees(planet.bearing - APPROACH_ANGLE_OFFSET),
        ),
        key=lambda bearing: abs(wrap_degrees(bearing - enemy.bearing)),
    )
    return ctx.steer(heading)


def earthling(ctx: BaselineContext) -> None:
    """Keep a nuke-range standoff, whipping off the planet away from the opponent.

    Above top speed it coasts, thrusting only to bend its path; it turns
    quickly enough to face the opponent for a nuke and turn back without
    losing speed.
    """

    ctx.coast_above_top_speed = True
    _earthling_point_defense(ctx)
    enemy = ctx.enemy
    if enemy is None:
        return
    nuke_range = ctx.weapon(1).range
    distance = enemy.distance
    can_nuke = (
        ctx.me.battery >= ctx.spec.a1_cost
        and ctx.me.a1_ready
        and EARTHLING_NUKE_MIN_DISTANCE <= distance <= nuke_range
    )
    if can_nuke:
        ctx.point_weapon(1)
        ctx.plan.claim_thrust(False)
        if ctx.pointing(1, tolerance=EARTHLING_NUKE_TOLERANCE):
            ctx.a1(True)
        return
    if _earthling_whip(ctx):
        return
    toward = _own_speed_toward(ctx, enemy)
    if distance < EARTHLING_BAND_NEAR:
        ctx.steer(ctx.side("earthling_away", enemy.bearing, AVOID_ANGLE_OFFSET))
    elif distance <= EARTHLING_BAND_FAR:
        if toward > 0.0 or enemy.closing_speed > 0.0:
            # Bend the path so the opponent stays off to the side.
            ctx.steer(ctx.side("earthling_away", enemy.bearing, PERPENDICULAR_AWAY_OFFSET))
        else:
            ctx.point_at_opponent()
            ctx.plan.claim_thrust(False)
    elif enemy.closing_speed > 0.0:
        # It comes to me: face it for the next nuke.
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(False)
    else:
        ctx.approach_range(EARTHLING_BAND_FAR, key="earthling_band")


def earthling_flee_whip(ctx: BaselineContext) -> None:
    """Earlier rules: flee a closing opponent, whip whenever slow."""

    # Point defense spares its own new nukes, so it can stay on.
    ctx.a2(True)
    if ctx.distance <= FAR_RANGE:
        ctx.avoid_opponent()
    if ctx.me.speed < ctx.spec.max_thrust:
        ctx.gravity_whip()
    ctx.point_at_opponent()
    if ctx.pointing(1, tolerance=45.0) and ctx.in_range(1):
        ctx.a1(True)


def _ilwrath_flame_tolerance(ctx: BaselineContext) -> float:
    """Aim error at which the flame stream still reaches the opponent's hull."""

    return math.degrees(math.atan2(ILWRATH_FLAME_HALF_WIDTH, max(1.0, ctx.distance)))


def _ilwrath(ctx: BaselineContext, *, hang_back: bool) -> None:
    enemy = ctx.enemy
    if enemy is None:
        return
    cloaked = not ctx.me.trackable
    fading = ctx.me.cloak_transition_direction != 0
    in_reach = ctx.distance <= ILWRATH_FLAME_REACH
    tolerance = _ilwrath_flame_tolerance(ctx)
    if cloaked and in_reach:
        # The first flame from the cloak turns the nose onto the opponent.
        ctx.a1(True)
    elif in_reach and ctx.pointing(1, tolerance=tolerance):
        ctx.a1(True)
    # Cloak only while the fade can finish before the opponent is in reach,
    # with battery left for a burst of flames afterwards.
    fade_frames = ILWRATH_FADE_FRAMES + ctx.span_frames
    cloak_distance = ILWRATH_FLAME_REACH + max(0.0, enemy.closing_speed) * fade_frames
    if (
        not cloaked
        and not fading
        and ctx.distance > cloak_distance
        and ctx.me.battery >= ctx.spec.a2_cost + ILWRATH_CLOAK_RESERVE
    ):
        ctx.a2(ctx.press("a2"))
    if (
        hang_back
        and not in_reach
        and enemy.closing_speed <= 0.0
        and ctx.enemy_block.max_thrust > ctx.spec.max_thrust
    ):
        # A faster opponent that is not coming cannot be caught; wait for it.
        ctx.plan.claim_thrust(False)
        ctx.point_at_opponent()
        return
    if not in_reach:
        ctx.approach_range(ILWRATH_FLAME_REACH, key="flame", inner_fraction=0.0)
    elif enemy.closing_speed < 0.0 and ctx.pointing(1, tolerance=tolerance):
        # Follow an opponent pulling out of reach.
        ctx.approach_opponent()
    ctx.point_weapon(1)


def ilwrath(ctx: BaselineContext) -> None:
    _ilwrath(ctx, hang_back=True)


def ilwrath_chase(ctx: BaselineContext) -> None:
    """Default rules, but chase faster opponents rather than wait for them."""

    _ilwrath(ctx, hang_back=False)


def ilwrath_always_cloak(ctx: BaselineContext) -> None:
    """Earlier rules: recloak whenever visible and fly straight in."""

    cloaked = not ctx.me.trackable or ctx.me.cloak_transition_direction > 0
    if ctx.in_range(1) and (cloaked or ctx.pointing(1)):
        ctx.a1(True)
    if not cloaked:
        ctx.a2(ctx.press("a2"))
    ctx.approach_opponent()


def kohr_ah(ctx: BaselineContext) -> None:
    if ctx.in_range(2):
        ctx.a2(True)
    # Every enemy object counts as harmful, which covers projectiles that
    # deal more than 3 damage.
    a2_range = ctx.weapon(2).range
    if any(threat.distance <= a2_range for threat in ctx.threats()):
        ctx.a2(True)
    ctx.avoid_opponent()
    ctx.point_weapon(1)
    if ctx.held("a1") and _shot_in_flight(ctx, minimum_speed=20.0):
        # Releasing parks the flying saw; keep it moving while it closes.
        ctx.a1(_hold_while_closing_on_enemy(ctx))
    elif ctx.pointing(1):
        ctx.a1(ctx.press("a1"))


def kzer_za(ctx: BaselineContext) -> None:
    fighter_counter = ctx.enemy_name in KZER_ZA_FIGHTER_COUNTERS or (
        ctx.enemy_name == "Chmmr" and ctx.enemy_block.chmmr_satellites > 0
    )
    if ctx.me.kzer_za_fighters < 8 and not fighter_counter and ctx.me.crew > 8:
        ctx.a2(True)
    ctx.approach_opponent()
    if ctx.pointing(1) and ctx.in_range(1):
        ctx.a1(True)


def mmrnmrhm(ctx: BaselineContext) -> None:
    x_form = not ctx.me.mmrnmrhm_ywing
    seen = ctx.enemy is not None
    in_x_range = seen and ctx.distance <= weapon_spec("Mmrnmrhm", 1, FORM_XFORM).range
    if x_form and in_x_range:
        ctx.a1(True)
        ctx.approach_opponent()
    if x_form and seen and not in_x_range:
        # Weapon 2 toggles the form on a fresh press.
        ctx.a2(ctx.press("a2"))
    if not x_form and in_x_range:
        ctx.a2(ctx.press("a2"))
    if not x_form:
        ctx.avoid_opponent()
        ctx.a1(True)
    ctx.point_at_opponent()


def melnorme(ctx: BaselineContext) -> None:
    charging = ctx.held("a1")
    if charging and ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(False)
    if not charging:
        ctx.a1(True)
    ctx.approach_weapon_range(1)
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if charging:
        ctx.a1(True)
    if ctx.in_range(2) and ctx.me.battery >= ctx.spec.a2_cost and ctx.pointing(2):
        ctx.a2(True)


def mycon(ctx: BaselineContext) -> None:
    if (
        ctx.me.crew < ctx.me.max_crew - ctx.spec.hp_gain
        and ctx.me.battery >= ctx.spec.a2_cost
    ):
        ctx.a2(True)
    if ctx.distance <= FAR_RANGE:
        ctx.avoid_opponent()
    if ctx.me.speed < ctx.spec.max_thrust:
        ctx.gravity_whip()
    ctx.point_weapon(1)
    velocity_bearing = None
    if ctx.me.speed > 1e-6:
        velocity_bearing = bearing_of(ctx.me.velocity_forward, ctx.me.velocity_right)
    facing_away_from_velocity = (
        velocity_bearing is not None and abs(velocity_bearing) > 90.0
    )
    if ctx.me.velocity_forward <= ctx.spec.max_thrust or facing_away_from_velocity:
        ctx.a1(True)


def orz(ctx: BaselineContext) -> None:
    # Satellites shoot marines down; board a Chmmr once at most one is left.
    satellites = ctx.chmmr_satellites()
    if (
        ctx.me.orz_marines < 2
        and ctx.me.crew > 2
        and ctx.enemy_name not in ORZ_MARINE_COUNTERS
        and len(satellites) <= ORZ_CHMMR_MARINE_SATELLITES
    ):
        # Weapon 1 and weapon 2 together launch a marine.
        ctx.a1(True)
        ctx.a2(True)
    if satellites:
        ctx.keep_standoff("orz_chmmr", ORZ_CHMMR_NEAR, ORZ_CHMMR_FAR)
    elif ctx.enemy_name in ORZ_KITE_OPPONENTS:
        ctx.keep_standoff("orz_kite", ORZ_KITE_NEAR, ORZ_KITE_FAR)
    else:
        ctx.approach_weapon_range(1)
    shot = ctx.satellite_shot(1) if satellites else None
    if shot is not None:
        # Turret shots at the weakest satellite in reach.
        satellite, bearing = shot
        tolerance = max(
            ctx.weapon(1).pointing_tolerance,
            math.degrees(math.atan2(SATELLITE_HIT_RADIUS, max(1.0, satellite.distance))),
        )
        if _orz_aim(ctx, bearing, tolerance):
            ctx.a1(True)
        return
    bearing = ctx.aim_bearing(1)
    if bearing is None:
        return
    if _orz_aim(ctx, bearing) and ctx.in_range(1):
        ctx.a1(True)


def _orz_aim(ctx: BaselineContext, bearing: float, tolerance: float | None = None) -> bool:
    """Bring the turret's line of fire onto ``bearing``; True once on it.

    With steering free the hull turns, twice as quick as the turret. While
    a rule holds the hull on a heading of its own (approaching, backing off,
    running) the turret turns instead, so the Orz fires off its course.
    """

    if tolerance is None:
        tolerance = max(ctx.weapon(1).pointing_tolerance, ctx.memory.tap_degrees / 2.0)
    if not ctx.plan.steering_claimed or ctx.enemy_name in ORZ_HULL_AIM_OPPONENTS:
        return ctx.aim_gun(1, bearing, tolerance)
    tolerance = max(float(tolerance), ORZ_TURRET_STEP / 2.0)
    if abs(ctx.gun_error(1, bearing)) <= tolerance:
        return True
    ctx.turn_turret(bearing)
    return False


def _pkunk_dodge(ctx: BaselineContext) -> bool:
    """Sidestep a hit about to land, away from where it will pass.

    Moves across the threat's track, on the side opposite its closest
    approach; thrust waits until the nose is within 90 degrees of the
    sidestep, as the Pkunk has no reverse.
    """

    threats = [
        threat for threat in ctx.threats() if ctx.imminent(threat, PKUNK_DODGE_SECONDS)
    ]
    if not threats:
        return False
    threat = min(threats, key=lambda item: item.distance)
    frames, _ = threat.closest_approach()
    frames = frames if math.isfinite(frames) and frames > 0.0 else 0.0
    pass_forward = threat.forward + threat.velocity_forward * frames
    pass_right = threat.right + threat.velocity_right * frames
    # Across the track: perpendicular to the threat's motion.
    across_forward, across_right = -threat.velocity_right, threat.velocity_forward
    if across_forward == 0.0 and across_right == 0.0:
        across_forward, across_right = -threat.right, threat.forward
    if across_forward * pass_forward + across_right * pass_right > 0.0:
        across_forward, across_right = -across_forward, -across_right
    heading = bearing_of(across_forward, across_right)
    return ctx.steer(heading, thrust=abs(heading) <= 90.0)


def _pkunk_pass(ctx: BaselineContext, enemy) -> None:
    """Fly past the opponent beside it, so a side gun bears as it goes by.

    Within gun range steering is left to aiming, which brings the nearest of
    the three guns onto the target; thrust then only while side-on, which
    carries the Pkunk across the opponent's nose rather than into it. An
    opponent that faces the Pkunk is passed on the side away from its nose.
    """

    reach = ctx.weapon(1).range + TYPICAL_HULL_RADIUS
    if enemy.distance <= reach:
        side_on = abs(abs(enemy.bearing) - 90.0) <= PKUNK_SIDE_ON_TOLERANCE
        ctx.plan.claim_thrust(side_on and ctx.me.speed < ctx.spec.max_thrust)
        return
    facing = ctx.enemy_block.opponent_bearing
    if facing is not None and 1.0 <= abs(facing) < PKUNK_FACING_ANGLE:
        # Its right is my left when it faces me: pass to my left.
        sign = -1.0 if facing > 0.0 else 1.0
        ctx.memory.sides["pkunk_pass"] = int(sign)
        ctx.memory.touched_sides.add("pkunk_pass")
    else:
        offset = math.degrees(math.asin(PKUNK_PASS_OFFSET / enemy.distance))
        sign = 1.0 if wrap_degrees(ctx.side("pkunk_pass", enemy.bearing, offset) - enemy.bearing) > 0.0 else -1.0
    # Aim point beside the opponent, perpendicular to the line to it.
    scale = sign * PKUNK_PASS_OFFSET / enemy.distance
    aim_forward = enemy.forward - enemy.right * scale
    aim_right = enemy.right + enemy.forward * scale
    # Seek it: thrust along the change from my velocity relative to the
    # opponent to top speed straight at the aim point, cancelling drift.
    aim_distance = max(1.0, math.hypot(aim_forward, aim_right))
    speed = ctx.spec.max_thrust
    correction_forward = aim_forward / aim_distance * speed + enemy.velocity_forward
    correction_right = aim_right / aim_distance * speed + enemy.velocity_right
    ctx.steer(bearing_of(correction_forward, correction_right))


def _pkunk(ctx: BaselineContext, *, passes: bool, dodge: bool, safe_station: bool) -> None:
    # Insults are the Pkunk's only battery source; they never block firing.
    ctx.a2(True)
    enemy = ctx.enemy
    a1_range = ctx.weapon(1).range
    charged = ctx.battery_at_least(0.50)
    if dodge and enemy is not None:
        _pkunk_dodge(ctx)
    if charged:
        if passes and enemy is not None:
            _pkunk_pass(ctx, enemy)
        else:
            # Close to where the short-lived shots reliably connect.
            ctx.approach_range(a1_range * PKUNK_FIRE_RANGE_FRACTION, key="a1_range")
    else:
        ctx.avoid_opponent()
        # Wait just outside weapon range so a recharge can strike at once,
        # and outside the opponent's own reach.
        station = a1_range * PKUNK_STATION_RANGE_FACTOR
        if safe_station:
            station = max(station, _enemy_reach(ctx) + TYPICAL_HULL_RADIUS + PKUNK_STATION_MARGIN)
        ctx.approach_range(
            station,
            key="station",
            inner_fraction=1.0 / PKUNK_STATION_RANGE_FACTOR,
        )
    # Aiming uses whichever of the front and side guns is nearest the target.
    if ctx.in_range(1) or not charged:
        ctx.point_weapon(1)
    enemy = ctx.enemy
    closing = enemy is not None and enemy.closing_speed > 0.0
    if ctx.pointing(1) and (
        ctx.in_range(1, PKUNK_FIRE_RANGE_FRACTION) or (closing and ctx.in_range(1))
    ):
        ctx.a1(True)


def pkunk(ctx: BaselineContext) -> None:
    _pkunk(ctx, passes=True, dodge=False, safe_station=True)


def pkunk_head_on(ctx: BaselineContext) -> None:
    """Earlier rules: close head-on and recharge just outside gun range."""

    _pkunk(ctx, passes=False, dodge=False, safe_station=False)


def pkunk_dodge(ctx: BaselineContext) -> None:
    _pkunk(ctx, passes=True, dodge=True, safe_station=True)


def pkunk_near_station(ctx: BaselineContext) -> None:
    _pkunk(ctx, passes=True, dodge=False, safe_station=False)


def _enemy_form(ctx: BaselineContext) -> str:
    if ctx.enemy_name == "Mmrnmrhm":
        return FORM_YWING if ctx.enemy_block.mmrnmrhm_ywing else FORM_XFORM
    if ctx.enemy_name == "Androsynth" and ctx.enemy_block.androsynth_blazer:
        return FORM_BLAZER
    return FORM_BASE


def _instant_hit_damage(ctx: BaselineContext) -> float:
    """Crew the opponent's instant-hit weapons can take from me right now."""

    enemy = ctx.enemy
    name = ctx.enemy_name
    if enemy is None or name is None:
        return 0.0
    form = _enemy_form(ctx)
    facing = ctx.enemy_block.opponent_bearing
    total = 0.0
    for action in (1, 2):
        weapon = weapon_spec(name, action, form)
        if weapon.kind not in ("laser", "area") or weapon.ability_name in NOT_INSTANT_HITS:
            continue
        if enemy.distance > weapon.range + TYPICAL_HULL_RADIUS:
            continue
        if not weapon.tracking:
            tolerance = UMGAH_CONE_HALF_ANGLE if name == "Umgah" else INSTANT_HIT_AIM_TOLERANCE
            if facing is None or min(
                abs(wrap_degrees(facing - gun)) for gun in weapon.gun_directions
            ) > tolerance:
                continue
        total += weapon.damage
    escort_laser = ESCORT_LASERS.get(name)
    if escort_laser is not None:
        definition = ABILITY_DEFINITIONS[escort_laser]
        reach = float(definition.range) + TYPICAL_HULL_RADIUS
        escorts = sum(
            1 for item in ctx.view.objects("enemy_non_a1") if item.distance <= reach
        )
        total += float(definition.damage[0]) * escorts
    return total


def _about_to_die(ctx: BaselineContext) -> bool:
    """Whether hits landing now or within the fatal-hit window take my last crew."""

    incoming = sum(
        max(0.0, -threat.crew_effect)
        for threat in ctx.threats()
        if ctx.imminent(threat, SHOFIXTI_FATAL_HIT_SECONDS)
    )
    incoming += _instant_hit_damage(ctx)
    return incoming > 0.0 and incoming >= ctx.me.crew - 1e-6


def _glory_blast_kills(ctx: BaselineContext) -> bool:
    """Whether the glory device kills the opponent from here."""

    enemy = ctx.enemy
    if enemy is None or ctx.enemy_block.damage_shield_active:
        return False
    opening = max(0.0, -enemy.closing_speed) * SHOFIXTI_BLAST_LEAD_FRAMES
    distance = enemy.distance + opening + SHOFIXTI_BLAST_MARGIN
    return distance <= shofixti_kill_distance(ctx.enemy_block.crew)


def shofixti(ctx: BaselineContext) -> None:
    values = ctx.memory.values
    if ctx.me.battery < ctx.spec.a1_cost:
        values["shofixti_recharging"] = True
    elif ctx.me.battery >= ctx.me.max_battery:
        values["shofixti_recharging"] = False
    recharging = bool(values.get("shofixti_recharging", False))
    in_blast_range = ctx.enemy is not None and ctx.distance <= ctx.weapon(2).range
    # Against another Shofixti a kill blast also kills me, so only blast
    # there when its shots are about to kill me anyway.
    mirror_match = ctx.enemy_name == "Shofixti"
    if in_blast_range and (
        _about_to_die(ctx) or (not mirror_match and _glory_blast_kills(ctx))
    ):
        ctx.a2(ctx.press("a2"))
    if recharging:
        ctx.avoid_opponent()
    else:
        ctx.approach_weapon_range(1)
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


def shofixti_glory_dive(ctx: BaselineContext) -> None:
    """Earlier rules: dive to half the blast's reach and blow up."""

    glory_range = ctx.weapon(2).range / 2.0
    mirror_match = ctx.enemy_name == "Shofixti"
    if not mirror_match and ctx.distance <= glory_range:
        ctx.a2(ctx.press("a2"))
    if not mirror_match:
        ctx.approach_range(glory_range, key="glory")
    if mirror_match:
        ctx.approach_weapon_range(1)
        ctx.point_weapon(1)
        if ctx.in_range(1) and ctx.pointing(1):
            ctx.a1(True)


def slylandro(ctx: BaselineContext) -> None:
    ctx.a2(True)
    if ctx.me.battery_fraction <= 0.10:
        _approach_enemy_object(ctx, ctx.view.asteroids)
    ctx.approach_weapon_range(1)
    if ctx.in_range(1):
        ctx.a1(True)


def spathi(ctx: BaselineContext) -> None:
    if ctx.battery_at_least(0.50):
        ctx.approach_weapon_range(1)
    else:
        ctx.avoid_opponent()
    ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)
    rearward_speed = -ctx.me.velocity_forward
    if (
        ctx.in_range(2)
        and ctx.me.battery >= ctx.spec.a2_cost
        and rearward_speed <= ctx.weapon(2).speed
    ):
        ctx.a2(True)


def _enemy_reach(ctx: BaselineContext) -> float:
    """Longest reach of the opponent's direct weapons (not escorts)."""

    name = ctx.enemy_name
    if name is None:
        return 0.0
    if name == "Ilwrath":
        return ILWRATH_FLAME_REACH
    form = _enemy_form(ctx)
    reach = 0.0
    for action in (1, 2):
        weapon = weapon_spec(name, action, form)
        if weapon.kind in ("projectile", "laser", "area"):
            reach = max(reach, weapon.range)
    return reach


def _supox_band(ctx: BaselineContext) -> tuple[float, float]:
    """Distance band to hold: just outside a shorter-reaching opponent."""

    longest = ctx.weapon(1).range * SUPOX_BAND_GUN_FRACTION
    outside = _enemy_reach(ctx) + TYPICAL_HULL_RADIUS + SUPOX_OUTRANGE_MARGIN
    if SUPOX_BAND_NEAR < outside <= longest - SUPOX_BAND_MIN_WIDTH:
        return outside, max(SUPOX_BAND_FAR, min(longest, outside + SUPOX_BAND_MIN_WIDTH))
    return SUPOX_BAND_NEAR, SUPOX_BAND_FAR


def _supox_slide(ctx: BaselineContext, lateral: int, *, backward: bool) -> None:
    """Hold A2 so turning keys slide the ship and thrust reverses it."""

    ctx.a2(True)
    ctx.plan.claim_turn(lateral)
    ctx.plan.claim_thrust(backward)


def _supox_dodge_side(ctx: BaselineContext) -> int | None:
    """Slide direction away from the path of a hit about to land."""

    threats = [
        threat for threat in ctx.threats() if ctx.imminent(threat, SUPOX_DODGE_SECONDS)
    ]
    if not threats:
        return None
    threat = min(threats, key=lambda item: item.distance)
    frames, _ = threat.closest_approach()
    frames = frames if math.isfinite(frames) and frames > 0.0 else 0.0
    passes_right = threat.right + threat.velocity_right * frames
    return TURN_LEFT if passes_right >= 0.0 else TURN_RIGHT


def _supox_evade_side(ctx: BaselineContext) -> int:
    """Slide direction that moves me further off the opponent's nose line.

    The opponent faces me, so its right is my left: on its right
    (positive bearing from its nose), keep sliding to my left. When its aim
    swings past me the sign flips and so does the slide.
    """

    facing = ctx.enemy_block.opponent_bearing
    values = ctx.memory.values
    if facing is not None and abs(facing) >= 1.0:
        values["supox_evade"] = TURN_LEFT if facing > 0.0 else TURN_RIGHT
    return int(values.get("supox_evade", TURN_LEFT))


def _supox_satellite_aim(ctx: BaselineContext) -> tuple[float, float] | None:
    """Gun error to the weakest Chmmr satellite in reach, and the fire
    tolerance for it; None with no satellite in reach."""

    shot = ctx.satellite_shot(1)
    if shot is None:
        return None
    satellite, bearing = shot
    tolerance = max(
        ctx.weapon(1).pointing_tolerance,
        math.degrees(math.atan2(SATELLITE_HIT_RADIUS, max(1.0, satellite.distance))),
    )
    return ctx.gun_error(1, bearing), tolerance


def supox(ctx: BaselineContext) -> None:
    enemy = ctx.enemy
    if enemy is None:
        return
    fire_tolerance = max(
        ctx.weapon(1).pointing_tolerance,
        math.degrees(math.atan2(TYPICAL_HULL_RADIUS, max(1.0, ctx.distance))),
    )
    # Against a Chmmr, shoot its satellites while settled in the band; the
    # band itself is still kept against the Chmmr, as sliding can't turn.
    satellite_aim = _supox_satellite_aim(ctx)
    if ctx.plan.steering_claimed:
        # The planet rule is flying the ship; A2 would turn its turns into slides.
        ctx.a2(False)
    else:
        near, far = _supox_band(ctx)
        error = ctx.aim_error(1)
        aimed = error is not None and abs(error) <= SUPOX_SLIDE_AIM_TOLERANCE
        facing = ctx.enemy_block.opponent_bearing
        faced = facing is not None and abs(facing) <= SUPOX_FACING_TOLERANCE
        dodge = _supox_dodge_side(ctx)
        if dodge is not None:
            _supox_slide(ctx, dodge, backward=ctx.distance < near)
        elif satellite_aim is not None and near <= ctx.distance <= far and not faced:
            ctx.a2(False)
            ctx.point_at(satellite_aim[0])
            ctx.plan.claim_thrust(False)
        elif not aimed:
            ctx.a2(False)
            ctx.point_weapon(1)
            ahead = error is not None and abs(error) <= 45.0
            ctx.plan.claim_thrust(ctx.distance > far and ahead and ctx.thrust_helps(0.0))
        elif ctx.distance < near:
            _supox_slide(
                ctx, _supox_evade_side(ctx) if faced else TURN_NONE, backward=True
            )
        elif ctx.distance > far:
            ctx.a2(False)
            ctx.point_weapon(1)
            ctx.plan.claim_thrust(ctx.thrust_helps(0.0))
        elif faced:
            _supox_slide(ctx, _supox_evade_side(ctx), backward=False)
        else:
            ctx.a2(False)
            ctx.point_weapon(1)
            ctx.plan.claim_thrust(False)
    error = ctx.aim_error(1)
    on_enemy = ctx.in_range(1) and error is not None and abs(error) <= fire_tolerance
    on_satellite = satellite_aim is not None and abs(satellite_aim[0]) <= satellite_aim[1]
    if on_enemy or on_satellite:
        ctx.a1(True)


def supox_no_slide(ctx: BaselineContext) -> None:
    """Earlier rules: never use A2."""

    if ctx.battery_at_least(0.30):
        ctx.approach_weapon_range(1)
    else:
        ctx.avoid_opponent()
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


def _song_viable(ctx: BaselineContext) -> bool:
    return ctx.enemy_block.crew > 1 and ctx.enemy_name not in SONG_IMMUNE_SHIPS


def _syreen_risky(ctx: BaselineContext, *, cautious: bool) -> bool:
    """Whether closing in to sing is too dangerous while the crew is small."""

    small = ctx.me.crew < SYREEN_GROWN_CREW
    outpaced = ctx.enemy is not None and ctx.enemy_block.max_thrust >= ctx.spec.max_thrust
    return cautious and small and ctx.enemy_name in SYREEN_CLOSE_DANGER_SHIPS and outpaced


def _syreen_may_evade(ctx: BaselineContext) -> bool:
    """Dodge, except while closing in with the song ready."""

    song_ready = ctx.me.battery >= ctx.spec.a2_cost
    return not (_song_viable(ctx) and song_ready and not _syreen_risky(ctx, cautious=True))


def _syreen(ctx: BaselineContext, *, cautious: bool, keep_away: bool) -> None:
    song_viable = _song_viable(ctx)
    song_ready = ctx.me.battery >= ctx.spec.a2_cost
    if song_viable and song_ready and ctx.distance <= SYREEN_SONG_DISTANCE:
        ctx.a2(True)
    pods = ctx.view.crew_pods
    if pods:
        ctx.approach_object(min(pods, key=lambda pod: pod.distance))
    risky = _syreen_risky(ctx, cautious=cautious)
    gun_range = ctx.weapon(1).range
    if keep_away and not song_ready:
        ctx.avoid_opponent()
        ctx.approach_range(gun_range, key="gun")
    elif song_viable and not risky:
        ctx.approach_range(SYREEN_SONG_DISTANCE, key="song")
    else:
        ctx.approach_range(gun_range, key="gun")
    ctx.point_weapon(1)
    spare = ctx.me.battery - ctx.spec.a1_cost >= ctx.spec.a2_cost
    if ctx.in_range(1) and ctx.pointing(1) and (spare or not song_viable):
        ctx.a1(True)


def syreen(ctx: BaselineContext) -> None:
    _syreen(ctx, cautious=True, keep_away=True)


def syreen_bold(ctx: BaselineContext) -> None:
    """Default rules, but dive to sing at dangerous opponents while small."""

    _syreen(ctx, cautious=False, keep_away=True)


def syreen_pressing(ctx: BaselineContext) -> None:
    """Default rules, but stay close while the song recharges."""

    _syreen(ctx, cautious=True, keep_away=False)


def syreen_full_reach(ctx: BaselineContext) -> None:
    """Earlier rules: fire constantly and sing at the song's full reach."""

    ctx.a1(True)
    enemy = ctx.enemy
    pods = ctx.view.crew_pods
    if pods and enemy is not None:
        nearest = min(pods, key=lambda pod: pod.distance)
        pod_to_enemy = (
            (nearest.forward - enemy.forward) ** 2 + (nearest.right - enemy.right) ** 2
        ) ** 0.5
        if nearest.distance < pod_to_enemy:
            ctx.approach_object(nearest)
    can_sing = ctx.enemy_block.crew > 1 and ctx.enemy_name != "Slylandro"
    if can_sing:
        ctx.approach_weapon_range(2)
    if can_sing and ctx.in_range(2) and ctx.me.battery >= ctx.spec.a2_cost:
        ctx.a2(True)
    ctx.approach_weapon_range(1)
    ctx.point_weapon(1)


def _thraddash_tail_burn(ctx: BaselineContext, target, *, burn: bool) -> None:
    """Point the tail at ``target``; burn once it lines up, laying the trail."""

    tail = wrap_degrees(target.bearing + 180.0)
    ctx.point_at(tail)
    if burn and abs(tail) <= THRADDASH_TAIL_TOLERANCE and ctx.me.battery >= ctx.spec.a2_cost:
        ctx.a2(True)


def thraddash(ctx: BaselineContext) -> None:
    enemy = ctx.enemy
    if enemy is None:
        return
    values = ctx.memory.values
    # Shots about to land from the side or behind: burn forward, and the
    # trail catches what follows.
    for threat in ctx.view.enemy_objects:
        if abs(threat.bearing) >= THRADDASH_DODGE_MIN_BEARING and ctx.imminent(
            threat, THRADDASH_AFTERBURNER_SECONDS
        ):
            ctx.a2(True)
            break
    if ctx.enemy_name == "Chmmr":
        _thraddash_vs_chmmr(ctx)
        return
    # Escape: burn away from an opponent coming too close, or arriving soon.
    # Slow turners are baited in nearer, so they follow into the trail.
    slow_turner = ctx.enemy_name in THRADDASH_SLOW_TURNERS
    escape_distance = THRADDASH_BAIT_DISTANCE if slow_turner else THRADDASH_ESCAPE_DISTANCE
    # Never wait inside the opponent's own reach.
    outside_reach = _enemy_reach(ctx) + TYPICAL_HULL_RADIUS + THRADDASH_REACH_MARGIN
    escape_distance = max(escape_distance, min(outside_reach, THRADDASH_STANDOFF_NEAR))
    closing = enemy.closing_speed > 0.0
    arriving = (
        not slow_turner
        and enemy.distance <= THRADDASH_STANDOFF_NEAR
        and enemy.closing_seconds() <= THRADDASH_ESCAPE_SECONDS
    )
    if closing and (enemy.distance <= escape_distance or arriving):
        values["thraddash_escaping"] = True
    elif not closing or enemy.distance > THRADDASH_STANDOFF_NEAR:
        values["thraddash_escaping"] = False
    escaping = bool(values.get("thraddash_escaping", False))
    # Escorts and homing shots chasing me (fighters, marines, Dogi,
    # missiles): lead them through the trail.
    pursuers = list(ctx.view.objects("enemy_non_a1"))
    if ctx.enemy_name is not None and weapon_spec(ctx.enemy_name, 1, _enemy_form(ctx)).tracking:
        pursuers.extend(ctx.view.objects("enemy_a1"))
    swarm = [
        item
        for item in pursuers
        if item.closing_speed > 0.0 and item.distance <= THRADDASH_SWARM_RANGE
    ]
    chasers = list(swarm)
    if escaping:
        chasers.append(enemy)
    if chasers:
        chaser = min(chasers, key=lambda item: item.distance)
        burn = chaser is enemy or chaser.distance <= THRADDASH_TRAIL_RANGE
        _thraddash_tail_burn(ctx, chaser, burn=burn)
        return
    # Far off: dash in on the afterburner past one side of the opponent,
    # then coast; a few frames of burn reach the afterburner's top speed.
    if (
        enemy.distance > THRADDASH_STANDOFF_FAR + THRADDASH_DASH_MARGIN
        and ctx.me.battery >= THRADDASH_BURN_RESERVE + THRADDASH_DASH_COST
    ):
        heading = ctx.drift_cancelling_side("standoff_far", enemy, APPROACH_ANGLE_OFFSET)
        ctx.steer(heading)
        if abs(heading) <= THRADDASH_DASH_ALIGN and ctx.me.speed < THRADDASH_DASH_SPEED:
            ctx.a2(True)
    # Hold the standoff band, coasting rather than thrusting above top speed.
    ctx.approach_range(
        THRADDASH_STANDOFF_FAR,
        key="standoff",
        inner_fraction=THRADDASH_STANDOFF_NEAR / THRADDASH_STANDOFF_FAR,
    )
    ctx.point_weapon(1)
    spare = ctx.me.battery - ctx.spec.a1_cost >= THRADDASH_BURN_RESERVE
    if ctx.distance <= THRADDASH_STANDOFF_FAR and ctx.pointing(1) and spare:
        ctx.a1(True)


def _thraddash_vs_chmmr(ctx: BaselineContext) -> None:
    """Snipe satellites from outside their reach; bait the Chmmr into flames.

    Its 2-HP shots survive one satellite zap. A closing Chmmr is led through
    the afterburner trail, whose 1-HP flames the satellites shoot first.
    """

    enemy = ctx.enemy
    values = ctx.memory.values
    closing = enemy.closing_speed > 0.0
    previous = values.get("thraddash_closing_speed", enemy.closing_speed)
    values["thraddash_closing_speed"] = enemy.closing_speed
    smoothing = THRADDASH_CHMMR_ACCEL_SMOOTHING
    speeding_up = (1.0 - smoothing) * values.get("thraddash_closing_accel", 0.0) + (
        smoothing * (enemy.closing_speed - previous)
    )
    values["thraddash_closing_accel"] = speeding_up
    tail = wrap_degrees(enemy.bearing + 180.0)
    turn_frames = (ctx.turn_seconds(tail) + THRADDASH_CHMMR_TURN_MARGIN) * const.FPS
    reach_at_turn = enemy.distance - (
        max(0.0, enemy.closing_speed) * turn_frames
        + 0.5 * max(0.0, speeding_up) * turn_frames * turn_frames
    )
    if closing and reach_at_turn <= THRADDASH_CHMMR_ESCAPE:
        values["thraddash_escaping"] = True
    elif enemy.distance > THRADDASH_CHMMR_RELEASE or (
        enemy.closing_speed < 0.0 and reach_at_turn > THRADDASH_CHMMR_NEAR
    ):
        values["thraddash_escaping"] = False
    if values.get("thraddash_escaping", False):
        _thraddash_tail_burn(ctx, enemy, burn=True)
        return
    if enemy.distance > THRADDASH_CHMMR_FAR:
        heading = ctx.drift_cancelling_side("chmmr_band", enemy, APPROACH_ANGLE_OFFSET)
        ctx.steer(heading, thrust=enemy.closing_speed < THRADDASH_CHMMR_APPROACH_SPEED)
    elif enemy.distance < THRADDASH_CHMMR_NEAR:
        ctx.steer(ctx.side("chmmr_near", enemy.bearing, PERPENDICULAR_AWAY_OFFSET))
    shot = ctx.satellite_shot(1)
    if shot is not None:
        pointing = ctx.aim_gun(1, shot[1])
    else:
        ctx.point_weapon(1)
        pointing = False
    # A shot lined up on the Chmmr itself goes too: with 2 HP it may get by.
    pointing = pointing or (ctx.pointing(1) and ctx.in_range(1))
    spare = ctx.me.battery - ctx.spec.a1_cost >= THRADDASH_BURN_RESERVE
    if pointing and spare:
        ctx.a1(True)


def thraddash_dodge_only(ctx: BaselineContext) -> None:
    """Earlier rules: close to weapon range and burn only to dodge."""

    ctx.approach_weapon_range(1)
    ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)
    if ctx.imminent_threat(THRADDASH_AFTERBURNER_SECONDS) is not None:
        ctx.a2(True)


def _umgah_dodge(ctx: BaselineContext) -> None:
    """Hop backward out of the way of a hit about to land from the side."""

    if ctx.me.battery < ctx.spec.a2_cost:
        return
    side_threats = list(ctx.view.enemy_objects)
    if ctx.enemy is not None:
        side_threats.append(ctx.enemy)
    for threat in side_threats:
        broadside = abs(abs(wrap_degrees(threat.bearing)) - 90.0) <= 45.0
        if broadside and ctx.imminent(threat, UMGAH_RETRO_SECONDS):
            ctx.a2(True)
            return


def _umgah_recharging(ctx: BaselineContext) -> bool:
    me = ctx.me
    values = ctx.memory.values
    if me.battery <= UMGAH_HOP_RESERVE:
        values["umgah_recharging"] = True
    elif me.battery >= me.max_battery:
        values["umgah_recharging"] = False
    return bool(values.get("umgah_recharging", False))


def _umgah_cone_entry(lateral: float) -> float:
    """How far in front the opponent must be, beside my path, to sit in the cone."""

    return max(abs(lateral), UMGAH_RUN_CLEARANCE) / math.tan(
        math.radians(UMGAH_CONE_HALF_ANGLE)
    )


def _umgah_hops_needed(enemy) -> int:
    """Backward hops until the opponent is far enough in front for the cone."""

    entry = _umgah_cone_entry(enemy.right)
    if enemy.forward >= entry:
        return 0
    gain = UMGAH_HOP_DISTANCE + enemy.velocity_forward * UMGAH_HOP_FRAMES
    if gain <= 0.0:
        return 0
    return math.ceil((entry - enemy.forward) / gain)


def _umgah_path_clear(ctx: BaselineContext, hops: int) -> bool:
    """Whether ``hops`` backward hops would run into an enemy shot, mine or blade.

    Hops carry the ship straight back ``UMGAH_HOP_DISTANCE`` every
    ``UMGAH_HOP_FRAMES``, so an object meets it about where their relative
    tracks pass closest. Only objects that would otherwise miss count: one
    already coming straight at me hits whether I hop or wait.
    """

    if hops <= 0:
        return True
    duration = float(max(hops, UMGAH_PATH_LOOKAHEAD_HOPS) * UMGAH_HOP_FRAMES)
    for obj in ctx.view.enemy_objects:
        if _closest_miss(obj.forward, obj.right, obj.velocity_forward, obj.velocity_right, duration) <= UMGAH_CORRIDOR_HALF_WIDTH:
            continue
        hop_miss = _closest_miss(
            obj.forward,
            obj.right,
            obj.velocity_forward + UMGAH_HOP_DISTANCE / UMGAH_HOP_FRAMES,
            obj.velocity_right,
            duration,
        )
        if hop_miss <= UMGAH_CORRIDOR_HALF_WIDTH:
            return False
    return True


def _closest_miss(
    forward: float, right: float, speed_forward: float, speed_right: float, frames: float
) -> float:
    """Closest distance to me of a point moving steadily, within ``frames``."""

    speed_squared = speed_forward * speed_forward + speed_right * speed_right
    when = 0.0
    if speed_squared > 1e-9:
        when = -(forward * speed_forward + right * speed_right) / speed_squared
        when = min(max(when, 0.0), float(frames))
    return math.hypot(forward + speed_forward * when, right + speed_right * when)


def _umgah_pass_sign(ctx: BaselineContext) -> int:
    """Side of my backward path to keep the opponent on (the sign of its right).

    When the opponent faces me, pass on the side its nose turns away from,
    so the run ends behind its guns. Its right is my left when it faces me,
    and my right flips again once my tail points at it. Zero: no preference.
    """

    facing = ctx.enemy_block.opponent_bearing
    if facing is None or abs(facing) >= UMGAH_FACING_ANGLE or abs(facing) < 1.0:
        return 0
    return -1 if facing > 0.0 else 1


def _umgah_tail_heading(ctx: BaselineContext, enemy) -> float:
    """Nose bearing that points the tail just beside the opponent."""

    clearance = min(UMGAH_RUN_CLEARANCE, enemy.distance)
    offset = math.degrees(math.asin(clearance / max(enemy.distance, clearance)))
    base = wrap_degrees(enemy.bearing + 180.0)
    sign = _umgah_pass_sign(ctx)
    if sign == 0:
        # No preference: keep the side that needs the least turning.
        return ctx.side("umgah_run", base, offset)
    return wrap_degrees(base + sign * offset)


def _umgah_run(ctx: BaselineContext, enemy) -> bool:
    """Hop past the opponent so it lands just in front of the cone.

    Far off, the tail need only point near the opponent, and the run steers
    as it closes; within ``UMGAH_LATERAL_AIM_DISTANCE`` the path itself must
    pass beside the opponent. A run starts with the battery for all of its
    hops, never crosses an enemy shot, and stops as soon as the opponent is
    far enough in front, so it never overshoots by more than one hop.
    """

    values = ctx.memory.values
    hops = _umgah_hops_needed(enemy)
    heading = _umgah_tail_heading(ctx, enemy)
    lateral = enemy.right + enemy.velocity_right * hops * UMGAH_HOP_FRAMES
    far = enemy.distance > UMGAH_LATERAL_AIM_DISTANCE
    if far:
        lined_up = abs(heading) <= UMGAH_HOP_ALIGN_TOLERANCE
    else:
        lined_up = enemy.forward < 0.0 and abs(lateral) <= UMGAH_RUN_MAX_LATERAL
    running = bool(values.get("umgah_running", False))
    if running:
        go = hops > 0 and lined_up and ctx.me.battery >= ctx.spec.a2_cost
    else:
        sign = _umgah_pass_sign(ctx)
        go = (
            hops > 0
            and lined_up
            and ctx.me.battery >= hops * ctx.spec.a2_cost + 1.0
            and (
                far
                or (
                    # Too near its line and the hulls bump, which stops the hop.
                    abs(lateral) >= UMGAH_RUN_MIN_LATERAL
                    and (
                        sign == 0
                        or lateral * sign >= 0.0
                        or abs(lateral) <= UMGAH_RUN_CLEARANCE
                    )
                )
            )
        )
    go = go and _umgah_path_clear(ctx, hops)
    values["umgah_running"] = go
    if not go:
        return False
    if far:
        ctx.point_at(heading)
    else:
        ctx.plan.claim_turn(TURN_NONE)
    ctx.plan.claim_thrust(False)
    ctx.a2(True)
    return True


def umgah(ctx: BaselineContext) -> None:
    """Hop backward past the opponent, landing it just in front of the cone.

    The backward hop (weapon 2) jumps 160 units every other frame per
    battery point, only while thrust is released; the cone (weapon 1) is
    free yet stops the all-at-once battery refill while it fires.
    """

    enemy = ctx.enemy
    recharging = _umgah_recharging(ctx)
    _umgah_dodge(ctx)
    if any(item.distance <= UMGAH_HOSTILE_RANGE for item in ctx.view.enemy_objects):
        ctx.plan.claim_thrust(False)
    if enemy is None:
        return
    ahead = abs(enemy.bearing) <= 90.0
    cone_reach = ctx.weapon(1).range + TYPICAL_HULL_RADIUS
    if ahead and ctx.distance <= cone_reach:
        ctx.memory.values["umgah_running"] = False
        ctx.point_at_opponent()
        if abs(enemy.bearing) <= UMGAH_CONE_HALF_ANGLE:
            ctx.a1(True)
        return
    if ahead and ctx.distance <= UMGAH_CLOSE_APPROACH_RANGE:
        ctx.memory.values["umgah_running"] = False
        ctx.approach_opponent()
        return
    if ctx.enemy_name in UMGAH_AMBUSH_SHIPS and (
        ahead or ctx.distance > UMGAH_LATERAL_AIM_DISTANCE
    ):
        # Short-reach ships come to me: wait facing them, cone ready.
        ctx.memory.values["umgah_running"] = False
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(False)
        return
    if _umgah_run(ctx, enemy):
        return
    if recharging:
        ctx.avoid_opponent()
        ctx.point_at_opponent()
        return
    # Line the tail up beside the opponent, ready for a run.
    ctx.point_at(_umgah_tail_heading(ctx, enemy))
    ctx.plan.claim_thrust(False)


def umgah_hop_in(ctx: BaselineContext) -> None:
    """Earlier rules: hop backward past the opponent from any distance."""

    enemy = ctx.enemy
    recharging = _umgah_recharging(ctx)
    # The reserve is kept for dodges, which need thrust released.
    _umgah_dodge(ctx)
    if any(item.distance <= UMGAH_HOSTILE_RANGE for item in ctx.view.enemy_objects):
        ctx.plan.claim_thrust(False)
    if enemy is None:
        return
    ahead = abs(enemy.bearing) <= 90.0
    cone_reach = ctx.weapon(1).range + TYPICAL_HULL_RADIUS
    if ahead and ctx.distance <= cone_reach:
        ctx.point_at_opponent()
        if abs(enemy.bearing) <= UMGAH_CONE_HALF_ANGLE:
            ctx.a1(True)
        return
    if recharging:
        # Back off and leave both weapons alone until the battery refills.
        ctx.avoid_opponent()
        ctx.point_at_opponent()
        return
    if ahead and ctx.distance <= UMGAH_CLOSE_APPROACH_RANGE:
        ctx.approach_opponent()
        return
    # Point the tail just beside the opponent so the hops pass it by.
    pass_angle = math.degrees(
        math.asin(UMGAH_PASS_CLEARANCE / max(ctx.distance, UMGAH_PASS_CLEARANCE))
    )
    nose = ctx.side("umgah_pass", wrap_degrees(enemy.bearing + 180.0), pass_angle)
    ctx.point_at(nose)
    if abs(nose) <= UMGAH_HOP_ALIGN_TOLERANCE:
        ctx.plan.claim_thrust(False)
        ctx.a2(True)


def umgah_approach_cone(ctx: BaselineContext) -> None:
    """Earlier rules: thrust at the opponent and cone when aimed."""

    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)
    frontal_threats = [
        threat
        for threat in ctx.view.enemy_objects
        if abs(wrap_degrees(threat.bearing)) <= 45.0
        and threat.closing_seconds() <= UMGAH_CONE_THREAT_SECONDS + ctx.span_seconds
    ]
    if frontal_threats:
        closest = min(frontal_threats, key=lambda threat: threat.distance)
        ctx.point_at(closest.bearing)
        ctx.a1(True)
    ctx.approach_opponent()
    side_threats = list(ctx.view.enemy_objects)
    if ctx.enemy is not None:
        side_threats.append(ctx.enemy)
    for threat in side_threats:
        broadside = abs(abs(wrap_degrees(threat.bearing)) - 90.0) <= 45.0
        if broadside and ctx.imminent(threat, UMGAH_RETRO_SECONDS):
            ctx.a2(True)
            break


def _shield_rules(ctx: BaselineContext) -> None:
    enemy = ctx.enemy
    # Shield wherever a Shofixti blast would kill, and always within half
    # its reach.
    glory_radius = max(
        SHOFIXTI_SHIELD_DISTANCE,
        shofixti_kill_distance(ctx.me.crew) + SHOFIXTI_BLAST_MARGIN,
    )
    shofixti_closing = (
        ctx.enemy_name == "Shofixti"
        and enemy is not None
        and ctx.distance
        <= glory_radius + max(0.0, enemy.closing_speed) * ctx.span_frames
    )
    if (
        ctx.imminent_threat(SHIELD_SECONDS) is not None
        or ctx.crew_dropped()
        or shofixti_closing
    ):
        ctx.a2(True)
    ctx.approach_weapon_range(1)
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


def utwig(ctx: BaselineContext) -> None:
    _shield_rules(ctx)


def _utwig_may_evade(ctx: BaselineContext) -> bool:
    """Dodge only without the battery to shield: shielded hits refill it."""

    return ctx.me.battery < ctx.spec.a2_cost


def yehat(ctx: BaselineContext) -> None:
    _shield_rules(ctx)


def _vux_disabled(ctx: BaselineContext) -> bool:
    """Whether limpets have left the opponent slower and no nimbler than me."""

    block = ctx.enemy_block
    return (
        block.max_thrust < ctx.spec.max_thrust
        and block.turn_wait >= VUX_DISABLED_TURN_WAIT
    )


def _own_speed_toward(ctx: BaselineContext, target) -> float:
    radians = math.radians(target.bearing)
    return (
        ctx.me.velocity_forward * math.cos(radians)
        + ctx.me.velocity_right * math.sin(radians)
    )


def _vux_keep_away(ctx: BaselineContext) -> None:
    """Head away from the opponent, then turn to face it while drifting away."""

    enemy = ctx.enemy
    values = ctx.memory.values
    toward = _own_speed_toward(ctx, enemy)
    retreat_fast = ctx.spec.max_thrust * VUX_RETREAT_SPEED_FRACTION
    if toward <= -retreat_fast:
        values["vux_retreating"] = False
    elif toward > -retreat_fast * VUX_RETREAT_RESUME_FRACTION:
        values["vux_retreating"] = True
    if enemy.distance > VUX_LIMPET_REACH:
        # Out past limpet reach: face it, and only thrust to stop drifting off.
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(toward < 0.0 and abs(enemy.bearing) <= 45.0)
        return
    if values.get("vux_retreating", True):
        away = wrap_degrees(enemy.bearing + 180.0)
        # Thrust only once the nose points away, never toward the opponent.
        ctx.steer(away, thrust=abs(away) <= 90.0)
    else:
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(False)


def _vux_blind_side_approach(ctx: BaselineContext) -> None:
    """Close in on a crippled opponent from where its nose is not pointing."""

    enemy = ctx.enemy
    if enemy.distance <= VUX_CLOSE_DISTANCE:
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(False)
        return
    facing = ctx.enemy_block.opponent_bearing
    if facing is None or abs(facing) >= VUX_BLIND_ANGLE:
        ctx.approach_opponent()
        return
    # It faces me: come in off its nose line, on the side I am already on.
    # Its right is my left when it faces me.
    offset = -VUX_BLIND_ANGLE if facing > 0.0 else VUX_BLIND_ANGLE
    ctx.steer(wrap_degrees(enemy.bearing + offset))


def _vux(ctx: BaselineContext, *, slingshot: bool, hold_beam: bool) -> None:
    enemy = ctx.enemy
    if enemy is None:
        return
    disabled = _vux_disabled(ctx)
    # Laser: whenever the hull crosses the beam within reach.
    tolerance = math.degrees(math.atan2(TYPICAL_HULL_RADIUS, max(1.0, enemy.distance)))
    in_reach = enemy.distance <= VUX_BURST_DISTANCE
    if in_reach and abs(enemy.bearing) <= tolerance:
        ctx.a1(True)
    # A Chmmr satellite crossing the beam takes the hit instead.
    if ctx.satellite_on_beam(0.0, ctx.weapon(1).range + SATELLITE_HIT_RADIUS):
        ctx.a1(True)
    if (
        hold_beam
        and in_reach
        and abs(enemy.bearing) <= tolerance + VUX_HOLD_BEAM_SLACK
        and ctx.me.battery >= VUX_HOLD_BEAM_BATTERY
    ):
        # The beam is on it (as at the close start): keep it there and burn
        # rather than turn away, until the battery runs low.
        ctx.point_at_opponent()
        ctx.plan.claim_thrust(False)
    # Limpets until the opponent is crippled, keeping laser battery when near.
    near = enemy.distance <= ctx.weapon(1).range + TYPICAL_HULL_RADIUS + VUX_LASER_RESERVE_MARGIN
    spare = ctx.me.battery - ctx.spec.a2_cost >= VUX_LASER_RESERVE
    # Satellites shoot limpets down unless busy shooting the Vux itself.
    shielded = bool(ctx.chmmr_satellites()) and not ctx.satellites_in_reach_of_me()
    if not disabled and not shielded and ctx.in_range(2) and (spare or not near):
        ctx.a2(True)
    if slingshot and not disabled and enemy.distance > VUX_SLINGSHOT_DISTANCE:
        ctx.coast_above_top_speed = True
        ctx.gravity_whip()
    if disabled:
        _vux_blind_side_approach(ctx)
    else:
        _vux_keep_away(ctx)


def vux(ctx: BaselineContext) -> None:
    _vux(ctx, slingshot=False, hold_beam=True)


def vux_keep_away(ctx: BaselineContext) -> None:
    """Default rules, but turn away even while the laser bears on the target."""

    _vux(ctx, slingshot=False, hold_beam=False)


def vux_slingshot(ctx: BaselineContext) -> None:
    """Default rules, plus a planet slingshot while the opponent is far off."""

    _vux(ctx, slingshot=True, hold_beam=True)


def vux_flee_or_chase(ctx: BaselineContext) -> None:
    """Earlier rules: flee faster ships, chase slower ones facing away."""

    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)
    enemy_faster = ctx.enemy_block.max_thrust >= ctx.spec.max_thrust
    if enemy_faster:
        ctx.avoid_opponent()
    if ctx.in_range(2):
        ctx.a2(True)
    if not enemy_faster:
        opponent_bearing = ctx.enemy_block.opponent_bearing
        facing_away = opponent_bearing is not None and abs(opponent_bearing) > 90.0
        if facing_away:
            ctx.approach_opponent()
        else:
            ctx.avoid_opponent()


def _tongue_reach(ctx: BaselineContext) -> float:
    """Tongue length from my centre plus the target's hull."""

    return ctx.weapon(2).range + TYPICAL_HULL_RADIUS


def _tongue_tolerance(ctx: BaselineContext) -> float:
    """Aim error at which the narrow tongue still touches the target hull."""

    half_width = ZOQ_FOT_PIK_TONGUE_WIDTH / 2.0 + TYPICAL_HULL_RADIUS
    return math.degrees(math.atan2(half_width, max(1.0, ctx.distance)))


def _tongue_viable(ctx: BaselineContext) -> bool:
    """Whether saving battery for the tongue can pay off against this opponent."""

    enemy = ctx.enemy
    if enemy is None or ctx.enemy_cloaked():
        return False
    if ctx.enemy_name in ZOQ_FOT_PIK_TONGUE_SKIPPED_SHIPS:
        return False
    outrunning = (
        ctx.enemy_block.max_thrust > ctx.spec.max_thrust and enemy.closing_speed < 0.0
    )
    return not outrunning


def zoq_fot_pik(ctx: BaselineContext) -> None:
    tongue_ready = ctx.me.battery >= ctx.spec.a2_cost
    tongue_viable = _tongue_viable(ctx)
    reach = _tongue_reach(ctx)
    in_tongue_reach = ctx.enemy is not None and ctx.distance <= reach
    if tongue_ready and in_tongue_reach:
        ctx.point_weapon(2)
        if ctx.pointing(2, tolerance=_tongue_tolerance(ctx)):
            ctx.a2(True)
    if (
        tongue_ready
        and tongue_viable
        and ctx.intercept_seconds(reach) <= ZOQ_FOT_PIK_LUNGE_SECONDS
    ):
        ctx.approach_opponent()
    if ctx.battery_at_least(0.30):
        ctx.approach_weapon_range(1)
    else:
        ctx.avoid_opponent()
    if ctx.in_range(1):
        ctx.point_weapon(1)
    # Spend only battery above the tongue's cost while the tongue can land.
    spare = ctx.me.battery - ctx.spec.a1_cost >= ctx.spec.a2_cost
    if ctx.in_range(1) and ctx.pointing(1) and (spare or not tongue_viable):
        ctx.a1(True)


def zoq_fot_pik_a1_first(ctx: BaselineContext) -> None:
    """Earlier rules: fire weapon 1 freely and tongue only when it is ready."""

    can_tongue = ctx.in_range(2) and ctx.me.battery >= ctx.spec.a2_cost
    if can_tongue:
        ctx.point_weapon(2)
    if can_tongue and ctx.pointing(2):
        ctx.a2(True)
    if ctx.battery_at_least(0.30):
        ctx.approach_weapon_range(1)
    else:
        ctx.avoid_opponent()
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


def pkunk_full_range(ctx: BaselineContext) -> None:
    """Earlier rules: fire at full reach and drift while recharging."""

    ctx.a2(True)
    if ctx.battery_at_least(0.50):
        ctx.approach_weapon_range(1)
    else:
        ctx.avoid_opponent()
    if ctx.in_range(1):
        ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


def generic(ctx: BaselineContext) -> None:
    """Fallback: close to weapon range, aim, and fire when aimed."""

    ctx.approach_weapon_range(1)
    ctx.point_weapon(1)
    if ctx.in_range(1) and ctx.pointing(1):
        ctx.a1(True)


STRATEGIES: dict[str, StrategyFunction] = {
    "Androsynth": androsynth,
    "Arilou": arilou,
    "Chenjesu": chenjesu,
    "Chmmr": chmmr,
    "Druuge": druuge,
    "Earthling": earthling,
    "Ilwrath": ilwrath,
    "KohrAh": kohr_ah,
    "KzerZa": kzer_za,
    "Mmrnmrhm": mmrnmrhm,
    "Melnorme": melnorme,
    "Mycon": mycon,
    "Orz": orz,
    "Pkunk": pkunk,
    "Shofixti": shofixti,
    "Slylandro": slylandro,
    "Spathi": spathi,
    "Supox": supox,
    "Syreen": syreen,
    "Thraddash": thraddash,
    "Umgah": umgah,
    "Utwig": utwig,
    "Vux": vux,
    "Yehat": yehat,
    "ZoqFotPik": zoq_fot_pik,
}


# Ships that let rule 1 dodge only some of the time: the Syreen must close
# in to sing, and the Utwig's shield turns hits into battery.
EVADE_GATES: dict[str, Callable[[BaselineContext], bool]] = {
    "Syreen": _syreen_may_evade,
    "Utwig": _utwig_may_evade,
}


def may_evade(ctx: BaselineContext) -> bool:
    """Whether rule 1 may dodge for this ship this decision."""

    gate = EVADE_GATES.get(ctx.ship_name)
    return gate is None or gate(ctx)


DEFAULT_VARIANT = "default"

# Alternative rule sets per ship, for comparing strategies in the arena.
# Training, opponents and battle mode always use ``STRATEGIES``.
STRATEGY_VARIANTS: dict[str, dict[str, StrategyFunction]] = {
    "Earthling": {"flee_whip": earthling_flee_whip},
    "Ilwrath": {"always_cloak": ilwrath_always_cloak, "chase": ilwrath_chase},
    "Pkunk": {
        "dodge": pkunk_dodge,
        "full_range": pkunk_full_range,
        "head_on": pkunk_head_on,
        "near_station": pkunk_near_station,
    },
    "Shofixti": {"glory_dive": shofixti_glory_dive},
    "Supox": {"no_slide": supox_no_slide},
    "Syreen": {
        "bold": syreen_bold,
        "full_reach": syreen_full_reach,
        "pressing": syreen_pressing,
    },
    "Thraddash": {"dodge_only": thraddash_dodge_only},
    "Vux": {
        "flee_or_chase": vux_flee_or_chase,
        "keep_away": vux_keep_away,
        "slingshot": vux_slingshot,
    },
    "Umgah": {"approach_cone": umgah_approach_cone, "hop_in": umgah_hop_in},
    "ZoqFotPik": {"a1_first": zoq_fot_pik_a1_first},
}


def strategy_for(ship_name: str, variant: str | None = None) -> StrategyFunction:
    if variant and variant != DEFAULT_VARIANT:
        variants = STRATEGY_VARIANTS.get(str(ship_name), {})
        if variant not in variants:
            known = ", ".join(sorted(variants)) or "none"
            raise KeyError(
                f"Unknown strategy variant {variant!r} for {ship_name}; known: {known}"
            )
        return variants[variant]
    return STRATEGIES.get(str(ship_name), generic)
