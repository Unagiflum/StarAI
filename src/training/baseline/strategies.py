"""Per-ship baseline strategies.

Each function lists one ship's rules in priority order (see ``engine``). The
shared rule 0, avoiding the planet, runs before every strategy.
"""

from __future__ import annotations

import math

from src.training.baseline.engine import BaselineContext, StrategyFunction
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
# Distances from the specification, in world units.
LONG_RANGE = 1000.0
FAR_RANGE = 2000.0
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


def arilou(ctx: BaselineContext) -> None:
    if ctx.imminent_threat(ARILOU_TELEPORT_SECONDS) is not None:
        ctx.a2(True)
    ctx.approach_weapon_range(1)
    if ctx.in_range(1) and ctx.me.battery >= ctx.spec.a1_cost + ctx.spec.a2_cost:
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


def earthling(ctx: BaselineContext) -> None:
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
    if ctx.me.orz_marines < 2 and ctx.me.crew > 2:
        # Weapon 1 and weapon 2 together launch a marine.
        ctx.a1(True)
        ctx.a2(True)
    ctx.a2(False)
    ctx.approach_weapon_range(1)
    # Aim the hull so the turret faces the target; never turn the turret.
    ctx.point_weapon(1)
    if ctx.pointing(1) and ctx.in_range(1):
        ctx.a1(True)


def pkunk(ctx: BaselineContext) -> None:
    # Insults are the Pkunk's only battery source; they never block firing.
    ctx.a2(True)
    a1_range = ctx.weapon(1).range
    charged = ctx.battery_at_least(0.50)
    if charged:
        # Close to where the short-lived shots reliably connect.
        ctx.approach_range(a1_range * PKUNK_FIRE_RANGE_FRACTION, key="a1_range")
    else:
        ctx.avoid_opponent()
        # Wait just outside weapon range so a recharge can strike at once.
        ctx.approach_range(
            a1_range * PKUNK_STATION_RANGE_FACTOR,
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


def supox(ctx: BaselineContext) -> None:
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


def _syreen(ctx: BaselineContext, *, cautious: bool, keep_away: bool) -> None:
    enemy = ctx.enemy
    song_viable = _song_viable(ctx)
    song_ready = ctx.me.battery >= ctx.spec.a2_cost
    if song_viable and song_ready and ctx.distance <= SYREEN_SONG_DISTANCE:
        ctx.a2(True)
    pods = ctx.view.crew_pods
    if pods:
        ctx.approach_object(min(pods, key=lambda pod: pod.distance))
    small = ctx.me.crew < SYREEN_GROWN_CREW
    outpaced = enemy is not None and ctx.enemy_block.max_thrust >= ctx.spec.max_thrust
    risky = cautious and small and ctx.enemy_name in SYREEN_CLOSE_DANGER_SHIPS and outpaced
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


def thraddash(ctx: BaselineContext) -> None:
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


def umgah(ctx: BaselineContext) -> None:
    """Hop backward past the opponent so it ends up in front of the cone.

    The backward hop (weapon 2) jumps 160 units per battery point but only
    while thrust is released; the cone (weapon 1) is free yet stops the
    all-at-once battery refill while it fires.
    """

    enemy = ctx.enemy
    me = ctx.me
    values = ctx.memory.values
    if me.battery <= UMGAH_HOP_RESERVE:
        values["umgah_recharging"] = True
    elif me.battery >= me.max_battery:
        values["umgah_recharging"] = False
    recharging = bool(values.get("umgah_recharging", False))
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


def yehat(ctx: BaselineContext) -> None:
    _shield_rules(ctx)


def vux(ctx: BaselineContext) -> None:
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


DEFAULT_VARIANT = "default"

# Alternative rule sets per ship, for comparing strategies in the arena.
# Training, opponents and battle mode always use ``STRATEGIES``.
STRATEGY_VARIANTS: dict[str, dict[str, StrategyFunction]] = {
    "Ilwrath": {"always_cloak": ilwrath_always_cloak, "chase": ilwrath_chase},
    "Pkunk": {"full_range": pkunk_full_range},
    "Shofixti": {"glory_dive": shofixti_glory_dive},
    "Syreen": {
        "bold": syreen_bold,
        "full_reach": syreen_full_reach,
        "pressing": syreen_pressing,
    },
    "Umgah": {"approach_cone": umgah_approach_cone},
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
