"""Per-ship baseline strategies.

Each function lists one ship's rules in priority order (see ``engine``). The
shared rule 0, avoiding the planet, runs before every strategy.
"""

from __future__ import annotations

import math

from src.training.baseline.engine import BaselineContext, StrategyFunction
from src.training.baseline.observation_view import bearing_of, wrap_degrees
from src.training.baseline.ship_specs import (
    FORM_XFORM,
    TYPICAL_HULL_RADIUS,
    weapon_spec,
)
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS

# Collision windows from the specification, in seconds. The engine adds one
# action span to each so a reaction is never a decision late.
ARILOU_TELEPORT_SECONDS = 0.1
SHIELD_SECONDS = 0.2
THRADDASH_AFTERBURNER_SECONDS = 0.5
UMGAH_CONE_THREAT_SECONDS = 1.0
UMGAH_RETRO_SECONDS = 0.2
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


def ilwrath(ctx: BaselineContext) -> None:
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


def shofixti(ctx: BaselineContext) -> None:
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


def syreen(ctx: BaselineContext) -> None:
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


def umgah(ctx: BaselineContext) -> None:
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
    glory_radius = weapon_spec("Shofixti", 2).range / 2.0
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
    "Pkunk": {"full_range": pkunk_full_range},
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
