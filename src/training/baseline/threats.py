"""Predict the enemy's shots in flight against moves of my own.

Every enemy object that costs something on contact (crew, battery, a
limpet's slowdown, a boarding marine), or within its laser's range for a
Kzer-Za fighter, becomes a ``ThreatTrack``.
Each candidate move is flown ahead with ``flight.ShipFlight``; the threats
are flown beside it (straight, or turning toward me at their own cadence
when they home) and a threat lands when its swept path comes within my
hull's reach. A move may also fire a shot at one threat, which removes it
when the shot meets it first and is strong enough to destroy it.

Coordinates are those of ``flight``: the observing ship's frame at the
start, ``x`` forward and ``y`` to the right, relative to its start position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

import src.const as const
from src.training.baseline.flight import ShipFlight, planet_gravity, turn_step_degrees
from src.training.baseline.observation_view import ObservedObject, bearing_of
from src.training.baseline.ship_specs import (
    PLANET_GRAVITY,
    ShipSpec,
    WeaponSpec,
    ability_spec,
    weapon_spec,
)

# A shot's own reach beyond the hull's: small shots are about this wide.
THREAT_SIZE = 10.0
# My shot destroys a threat when their swept centres pass this close, and a
# beam when it passes this close to the threat's centre.
SHOT_CONTACT = 20.0
BEAM_CONTACT = 14.0
# Weight of hazards that take no crew on contact: a limpet's lasting slowdown
# and a boarding marine are each worth about one crew; battery drain counts
# at this many crew per point.
LIMPET_WEIGHT = 1.0
MARINE_WEIGHT = 1.0
FIGHTER_WEIGHT = 1.0
DRAIN_WEIGHT = 0.25
# Enemy objects outside weapon 1, by ship, when they are not its weapon 2.
_OTHER_ABILITIES = {"Orz": "OrzA3"}
# Objects that chase me under their own logic although not flagged tracking.
_CHASING_ABILITIES = frozenset({"OrzA3", "ChenjesuA2"})
_EXTRA_WEIGHTS = {
    "VuxA2": LIMPET_WEIGHT,
    "OrzA3": MARINE_WEIGHT,
    "KzerZaA2": FIGHTER_WEIGHT,
}
# Objects that hurt from a distance rather than on contact: a Kzer-Za
# fighter fires its laser at a ship within the laser's range.
_STANDOFF_REACH = {
    "KzerZaA2": float(ability_spec("KzerZaA2Laser").range),
}


@dataclass(slots=True)
class ThreatTrack:
    """One enemy object in flight, with what touching it costs."""

    source: ObservedObject
    x: float
    y: float
    vx: float
    vy: float
    weight: float
    hit_points: float
    # Frames it has left; infinity when it never expires.
    life: float
    # Frames between heading steps when it homes on me, else 0.
    turn_period: int = 0
    inertial: bool = False
    # Centre distance at which it hurts me when it hurts from afar.
    standoff: float = 0.0
    # Positions of a straight threat, once flown: (frames, xs, ys).
    fixed_path: tuple[int, list[float], list[float]] | None = None

    @property
    def homing(self) -> bool:
        return self.turn_period > 0

    def contact(self, hull_reach: float) -> float:
        """Centre distance at which this threat lands on a hull of ``hull_reach``."""

        return max(hull_reach + THREAT_SIZE, self.standoff)


@dataclass(slots=True)
class MoveOutcome:
    """Crew a move is predicted to lose, and how narrowly the rest miss."""

    damage: float
    clearance: float
    # Index (in the track list) of the threat its shot removes, if any.
    removed: int | None = None
    hits: list[int] = field(default_factory=list)


def enemy_weapon_for(enemy_name: str | None, group: str, enemy_form: str) -> WeaponSpec | None:
    """Best guess at the ability an enemy object in ``group`` belongs to."""

    if enemy_name is None:
        return None
    if group == "enemy_a1":
        return weapon_spec(enemy_name, 1, enemy_form)
    other = _OTHER_ABILITIES.get(enemy_name)
    if other is not None:
        return ability_spec(other)
    weapon = weapon_spec(enemy_name, 2, enemy_form)
    if weapon.kind in ("projectile", "special_object"):
        return weapon
    return None


def threat_tracks(
    objects: Sequence[ObservedObject],
    own_velocity: tuple[float, float],
    *,
    enemy_name: str | None,
    enemy_form: str,
) -> list[ThreatTrack]:
    """Tracks for the enemy objects that cost something on contact."""

    tracks: list[ThreatTrack] = []
    own_forward, own_right = own_velocity
    for obj in objects:
        if not math.isfinite(obj.distance):
            continue
        weapon = enemy_weapon_for(enemy_name, obj.group, enemy_form)
        name = weapon.ability_name if weapon is not None else ""
        weight = max(0.0, -obj.crew_effect) + DRAIN_WEIGHT * max(0.0, -obj.battery_effect)
        weight += _EXTRA_WEIGHTS.get(name, 0.0)
        if weight <= 0.0:
            continue
        homing = weapon is not None and (weapon.tracking or name in _CHASING_ABILITIES)
        life = obj.remaining_seconds * const.FPS if obj.expires else math.inf
        tracks.append(
            ThreatTrack(
                source=obj,
                x=obj.forward,
                y=obj.right,
                vx=obj.velocity_forward + own_forward,
                vy=obj.velocity_right + own_right,
                weight=weight,
                hit_points=max(0.0, obj.hit_points),
                life=life,
                turn_period=(weapon.turn_wait + 1) if homing and weapon is not None else 0,
                inertial=bool(weapon is not None and weapon.inertial),
                standoff=_STANDOFF_REACH.get(name, 0.0),
            )
        )
    return tracks


def straight_line_danger(track: ThreatTrack, frames: int, reach: float) -> bool:
    """Whether ``track`` might matter within ``frames``: a homing threat
    near enough to arrive, or a straight one whose path passes within
    ``reach`` (beyond its own standoff) of where I am now, coasting."""

    source = track.source
    reach += track.standoff
    span = min(float(frames), track.life)
    if span <= 0.0:
        return False
    if track.homing:
        speed = math.hypot(track.vx, track.vy) + source.relative_speed
        return source.distance <= reach + speed * span
    closest_frame, miss = source.closest_approach()
    if closest_frame < 0.0:
        return source.distance <= reach
    return closest_frame <= span and miss <= reach


def arrival_frames(track: ThreatTrack, limit: int) -> int:
    """Frames worth flying to see how ``track`` ends: a homing threat
    until it expires, a straight one until its closest pass."""

    if track.homing:
        span = track.life
    else:
        closest_frame, _ = track.source.closest_approach()
        span = min(track.life, max(0.0, closest_frame))
    if not math.isfinite(span):
        return limit
    return max(1, min(limit, int(math.ceil(span))))


@dataclass(slots=True)
class FlownPath:
    """My positions, velocities and noses at frames 0..n of a move."""

    xs: list[float]
    ys: list[float]
    vxs: list[float]
    vys: list[float]
    noses: list[float]


# A controller returns ``(turn_error, thrust, fire)`` for the frame, or
# ``(turn_error, thrust, fire, hop)`` to hop ``hop`` units backward.
Controller = Callable[[ShipFlight], tuple]


def fly_move(
    spec: ShipSpec,
    velocity: tuple[float, float],
    frames: int,
    controller: Controller,
    *,
    planet: tuple[float, float] | None,
    recoil: tuple[float, float] | None = None,
) -> tuple[FlownPath, list[int]]:
    """Fly ``frames`` frames under ``controller``.

    The controller returns ``(turn_error, thrust, fire)`` for the frame,
    with an optional fourth value, a backward hop's length; a fire with
    ``recoil = (increment, max_speed)`` kicks the ship backward as the
    Druuge cannon does. Returns the path and the frames fired on.
    """

    flight = ShipFlight(spec, (0.0, 0.0), velocity, planet=planet)
    xs, ys = [0.0], [0.0]
    vxs, vys = [flight.vx], [flight.vy]
    noses = [0.0]
    fired: list[int] = []
    for frame in range(frames):
        controls = controller(flight)
        turn_error, thrust, fire = controls[0], controls[1], controls[2]
        if fire:
            fired.append(frame)
            if recoil is not None:
                flight.kick(180.0, recoil[0], recoil[1])
        flight.step(turn_error, thrust, hop=controls[3] if len(controls) > 3 else 0.0)
        xs.append(flight.x)
        ys.append(flight.y)
        vxs.append(flight.vx)
        vys.append(flight.vy)
        noses.append(flight.nose)
    return FlownPath(xs, ys, vxs, vys, noses), fired


def threat_path(
    track: ThreatTrack,
    path: FlownPath,
    *,
    planet: tuple[float, float] | None,
) -> tuple[list[float], list[float]]:
    """The threat's positions at frames 0..n (cut short where it expires)."""

    frames = len(path.xs) - 1
    last = frames if math.isinf(track.life) else min(frames, int(track.life))
    homing = track.homing
    if not homing and track.fixed_path is not None and track.fixed_path[0] == last:
        # A straight threat's path does not depend on my move.
        return track.fixed_path[1], track.fixed_path[2]
    x, y, vx, vy = track.x, track.y, track.vx, track.vy
    xs, ys = [x], [y]
    pull = const.GRAVITY_MULTIPLIER * PLANET_GRAVITY if track.inertial and planet else 0.0
    if not homing and pull <= 0.0:
        for frame in range(1, last + 1):
            xs.append(x + vx * frame)
            ys.append(y + vy * frame)
        track.fixed_path = (last, xs, ys)
        return xs, ys
    speed = math.hypot(vx, vy)
    heading = bearing_of(vx, vy)
    step = turn_step_degrees()
    fine_step = float(const.TURN_ANGLE)
    next_turn = 0
    for frame in range(last):
        if track.homing and frame >= next_turn and speed > 0.0:
            target = bearing_of(path.xs[frame] - x, path.ys[frame] - y)
            error = (target - heading + 180.0) % 360.0 - 180.0
            if abs(error) >= fine_step / 2.0:
                heading += math.copysign(min(step, abs(error)), error)
                next_turn = frame + track.turn_period
            radians = math.radians(heading)
            vx, vy = speed * math.cos(radians), speed * math.sin(radians)
        if pull > 0.0:
            px, py = x - planet[0], y - planet[1]
            gx, gy = planet_gravity(px, py, math.hypot(px, py), pull)
            vx += gx
            vy += gy
        x += vx
        y += vy
        xs.append(x)
        ys.append(y)
    if not homing:
        track.fixed_path = (last, xs, ys)
    return xs, ys


def _segment_closest(
    ax: float, ay: float, bx: float, by: float
) -> float:
    """Closest distance to the origin of the segment from ``a`` to ``b``."""

    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / length_squared))
    return math.hypot(ax + dx * t, ay + dy * t)


def first_contact(
    txs: list[float],
    tys: list[float],
    path: FlownPath,
    reach: float,
) -> tuple[int | None, float]:
    """Frame the threat's swept path first touches me, and its closest pass."""

    closest = math.hypot(txs[0] - path.xs[0], tys[0] - path.ys[0])
    if closest <= reach:
        return 0, closest
    previous_x = txs[0] - path.xs[0]
    previous_y = tys[0] - path.ys[0]
    for frame in range(1, len(txs)):
        rx = txs[frame] - path.xs[frame]
        ry = tys[frame] - path.ys[frame]
        distance = _segment_closest(previous_x, previous_y, rx, ry)
        if distance < closest:
            closest = distance
        if distance <= reach:
            return frame, closest
        previous_x, previous_y = rx, ry
    return None, closest


def shot_meets(
    txs: list[float],
    tys: list[float],
    path: FlownPath,
    fire_frame: int,
    weapon: WeaponSpec,
    direction: float,
) -> int | None:
    """Frame my shot, fired at ``fire_frame`` toward ``direction`` (degrees
    in the starting frame), meets the threat, if it does before expiring."""

    if fire_frame >= len(txs):
        return None
    radians = math.radians(direction)
    ux, uy = math.cos(radians), math.sin(radians)
    sx, sy = path.xs[fire_frame], path.ys[fire_frame]
    if weapon.kind == "laser":
        rx, ry = txs[fire_frame] - sx, tys[fire_frame] - sy
        along = rx * ux + ry * uy
        if 0.0 <= along <= weapon.range and abs(rx * uy - ry * ux) <= BEAM_CONTACT:
            return fire_frame
        return None
    carry = weapon.parent_velocity
    vx = ux * weapon.speed + carry * path.vxs[fire_frame]
    vy = uy * weapon.speed + carry * path.vys[fire_frame]
    life = weapon.life_frames if weapon.life_frames > 0.0 else math.inf
    previous_x = txs[fire_frame] - sx
    previous_y = tys[fire_frame] - sy
    for frame in range(fire_frame + 1, len(txs)):
        elapsed = frame - fire_frame
        if elapsed > life:
            return None
        rx = txs[frame] - (sx + vx * elapsed)
        ry = tys[frame] - (sy + vy * elapsed)
        if _segment_closest(previous_x, previous_y, rx, ry) <= SHOT_CONTACT:
            return frame
        previous_x, previous_y = rx, ry
    return None


def judge_move(
    tracks: Sequence[ThreatTrack],
    path: FlownPath,
    reach: float,
    *,
    planet: tuple[float, float] | None,
    shot: tuple[int, WeaponSpec, float, int] | None = None,
) -> MoveOutcome:
    """Crew lost on ``path``, the narrowest miss, and the threat ``shot``
    removes: ``(fire_frame, weapon, direction, track_index)``."""

    damage = 0.0
    clearance = math.inf
    removed = None
    hits: list[int] = []
    for index, track in enumerate(tracks):
        txs, tys = threat_path(track, path, planet=planet)
        touch = track.contact(reach)
        contact, closest = first_contact(txs, tys, path, touch)
        if shot is not None and shot[3] == index:
            fire_frame, weapon, direction, _ = shot
            met = shot_meets(txs, tys, path, fire_frame, weapon, direction)
            if (
                met is not None
                and (contact is None or met < contact)
                and weapon.damage >= track.hit_points
            ):
                removed = index
                continue
        if contact is not None:
            damage += track.weight
            hits.append(index)
        else:
            clearance = min(clearance, closest - touch)
    return MoveOutcome(damage=damage, clearance=clearance, removed=removed, hits=hits)
