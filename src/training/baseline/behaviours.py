"""Per-ship parameters for the shared baseline behaviours.

The shared behaviours in ``engine`` read each ship's entry here instead of
naming ships themselves, and ``strategies.shared`` runs a ship's whole
strategy from its entry. A ship missing from ``SHIP_BEHAVIOURS`` takes the
defaults. Ship facts (weapon mechanics, hull sizes) stay in ``ship_specs``;
this table holds only choices the arena tunes.

Behaviours run in priority order, each claiming only the controls it sets
(``engine.ControlPlan``):

0. avoid the planet (rule 0);
1. avoid damage (rule 1): dodge, shoot down, or run from threats;
2. specials: unique abilities, named in ``specials``;
3. mode switch, named in ``form_switch``;
4. recharge: keep out of the opponent's reach while the battery is low;
5. hold the distance band;
6. late specials: unique rules that depend on the movement chosen (a
   turret that turns only while the hull is held on a course);
7. fire each weapon by its ``WeaponUse``;
8. keep moving (last rule).

Specials and mode switches are functions in ``strategies.SPECIAL_RULES`` and
``strategies.FORM_SWITCHES``, named here so this table needs no imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

# What a band does with steering while the opponent is inside it and not
# closing in fast: leave it free (for aiming), face the opponent without
# thrust, or fly across the line to the opponent.
BAND_INSIDE_FREE = "free"
BAND_INSIDE_FACE = "face"
BAND_INSIDE_ORBIT = "orbit"
BAND_INSIDE_POSTURES = (BAND_INSIDE_FREE, BAND_INSIDE_FACE, BAND_INSIDE_ORBIT)
# Close in from beyond the band no faster than this closing speed, in world
# units per frame.
BAND_APPROACH_SPEED = 15.0


@dataclass(frozen=True, slots=True)
class Band:
    """A distance band to hold the opponent in (``engine.hold_band``).

    ``near`` and ``far`` are centre distances. With ``outside_reach_margin``
    set, ``near`` rises to just outside the opponent's own reach (its
    weapons' reach plus a hull plus the margin) when that is farther, and
    ``far`` with it to keep ``min_width``. ``far`` of None never closes in:
    the band is "as far as possible". ``approach_speed`` caps the closing
    speed when coming in from beyond ``far``. ``inside`` is the posture
    within the band (``BAND_INSIDE_POSTURES``). ``run_offset`` is the
    heading, in degrees off the opponent's bearing, it runs from a closer
    on: 180 straight away, less to veer off its line. With ``back_off``
    the near edge works as ``approach_range``'s does: inside it the ship
    steers ``run_offset`` off the bearing whether or not the opponent is
    closing, with no prediction and no release distance.
    """

    near: float
    far: float | None
    outside_reach_margin: float | None = None
    min_width: float = 150.0
    approach_speed: float = BAND_APPROACH_SPEED
    inside: str = BAND_INSIDE_FREE
    run_offset: float = 180.0
    back_off: bool = False

    def __post_init__(self) -> None:
        if self.inside not in BAND_INSIDE_POSTURES:
            raise ValueError(f"Unknown band posture {self.inside!r}")


# How a weapon is fired (``engine.fire_weapon``):
# aimed: turn the gun onto the target, and fire once on it within range;
# launch: a shot that aims itself, fired within range whenever it would
#   outrun my own motion along its gun;
# defend: pressed when a hit is about to land, or after crew was lost
#   (shields, a teleport, an afterburner dodge);
# special: left to the ship's specials.
FIRE_AIMED = "aimed"
FIRE_LAUNCH = "launch"
FIRE_DEFEND = "defend"
FIRE_SPECIAL = "special"
FIRE_KINDS = (FIRE_AIMED, FIRE_LAUNCH, FIRE_DEFEND, FIRE_SPECIAL)


@dataclass(frozen=True, slots=True)
class WeaponUse:
    """When to fire one weapon.

    ``range_fraction`` scales the weapon's reach. ``reserve`` is battery
    that must be left after the shot (for another weapon, say); with 0 an
    aimed shot doesn't check the battery. ``aim`` lets an aimed weapon turn
    the nose; without it, it fires only when already on target.
    ``satellites`` aims at the Chmmr satellite ``satellite_shot`` picks
    while one is in reach. ``tolerance`` overrides the aim tolerance, in
    degrees. A defending weapon presses when an enemy object will touch
    the ship within ``threat_seconds``, or, with ``on_crew_loss``, when crew
    was lost since the last decision.
    """

    kind: str = FIRE_AIMED
    range_fraction: float = 1.0
    reserve: float = 0.0
    aim: bool = True
    satellites: bool = False
    tolerance: float | None = None
    threat_seconds: float = 0.2
    on_crew_loss: bool = False

    def __post_init__(self) -> None:
        if self.kind not in FIRE_KINDS:
            raise ValueError(f"Unknown fire kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class ShipBehaviour:
    """How the shared behaviours apply to one ship.

    Avoid damage (rule 1):
        ``evade``: rule 1 runs at all. A strategy may still gate it per
        decision (``strategies.EVADE_GATES``).
        ``evade_headings``: rule 1 may fly headings to dodge. The Umgah
        turns and accelerates too slowly, and hops instead.
        ``shoot_down``: rule 1 may turn to shoot a threat down with
        weapon 1. Off for homing, parked or self-aiming weapons, and where
        the arena showed shoot-downs cost more than they saved.
        ``run_from_threats``: rule 1 also weighs flying straight away from
        the threats.
    Avoid the planet (rule 0):
        ``planet_horizon_frames``: how far ahead rule 0 follows the coasting
        path; None derives it from the ship's turn and acceleration.
    Specials and form:
        ``specials``: names of unique rules (``strategies.SPECIAL_RULES``)
        run before movement, in order.
        ``form_switch``: name of the ship's mode-switch rule
        (``strategies.FORM_SWITCHES``).
        ``late_specials``: unique rules run after recharge or the band,
        before the weapons, for rules that depend on whether movement
        claimed the steering.
    Recharge:
        ``recharge_battery``: below this battery, the ship recharges: it
        keeps out of the opponent's weapon range (``recharge_retreat``)
        instead of holding its band. None leaves it off.
        ``recharge_resume``: the battery at which recharging ends; None
        ends it at ``recharge_battery``.
        ``recharge_run_offset``: the heading it runs from a closer on while
        recharging, in degrees off the opponent's bearing (180: straight
        away).
    Distance band:
        ``band``: the ``Band`` ``hold_band`` keeps the opponent in; None
        leaves it off.
    Weapons:
        ``weapons``: a ``WeaponUse`` (or None, unused) for weapon 1, then
        weapon 2.
    Keep moving (last rule):
        ``keep_moving``: the rule runs. Off for the Slylandro, at constant
        speed, and the Umgah, whose hops leave it at rest by design.
        ``keep_moving_takes_aim``: with turning claimed only to aim, the
        rule may take it back while weapon 1 could not fire anyway. The
        slow turners that gained in the arena.
    Speed:
        ``coast_above_top_speed``: coast rather than thrust above top
        speed. Across every ship the arena showed no gain, so only the
        afterburner ship uses it.
    Evaluation:
        ``harm_taken_weight``: λ, the weight on harm taken in the arena's λ
        score, which only breaks near-ties in harm dealt. Set explicitly
        when a ship moves: low for ships that can escape (so running away
        doesn't pay), higher for ships that can't. Unmoved ships use 0.5.
    Opponents:
        ``opponents``: per-opponent overrides, mapping an opponent's name to
        the fields that differ against it (``ship_behaviour``).
    """

    evade: bool = True
    evade_headings: bool = True
    shoot_down: bool = True
    run_from_threats: bool = False
    planet_horizon_frames: int | None = None
    specials: tuple[str, ...] = ()
    form_switch: str | None = None
    late_specials: tuple[str, ...] = ()
    recharge_battery: float | None = None
    recharge_resume: float | None = None
    recharge_run_offset: float = 180.0
    band: Band | None = None
    weapons: tuple[WeaponUse | None, ...] = ()
    keep_moving: bool = True
    keep_moving_takes_aim: bool = False
    coast_above_top_speed: bool = False
    harm_taken_weight: float = 0.5
    opponents: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def weapon_use(self, action: int) -> WeaponUse | None:
        index = int(action) - 1
        return self.weapons[index] if 0 <= index < len(self.weapons) else None


DEFAULT_BEHAVIOUR = ShipBehaviour()

SHIP_BEHAVIOURS: dict[str, ShipBehaviour] = {
    # Pass 1 at the default perception delay, by the harm ranking: the
    # blazer is the mode switch (blaze in within 1,000 on half a battery,
    # ram while it lasts); bubbles home, so they launch keeping 9 battery
    # back; it keeps the opponent at least 1,000 off, veering 135 degrees
    # from a closer, and below half its 24 battery keeps out of reach.
    # Against a Chmmr it saves a full battery for one ram from close in:
    # bubbles don't get through its satellites (Phase 6 ship checks).
    # λ stays 0.5: slow in its normal form, it escapes only by blazing,
    # which spends the battery the bubbles need.
    "Androsynth": ShipBehaviour(
        harm_taken_weight=0.5,
        shoot_down=False,
        form_switch="androsynth_blazer",
        recharge_battery=12.0,
        band=Band(1000.0, None, run_offset=135.0),
        weapons=(WeaponUse(FIRE_LAUNCH, reserve=9.0), None),
        opponents={"Chmmr": {"specials": ("androsynth_chmmr",)}},
    ),
    "Arilou": ShipBehaviour(shoot_down=False),
    "Chenjesu": ShipBehaviour(keep_moving_takes_aim=True),
    "Earthling": ShipBehaviour(shoot_down=False),
    # The Kohr-Ah's shoot-downs cut crew lost by 0.9/min but crew dealt by
    # 1.3/min over 30 seeds.
    "KohrAh": ShipBehaviour(shoot_down=False),
    "Mmrnmrhm": ShipBehaviour(keep_moving_takes_aim=True),
    "Mycon": ShipBehaviour(shoot_down=False),
    "Slylandro": ShipBehaviour(keep_moving=False),
    # Pass 1 at the default perception delay, by the harm ranking: the band
    # runs from 0.7 of the gun's 1,080 reach to all of it, closing in at up
    # to 60, with no recharging; the rear missile homes, so it launches
    # whenever the opponent is in its reach. Against the Vux it closes in to
    # 300-756 at full speed and veers off at 110 degrees rather than running
    # into its limpets. The earlier per-opponent settings (Chmmr, Earthling,
    # Pkunk, Thraddash, Utwig) dealt less harm and were dropped.
    # λ is low: the Spathi is cheap and outruns nearly everything, so a high
    # λ would reward running away.
    "Spathi": ShipBehaviour(
        harm_taken_weight=0.25,
        band=Band(756.0, 1080.0, approach_speed=60.0),
        weapons=(WeaponUse(FIRE_AIMED), WeaponUse(FIRE_LAUNCH)),
        opponents={
            "Vux": {
                "band": Band(300.0, 756.0, approach_speed=math.inf, run_offset=110.0),
            },
        },
    ),
    "Thraddash": ShipBehaviour(coast_above_top_speed=True),
    "Umgah": ShipBehaviour(evade_headings=False, shoot_down=False, keep_moving=False),
}

# (id of the base entry, opponent) -> (base entry, entry with the overrides
# applied). The base is kept so a reused id is never served stale.
_AGAINST: dict[tuple[int, str], tuple[ShipBehaviour, ShipBehaviour]] = {}


def ship_behaviour(
    ship_name: str | None,
    opponent: str | None = None,
    entry: ShipBehaviour | None = None,
) -> ShipBehaviour:
    """The ship's behaviour, with its overrides against ``opponent`` applied.

    ``entry`` replaces the ship's table entry, for flying candidate
    settings in the arena.
    """

    base = entry if entry is not None else SHIP_BEHAVIOURS.get(str(ship_name), DEFAULT_BEHAVIOUR)
    overrides = base.opponents.get(str(opponent)) if opponent is not None else None
    if not overrides:
        return base
    key = (id(base), str(opponent))
    cached = _AGAINST.get(key)
    if cached is None or cached[0] is not base:
        cached = (base, replace(base, **overrides))
        _AGAINST[key] = cached
    return cached[1]
