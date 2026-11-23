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
# thrust, fly across the line to the opponent, or fly straight at it.
BAND_INSIDE_FREE = "free"
BAND_INSIDE_FACE = "face"
BAND_INSIDE_ORBIT = "orbit"
BAND_INSIDE_CHASE = "chase"
BAND_INSIDE_POSTURES = (BAND_INSIDE_FREE, BAND_INSIDE_FACE, BAND_INSIDE_ORBIT, BAND_INSIDE_CHASE)
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
    was lost since the last decision. ``fire_share`` below 1 lets an aimed
    or launched shot that would fire go ahead only on that share of
    decisions, chosen by a hash of the distance and bearing to the target
    (``engine.fire_gate_open``) so a shielding opponent can't time its
    shield to the fire; fights stay repeatable.
    """

    kind: str = FIRE_AIMED
    range_fraction: float = 1.0
    reserve: float = 0.0
    aim: bool = True
    satellites: bool = False
    tolerance: float | None = None
    threat_seconds: float = 0.2
    on_crew_loss: bool = False
    fire_share: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in FIRE_KINDS:
            raise ValueError(f"Unknown fire kind {self.kind!r}")
        if not 0.0 < self.fire_share <= 1.0:
            raise ValueError(f"fire_share must be in (0, 1], not {self.fire_share!r}")


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


# Collision window from the specification: the Arilou teleports from a hit
# due within this many seconds.
ARILOU_TELEPORT_SECONDS = 0.1

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
    # Pass 1 at the default perception delay, by the harm ranking: close in
    # at full speed beyond the laser's 400 reach and, inside 200, steer 110
    # degrees off the bearing (a back-off band, as approach_range did); the
    # laser aims itself, so it launches keeping a teleport's 3 battery; the
    # teleport defends against hits due within 0.1 s. Against a Chmmr it
    # holds outside the laser and fires as satellites cross the beam.
    # λ stays 0.5: it escapes by teleporting, but only from a hit about to
    # land, not by running.
    "Arilou": ShipBehaviour(
        harm_taken_weight=0.5,
        shoot_down=False,
        band=Band(200.0, 400.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(
            WeaponUse(FIRE_LAUNCH, reserve=3.0),
            WeaponUse(FIRE_DEFEND, threat_seconds=ARILOU_TELEPORT_SECONDS),
        ),
        opponents={"Chmmr": {"specials": ("arilou_chmmr",)}},
    ),
    # Pass 1 at the default perception delay, by the harm ranking: keep up
    # to 4 Dogi out, and aim the crystal at the opponent every decision,
    # holding it while it closes; keep moving does the rest. No band: every
    # band that moved the ship first left the crystal pointing away and
    # dealt far less harm. λ is high: costly and outrunning nothing, it
    # can't stall, so λ only picks the safer of two equally harmful plans.
    "Chenjesu": ShipBehaviour(
        harm_taken_weight=1.0,
        keep_moving_takes_aim=True,
        specials=("chenjesu_dogi", "chenjesu_crystal"),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: no band
    # (keep moving only), the cannon aimed, recoiling away from close
    # escorts and limpets of the ships that send them, and trading crew for
    # battery when a shot can't be afforded; against the Chenjesu it shoots
    # the nearest Dogi first. Every band dealt far less harm: the recoil
    # already keeps its distance. λ is high: slow, it can't outrun anything.
    # Pass 1 at the default perception delay, by the harm ranking: close in
    # at full speed to the laser's 600 reach and inside it face the
    # opponent without thrust (a back-off band that never backs off); the
    # tractor pulls the opponent into reach (a special); the laser is aimed.
    # λ is high: the costliest ship, it never runs, so λ only picks the
    # safer of two equally harmful plans.
    "Chmmr": ShipBehaviour(
        harm_taken_weight=1.0,
        specials=("chmmr_tractor",),
        band=Band(0.0, 600.0, approach_speed=math.inf, inside=BAND_INSIDE_FACE, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED), None),
    ),
    "Druuge": ShipBehaviour(
        harm_taken_weight=1.0,
        specials=("druuge_recoil_escape", "druuge_crew_battery"),
        weapons=(WeaponUse(FIRE_AIMED), None),
        opponents={
            "Chenjesu": {
                "specials": ("druuge_recoil_escape", "druuge_dogi_targets", "druuge_crew_battery")
            }
        },
    ),
    # Pass 1 at the default perception delay, by the harm ranking: point
    # defence only at what matters, nukes from a standstill turn, and a
    # planet whip away from the opponent are specials; it holds 1,000-2,160
    # (the nuke's reach), facing the opponent inside, veering 135 degrees
    # from a closer, coasting above top speed. Against a Chmmr it circles
    # across the bearing, out of the tractor's drag, nuking when it can.
    # λ stays 0.5: slow, it keeps its distance only by turning early.
    "Earthling": ShipBehaviour(
        harm_taken_weight=0.5,
        shoot_down=False,
        coast_above_top_speed=True,
        specials=("earthling_point_defense", "earthling_nuke", "earthling_whip"),
        band=Band(1000.0, 2160.0, inside=BAND_INSIDE_FACE, run_offset=135.0),
        opponents={
            "Chmmr": {
                "specials": ("earthling_point_defense", "earthling_nuke"),
                "band": Band(0.0, None, inside=BAND_INSIDE_ORBIT, run_offset=135.0),
            }
        },
    ),
    # Pass 1 at the default perception delay, by the harm ranking: the
    # flame (from the cloak, or on target) and cloaking while the fade can
    # finish out of reach are specials, as are waiting for a faster
    # opponent that isn't coming and following one pulling out of reach;
    # it closes in at full speed to the flame's 300 reach (a back-off band
    # with no near edge, as approach_range did). Equal to the earlier rules.
    # λ stays 0.5: it can't outrun much, but the cloak hides its approach.
    "Ilwrath": ShipBehaviour(
        harm_taken_weight=0.5,
        specials=("ilwrath_flame_cloak", "ilwrath_hang_back", "ilwrath_follow"),
        band=Band(0.0, 300.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: gas when
    # the opponent or any enemy object is within its reach, and aim the saw
    # every decision, keeping a flying saw moving while it closes (both
    # specials); keep moving does the rest. No band: every band that moved
    # the ship first left the saw aimed away. No shoot-downs: they cut crew
    # lost by 0.9/min but crew dealt by 1.3/min over 30 seeds. λ is high:
    # costly and slow, it can't stall.
    "KohrAh": ShipBehaviour(
        harm_taken_weight=1.0,
        shoot_down=False,
        specials=("kohr_ah_gas", "kohr_ah_saw"),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: fighters
    # (a special) and the gun aimed, as before; a back-off band closes in at
    # full speed to 1,000 and faces the opponent inside it, in place of
    # flying straight at it. Level with the earlier rules on harm dealt (+3%).
    # λ is high: the slowest ship with the most crew, it can't stall.
    "KzerZa": ShipBehaviour(
        harm_taken_weight=1.0,
        specials=("kzer_za_fighters",),
        band=Band(0.0, 1000.0, approach_speed=math.inf, inside=BAND_INSIDE_FACE, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED), None),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: the
    # charged shot (a special) released on target, aimed late; confusion
    # fired when aimed, without turning the nose. A back-off band closes in
    # at full speed to 1,400 and steers 110 degrees off the bearing inside
    # 800, in place of approach_range to the gun's 1,620. λ is middling: it
    # keeps its range but can't outrun the fast ships.
    "Melnorme": ShipBehaviour(
        harm_taken_weight=0.5,
        specials=("melnorme_charge",),
        late_specials=("melnorme_aim",),
        band=Band(800.0, 1400.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(None, WeaponUse(FIRE_AIMED, aim=False)),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: the forms
    # as the mode switch (the X-form within its laser's reach, the Y-wing
    # beyond), the X-form lasers flying at the opponent and the Y-wing
    # missiles (a special), facing the opponent late. No band and no
    # fleeing: the earlier Y-wing fled a closing opponent, which kept harm
    # taken down but dealt far less. λ is low: the Y-wing outruns most
    # ships, so running away mustn't pay.
    "Mmrnmrhm": ShipBehaviour(
        harm_taken_weight=0.25,
        keep_moving_takes_aim=True,
        form_switch="mmrnmrhm_forms",
        specials=("mmrnmrhm_attack",),
        late_specials=("face_opponent",),
    ),
    # Pass 1 at the default perception delay, by the harm ranking:
    # regeneration (a special), the planet whip when slow and the plasmoid
    # aimed late, and no movement rule of its own: keep moving does the
    # rest. The earlier rules also fled a closing opponent within 2,000;
    # without it the ship deals as much and takes less. λ is high: slow,
    # it can't run.
    "Mycon": ShipBehaviour(
        harm_taken_weight=1.0,
        shoot_down=False,
        specials=("mycon_regenerate",),
        late_specials=("gravity_whip_when_slow", "mycon_plasmoid"),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: marines
    # (a special) and the turret late. A back-off band closes in at full
    # speed to the gun's 1,296 and steers 110 degrees off the bearing inside
    # 907, as approach_range did; kiting opponents are held at 700-1,100 and
    # a Chmmr at 1,550-1,800, as before. Level with the earlier rules.
    # λ is middling: it can't outrun the fast ships.
    "Orz": ShipBehaviour(
        harm_taken_weight=0.5,
        specials=("orz_marines",),
        late_specials=("orz_turret",),
        band=Band(907.2, 1296.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        opponents={
            **{
                name: {"band": Band(700.0, 1100.0)}
                for name in ("Arilou", "Druuge", "KzerZa", "Orz", "Pkunk", "Shofixti", "Spathi", "Syreen", "Yehat")
            },
            "Chmmr": {"band": Band(1550.0, 1800.0)},
        },
    ),
    # Pass 1 at the default perception delay, by the harm ranking: insults
    # (its only battery source) and passing beside the opponent while
    # charged are specials, the nearest gun aimed late; below 4 of 12 it
    # recharges outside the opponent's reach. Level with the earlier rules.
    # λ is low: among the fastest ships, it can always run.
    "Pkunk": ShipBehaviour(
        harm_taken_weight=0.25,
        specials=("pkunk_insult", "pkunk_pass"),
        late_specials=("pkunk_gun",),
        recharge_battery=4.0,
    ),
    # Pass 1 at the default perception delay, by the harm ranking: the
    # glory device a special, the gun aimed; from an empty battery it
    # recharges outside the opponent's reach until 4. A back-off band closes
    # in at full speed to 700 and steers 110 degrees off the bearing inside
    # 400, in place of approach_range to the gun's 864. Level with the
    # earlier rules. λ is low: fast, and its glory blast spends its own crew
    # on purpose.
    "Shofixti": ShipBehaviour(
        harm_taken_weight=0.25,
        specials=("shofixti_glory",),
        recharge_battery=1.0,
        recharge_resume=4.0,
        band=Band(400.0, 700.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED), None),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: harvest
    # asteroids throughout (a special, flying at the nearest below 10%
    # battery); lightning seeks its target, so it launches within its 1,152
    # reach. A back-off band closes in at full speed to 300 and steers 110
    # degrees off the bearing inside 150, in place of approach_range to the
    # lightning's reach: the lightning lands far more often from close in.
    # Against a Chmmr it circles 700-850 out, firing within 1,000.
    # Constant speed, so keep moving is off. λ is low: among the fastest
    # ships, and the tractor can't hold it, so it can always run.
    "Slylandro": ShipBehaviour(
        harm_taken_weight=0.25,
        keep_moving=False,
        specials=("slylandro_harvest",),
        band=Band(150.0, 300.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_LAUNCH), None),
        opponents={
            "Chmmr": {
                "specials": ("slylandro_low_harvest",),
                "band": Band(700.0, 850.0, inside=BAND_INSIDE_ORBIT),
                "weapons": (WeaponUse(FIRE_LAUNCH, range_fraction=1000.0 / 1152.0), None),
            },
        },
    ),
    # Pass 1: the slides are its movement and fire (a special), holding the
    # earlier band (600-900, or just outside a shorter-reaching opponent
    # while its gun still reaches) with no table band; fight for fight as
    # before. Table bands with a reach margin don't stop at the gun's reach,
    # and held it out of range. λ is middling: it slides out of the way but
    # can't outrun the fast ships.
    "Supox": ShipBehaviour(harm_taken_weight=0.5, specials=("supox_slides",)),
    # Pass 1 at the default perception delay, by the harm ranking: sing
    # within 300 when the song can take crew, collect crew pods, and with
    # the song ready close in to sing unless that is too risky while the
    # crew is small (specials); the gun is aimed late, keeping the song's 5
    # battery spare. Below the song's cost it keeps out of reach (level with
    # not recharging); otherwise a back-off band closes in at full speed to
    # 900 and steers 110 degrees off inside 500, nearer than the gun's
    # 1,080 that approach_range held. Against a Chmmr, the earlier rule.
    # λ is middling: it can't outrun the fast ships.
    "Syreen": ShipBehaviour(
        harm_taken_weight=0.5,
        specials=("syreen_song", "syreen_pods", "syreen_close_in"),
        late_specials=("syreen_gun",),
        recharge_battery=5.0,
        band=Band(500.0, 900.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        opponents={
            "Chmmr": {
                "specials": ("syreen_chmmr",),
                "late_specials": (),
                "recharge_battery": None,
                "band": None,
            }
        },
    ),
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
    # Pass 1: the Thraddash, Umgah and Vux run their whole earlier rules as
    # one special, fight for fight as before; their movement is split out in
    # pass 2. λ low for the Thraddash, which outruns most ships on its
    # afterburner; middling for the Umgah, which escapes only by hopping
    # backwards; high for the Vux, slow and unable to run.
    # Pass 2, split from its whole earlier rules: burning forward from side
    # or rear shots, leading a closing opponent or homing escorts through
    # the trail tail first (holding fire), and dashing in from far off on
    # the afterburner are specials; a back-off band closes in to 1,400 and
    # steers 110 degrees off inside 900, as approach_range did; the gun is
    # aimed within 1,400 keeping 8 battery for the afterburner. Against a
    # Chmmr, the earlier rule. λ is low: it outruns most ships on its
    # afterburner.
    "Thraddash": ShipBehaviour(
        harm_taken_weight=0.25,
        coast_above_top_speed=True,
        specials=("thraddash_dodge_burn", "thraddash_trail"),
        band=Band(900.0, 1400.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED, range_fraction=1400.0 / 1620.0, reserve=8.0), None),
        opponents={
            "Chmmr": {
                "specials": ("thraddash_dodge_burn", "thraddash_chmmr"),
                "band": None,
                "weapons": (),
            },
        },
    ),
    "Umgah": ShipBehaviour(
        harm_taken_weight=0.5,
        evade_headings=False,
        shoot_down=False,
        keep_moving=False,
        specials=("umgah_rules",),
    ),
    # Pass 1 at the default perception delay, by the harm ranking: the
    # shield defends against a hit due within 0.2 s or after crew was lost
    # (against the Shofixti also wherever its blast would kill); the bolts
    # are aimed. A back-off band closes in at full speed to 900 and steers
    # 110 degrees off inside 500, nearer than the bolts' 1,080 that
    # approach_range held. Against the Spathi it keeps that earlier band
    # (756-1,080), which deals more than twice the harm; against a Chmmr,
    # the earlier rule. λ is middling: heavy and no faster than most, it
    # can't run.
    "Utwig": ShipBehaviour(
        harm_taken_weight=0.5,
        band=Band(500.0, 900.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED), WeaponUse(FIRE_DEFEND, threat_seconds=0.2, on_crew_loss=True)),
        opponents={
            "Shofixti": {"specials": ("shield_glory",)},
            "Spathi": {"band": Band(756.0, 1080.0, approach_speed=math.inf, run_offset=110.0, back_off=True)},
            "Chmmr": {"specials": ("utwig_chmmr",), "band": None, "weapons": ()},
        },
    ),
    "Vux": ShipBehaviour(harm_taken_weight=1.0, specials=("vux_rules",)),
    # Pass 1 at the default perception delay, by the harm ranking: the
    # shield defends against a hit due within 0.2 s or after crew was lost;
    # the guns are aimed. A back-off band closes in at full speed to 700,
    # near the guns' 720, and steers 110 degrees off only inside 100, where
    # approach_range backed off inside 504. Against the Shofixti it keeps
    # that earlier band, shielding wherever its blast would kill (closer in
    # it took still more harm); against a Chmmr, the earlier rule. λ is
    # middling: slow, it can't run.
    "Yehat": ShipBehaviour(
        harm_taken_weight=0.5,
        band=Band(100.0, 700.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        weapons=(WeaponUse(FIRE_AIMED), WeaponUse(FIRE_DEFEND, threat_seconds=0.2, on_crew_loss=True)),
        opponents={
            "Shofixti": {
                "specials": ("shield_glory",),
                "band": Band(504.0, 720.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
            },
            "Chmmr": {"specials": ("yehat_chmmr",), "band": None, "weapons": ()},
        },
    ),
    # Pass 1 at the default perception delay, by the harm ranking: with the
    # tongue ready, lick when aimed within its reach and lunge at an
    # opponent it can reach within a second (specials); the gun is aimed
    # late, keeping the tongue's 7 battery spare while the tongue can land.
    # Below 30% battery it keeps out of reach; otherwise a back-off band
    # closes in at full speed to the gun's 864 and steers 110 degrees off
    # only inside 100, where approach_range backed off inside 605. Against
    # the Orz it keeps that earlier band. λ is middling: it outruns only
    # the slow ships.
    "ZoqFotPik": ShipBehaviour(
        harm_taken_weight=0.5,
        specials=("zoq_fot_pik_tongue", "zoq_fot_pik_lunge"),
        late_specials=("zoq_fot_pik_gun",),
        recharge_battery=3.0,
        band=Band(100.0, 864.0, approach_speed=math.inf, run_offset=110.0, back_off=True),
        opponents={
            "Orz": {"band": Band(604.8, 864.0, approach_speed=math.inf, run_offset=110.0, back_off=True)},
        },
    ),
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
