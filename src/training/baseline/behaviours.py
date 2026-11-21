"""Per-ship parameters for the shared baseline behaviours.

The shared behaviours in ``engine`` read each ship's entry here instead of
naming ships themselves. A ship missing from ``SHIP_BEHAVIOURS`` takes the
defaults. Ship facts (weapon mechanics, hull sizes) stay in ``ship_specs``;
this table holds only choices the arena tunes.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    Recharge:
        ``recharge_battery``: below this battery, ``recharge_retreat`` keeps
        the ship out of the opponent's weapon range; None leaves it off.
    Distance band:
        ``band``: the (near, far) distance ``hold_band`` keeps the opponent
        at; None leaves it off.
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
    """

    evade: bool = True
    evade_headings: bool = True
    shoot_down: bool = True
    run_from_threats: bool = False
    planet_horizon_frames: int | None = None
    recharge_battery: float | None = None
    band: tuple[float, float] | None = None
    keep_moving: bool = True
    keep_moving_takes_aim: bool = False
    coast_above_top_speed: bool = False


DEFAULT_BEHAVIOUR = ShipBehaviour()

SHIP_BEHAVIOURS: dict[str, ShipBehaviour] = {
    "Androsynth": ShipBehaviour(shoot_down=False),
    "Arilou": ShipBehaviour(shoot_down=False),
    "Chenjesu": ShipBehaviour(keep_moving_takes_aim=True),
    "Earthling": ShipBehaviour(shoot_down=False),
    # The Kohr-Ah's shoot-downs cut crew lost by 0.9/min but crew dealt by
    # 1.3/min over 30 seeds.
    "KohrAh": ShipBehaviour(shoot_down=False),
    "Mmrnmrhm": ShipBehaviour(keep_moving_takes_aim=True),
    "Mycon": ShipBehaviour(shoot_down=False),
    "Slylandro": ShipBehaviour(keep_moving=False),
    "Thraddash": ShipBehaviour(coast_above_top_speed=True),
    "Umgah": ShipBehaviour(evade_headings=False, shoot_down=False, keep_moving=False),
}


def ship_behaviour(ship_name: str | None) -> ShipBehaviour:
    return SHIP_BEHAVIOURS.get(str(ship_name), DEFAULT_BEHAVIOUR)
