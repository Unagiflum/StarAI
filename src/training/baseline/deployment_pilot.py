"""Paired fleet-deployment pilot for the screened projectile matchups.

Run from the repository root with python -m src.training.baseline.deployment_pilot.
Both scores come from the same battle. The evaluated ships' weapons and
marines are removed with their owner, so the death frame closes the final
exchange. The training arena's ordinary respawns cannot model a Pkunk rebirth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import math
import statistics
from types import SimpleNamespace

import src.const as const
from src.Battle.collision_geometry import distance_from_segment_to_point, radius
from src.training.baseline import arena
from src.training.baseline.behaviours import Band, ship_behaviour
from src.training.baseline.shot_outcomes import ShotTracker
from src.training.harm import HARM_DAMAGE_ENEMY, HARM_KILL_ENEMY, HarmTracker


SEED_BLOCKS = {"selection": range(20), "confirmation": range(1000, 1020)}
PRESSURE_LOOKAHEAD = const.FPS / 2
PACE_LIMIT = 10 * const.FPS
PILOT_PAIRS = {("Orz", "Chmmr"), ("Spathi", "Utwig"), ("Utwig", "Spathi"), ("Spathi", "Yehat")}


def pilot_settings():
    orz = ship_behaviour("Orz", "Chmmr")
    spathi = ship_behaviour("Spathi", "Utwig")
    return {
        ("Orz", "Chmmr"): {
            "default": None,
            "base_band": replace(ship_behaviour("Orz"), opponents={}),
            "far_band": replace(orz, opponents={}, satellite_band=None, band=Band(1550, 1800)),
            "dash": replace(orz, opponents={}, specials=("orz_chmmr_dash", *orz.specials)),
        },
        ("Spathi", "Utwig"): {
            "default": None,
            "base_band": replace(ship_behaviour("Spathi"), opponents={}),
            "old_standoff": replace(spathi, opponents={}, specials=(), band=Band(900, 1300, approach_speed=60)),
            "approach": None,
        },
    }


def on_intercept(shot, target):
    """Could these moving collision circles touch within the remaining flight?"""
    frames = min(PRESSURE_LOOKAHEAD, max(0.0, shot.expiration_timer))
    x, y = shot.position
    vx, vy = shot.velocity
    tx, ty = target.velocity
    end = (x + (vx - tx) * frames, y + (vy - ty) * frames)
    separation = distance_from_segment_to_point(shot.position, end, target.position)
    return separation <= radius(shot) + radius(target)


def threatens(world, ship, enemy):
    shots = [
        obj for obj in world.objects
        if getattr(obj, "parent", None) is ship
        and getattr(obj, "currently_alive", False)
        and (getattr(obj, "type", None) == "projectile" or obj.name == "OrzA3")
    ]
    if not shots:
        return False
    targets = [enemy]
    if enemy.name == "Chmmr":
        targets.extend(
            obj for obj in world.objects
            if getattr(obj, "name", None) == "ChmmrSatellite"
            and obj.parent is enemy and obj.current_hp > 0
        )
    for shot in shots:
        if shot.name == "OrzA3":
            if shot.is_boarded:
                if shot.boarded_ship is enemy:
                    return True
            elif not shot.is_returning and on_intercept(shot, enemy):
                return True
        elif shot.current_damage > 0 and any(on_intercept(shot, target) for target in targets):
            return True
    return False


@dataclass
class Pace:
    active: int = 0
    pressure: int = 0
    quiet: int = 0
    longest: int = 0

    def step(self, can_fight, under_pressure):
        if can_fight:
            self.active += 1
            if under_pressure:
                self.pressure += 1
                self.quiet = 0
            else:
                self.quiet += 1
                self.longest = max(self.longest, self.quiet)


class Deployment:
    """Observe the initial ship, including the step that replaces it."""

    def __init__(self, facing_shield):
        self.harm = HarmTracker()
        self.pace = Pace()
        self.shots = ShotTracker() if facing_shield else None
        self.before = None
        self.event_offset = 0
        self.can_fight = False
        self.incoming = False
        self.lost = False
        self.frames = 0
        self.kills = 0

    def on_step(self, simulation):
        if self.lost:
            return
        ledger = simulation.training_event_ledger
        absorbed = False
        if self.shots is not None:
            finished = len(self.shots.finished)
            self.shots.on_step(simulation)
            if self.before is not None:
                absorbed = any(
                    shot.shooter == self.before.self_ship.name and shot.outcome == "absorbed"
                    for shot in self.shots.finished[finished:]
                )
        if self.before is not None:
            events = ledger.events[self.event_offset:]
            effects = self.harm.contributions(self.before, SimpleNamespace(events=events))
            dealt = any(effect.component in (HARM_DAMAGE_ENEMY, HARM_KILL_ENEMY) for effect in effects)
            self.pace.step(self.can_fight, self.incoming or absorbed or dealt)
            self.frames += 1
            if 2 in simulation.training_episode_deaths and 1 in simulation.training_episode_kills:
                self.kills += 1
            if simulation.player1 is not self.before.self_ship:
                self.lost = True
                return

        ship, enemy = simulation.player1, simulation.player2
        self.event_offset = len(ledger.events)
        self.before = SimpleNamespace(
            self_ship=ship, enemy_ship=enemy,
            self_crew=ship.current_hp, enemy_crew=enemy.current_hp,
            self_limpets=ship.limpets_attached, enemy_limpets=enemy.limpets_attached,
        )
        self.can_fight = ship.currently_alive and enemy.currently_alive and simulation.entry is None
        self.incoming = self.can_fight and threatens(simulation.world, ship, enemy)


@dataclass(frozen=True)
class PilotResult:
    ship: str
    opponent: str
    variant: str
    seed: int
    cell: str
    legacy_harm: float
    legacy_score: float
    deployment_harm: float
    deployment_kills: int
    frames: int
    active_frames: int
    pressure_frames: int
    longest_gap: int
    survived_cap: bool


def measure(match):
    if (match.ship, match.opponent) not in PILOT_PAIRS:
        raise ValueError(f"Deployment and pressure attribution have not been verified for {match.ship} vs {match.opponent}.")
    deployment = Deployment(facing_shield=match.opponent in ("Utwig", "Yehat"))
    legacy = arena.run_match(match, on_step=deployment.on_step)
    totals = deployment.harm.totals
    return PilotResult(
        ship=match.ship, opponent=match.opponent, variant=match.variant,
        seed=match.seed, cell=legacy.cell,
        legacy_harm=legacy.harm_dealt, legacy_score=legacy.score,
        deployment_harm=totals[HARM_DAMAGE_ENEMY] + totals[HARM_KILL_ENEMY],
        deployment_kills=deployment.kills, frames=deployment.frames,
        active_frames=deployment.pace.active, pressure_frames=deployment.pace.pressure,
        longest_gap=deployment.pace.longest, survived_cap=not deployment.lost,
    )


def p90_gap(rows):
    gaps = sorted(r.longest_gap for r in rows)
    return gaps[math.ceil(0.9 * len(gaps)) - 1]


def fleet_difference(base, group):
    if [(r.seed, r.cell) for r in base] != [(r.seed, r.cell) for r in group]:
        raise ValueError("A paired comparison requires the same ordered seeds and cells.")
    return arena.paired_difference([r.deployment_harm for r in base], [r.deployment_harm for r in group])


def eligible_challenger(base, group):
    delta = fleet_difference(base, group)
    required_lead = 1 if p90_gap(base) <= PACE_LIMIT else 0
    return p90_gap(group) <= PACE_LIMIT and delta.other_mean > 0 and delta.lead >= required_lead


def report(rows):
    selected = {}
    for block, seeds in SEED_BLOCKS.items():
        for (ship, opponent), settings in pilot_settings().items():
            print(f"\n{block}  {ship} vs {opponent}")
            print("variant       n  old harm  old score  fleet harm    paired fleet difference [95% CI]"
                  "   p90 gap   no threat   at cap")
            by_name = {
                name: sorted(
                    (r for r in rows if r.ship == ship and r.variant == name and r.seed in seeds),
                    key=lambda r: (r.seed, r.cell),
                )
                for name in settings
            }
            base = by_name["default"]
            for name, group in by_name.items():
                delta = fleet_difference(base, group)
                print(
                    f"{name:12s} {len(group):3d}  {statistics.fmean(r.legacy_harm for r in group):8.3f}"
                    f"  {statistics.fmean(r.legacy_score for r in group):9.3f}  {delta.other_mean:10.3f}"
                    f"    {delta.text():30s}  {p90_gap(group) / const.FPS:6.2f}s"
                    f"     {statistics.fmean(r.pressure_frames == 0 for r in group):6.1%}"
                    f"   {statistics.fmean(r.survived_cap for r in group):6.1%}"
                )
            if block == "selection":
                challengers = [
                    name for name, group in by_name.items()
                    if name != "default" and eligible_challenger(base, group)
                ]
                selected[ship] = max(
                    challengers, key=lambda name: statistics.fmean(r.deployment_harm for r in by_name[name]),
                    default="default",
                )
                print(f"Selection: {selected[ship]}")
            else:
                name = selected[ship]
                winner = name if name != "default" and eligible_challenger(base, by_name[name]) else "default"
                note = " (pace target missed)" if p90_gap(by_name[winner]) > PACE_LIMIT else ""
                print(f"Confirmation: {winner}{note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=arena.DEFAULT_WORKERS)
    parser.add_argument("--csv", help="Write one row per flown setting, seed and cell.")
    args = parser.parse_args()
    candidates = pilot_settings()
    matches = [
        arena.ArenaMatch(ship, opponent, seed, name, mine, theirs, entry)
        for seeds in SEED_BLOCKS.values()
        for seed in seeds
        for (ship, opponent), settings in candidates.items()
        for name, entry in settings.items()
        for mine, theirs in arena.CELLS
    ]
    settings = (arena.DEFAULT_SHIP_DIRECTIONS, arena.DEFAULT_PERCEPTION_DELAY)
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=arena._worker_init, initargs=settings,
    ) as pool:
        for row in pool.map(measure, matches, chunksize=1):
            rows.append(row)
            if len(rows) % 80 == 0:
                print(f"{len(rows)}/{len(matches)} fights", flush=True)
    report(rows)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()
