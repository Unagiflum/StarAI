"""Local paired follow-up screens. Launch as python -m work.measurements.return_screen."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
from functools import partial
import math
from pathlib import Path

from src import const
from src.training.baseline import arena, deployment_pilot as pilot
from src.training.baseline.behaviours import BAND_INSIDE_FACE, Band, ship_behaviour
from src.training.baseline.strategies import SPECIAL_RULES


def aimed_return(ctx, distance, flank=0):
    """Return on gun aim when a gun can fire, leaving a live dodge first claim."""
    if ctx.me.battery < ctx.spec.a1_cost or ctx.distance <= distance or ctx.plan.steering_claimed:
        return
    enemy = ctx.enemy
    if enemy is None:
        return
    aim = ctx.aim_bearing(1)
    if flank and ctx.distance < 1600:
        facing = ctx.enemy_block.opponent_bearing
        if facing is not None and abs(facing) < 40:
            aim = enemy.bearing + (-flank if facing > 0 else flank)
    ctx.steer(aim, thrust=abs((aim + 180) % 360 - 180) <= 60)


def init_worker():
    arena._worker_init(arena.DEFAULT_SHIP_DIRECTIONS, arena.DEFAULT_PERCEPTION_DELAY)
    for distance in (432, 648, 864):
        SPECIAL_RULES[f"aimed_{distance}"] = partial(aimed_return, distance=distance)
    SPECIAL_RULES["flank_648"] = partial(aimed_return, distance=648, flank=45)


def settings(ship, opponent):
    current = replace(ship_behaviour(ship, opponent), opponents={})
    group = {"default": None}
    if (ship, opponent) == ("Spathi", "Yehat"):
        group["missile_range"] = replace(current, band=None, specials=("spathi_return_to_missile_range",))
    elif (ship, opponent) == ("Utwig", "Spathi"):
        band = Band(756.0, 1080.0, approach_speed=60.0)
        group.update({
            "earlier": None,
            "base_band": replace(current, band=ship_behaviour("Utwig").band),
            "no_band": replace(current, band=None),
            "near": replace(current, band=Band(529.2, 756.0, approach_speed=60.0)),
            "far": replace(current, band=Band(982.8, 1404.0, approach_speed=60.0)),
            "face": replace(current, band=replace(band, inside=BAND_INSIDE_FACE)),
        })
    elif (ship, opponent) == ("Spathi", "Utwig"):
        current = replace(current, specials=())
        group["approach"] = None
        for near, far in ((0, 1080), (0, 864), (0, 648), (0, 432), (320, 648)):
            name = f"band_{near}_{far}"
            group[name] = replace(
                current, recharge_battery=3, band=Band(near, far, approach_speed=math.inf,
                                                      run_offset=110, back_off=True),
            )
        for name in ("aimed_432", "aimed_648", "aimed_864", "flank_648"):
            group[name] = replace(current, band=None, specials=(name,))
    return group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ship")
    p.add_argument("opponent")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--variants", nargs="+")
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    candidates = settings(args.ship, args.opponent)
    names = args.variants or list(candidates)
    matches = [
        arena.ArenaMatch(args.ship, args.opponent, seed, name, mine, theirs, candidates[name])
        for seed in range(args.first, args.first + args.seeds)
        for name in names
        for mine, theirs in arena.CELLS
    ]
    rows = {name: [] for name in names}
    with args.csv.open("x", newline="", encoding="utf-8") as out, ProcessPoolExecutor(
        max_workers=args.workers, initializer=init_worker,
    ) as pool:
        writer = csv.DictWriter(out, fieldnames=pilot.PilotResult.__dataclass_fields__)
        writer.writeheader()
        for n, result in enumerate(pool.map(pilot.measure, matches, chunksize=1), 1):
            rows[result.variant].append(result)
            writer.writerow(asdict(result))
            if n % 160 == 0 or n == len(matches):
                print(f"{n}/{len(matches)} fights", flush=True)
                out.flush()
    base = rows["default"]
    for name, group in rows.items():
        diff = pilot.fleet_difference(base, group)
        gap = pilot.p90_gap(group) / const.FPS
        slow = sum(r.longest_gap > pilot.PACE_LIMIT for r in group)
        alive = sum(r.survived_cap for r in group)
        print(f"{name:16}   H {diff.other_mean:.4f}  delta {diff.text():30}   "
              f"P90 {gap:6.2f}   over10 {slow:3}/{len(group)}   at cap {alive:3}   "
              f"kills {sum(r.deployment_kills for r in group)}")


if __name__ == "__main__":
    main()
