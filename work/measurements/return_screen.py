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
from src.training.baseline.behaviours import BAND_INSIDE_FACE, SHIP_BEHAVIOURS, Band, ship_behaviour
from src.training.baseline.observation_view import wrap_degrees
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
    ctx.steer(aim, thrust=abs(wrap_degrees(aim)) <= 60)


def turn_out(ctx, near, angle):
    if ctx.enemy is None or ctx.plan.steering_claimed:
        return
    if ctx.keep_standoff("spathi_turn_out", near, math.inf, release=1080.0, run_offset=angle):
        return
    aimed_return(ctx, 1080.0)


def delayed_return(ctx, seconds):
    if ctx.enemy is None:
        return
    key = "spathi_out_of_range"
    if ctx.in_range(1):
        ctx.memory.values.pop(key, None)
        return
    elapsed = ctx.memory.values.get(key, 0) + 1
    ctx.memory.values[key] = elapsed
    if elapsed >= seconds * const.FPS:
        aimed_return(ctx, ctx.weapon(1).range)


def cross_the_nose(ctx, distance, angle):
    if ctx.enemy is None or ctx.plan.steering_claimed:
        return
    facing = ctx.enemy_block.opponent_bearing
    if ctx.distance <= distance and facing is not None and abs(facing) <= 45:
        heading = ctx.side("spathi_cross_nose", ctx.enemy.bearing, angle)
        ctx.steer(heading, thrust=abs(wrap_degrees(heading)) <= 60)
    else:
        aimed_return(ctx, ctx.weapon(1).range)


def init_worker(original_spathi_opponent=False):
    arena._worker_init(arena.DEFAULT_SHIP_DIRECTIONS, arena.DEFAULT_PERCEPTION_DELAY)
    if original_spathi_opponent:
        spathi = ship_behaviour("Spathi")
        opponents = {**spathi.opponents, "Utwig": {"band": None}}
        SHIP_BEHAVIOURS["Spathi"] = replace(spathi, opponents=opponents)
    for distance in (432, 648, 864, 1080, 1350, 1620, 2160, 2700):
        SPECIAL_RULES[f"aimed_{distance}"] = partial(aimed_return, distance=distance)
    SPECIAL_RULES["flank_648"] = partial(aimed_return, distance=648, flank=45)
    for near in (432, 648, 864):
        for angle in (120, 160):
            SPECIAL_RULES[f"turn_{near}_{angle}"] = partial(turn_out, near=near, angle=angle)
    for seconds in (1, 2, 3, 4):
        SPECIAL_RULES[f"delay_{seconds}"] = partial(delayed_return, seconds=seconds)
    for distance in (864, 1350):
        for angle in (90, 135):
            SPECIAL_RULES[f"cross_{distance}_{angle}"] = partial(cross_the_nose, distance=distance, angle=angle)


def settings(ship, opponent):
    current = replace(ship_behaviour(ship, opponent), opponents={})
    group = {"default": None}
    if (ship, opponent) == ("Spathi", "Yehat"):
        group["missile_range"] = replace(current, band=None, specials=("aimed_864",))
    elif (ship, opponent) == ("Utwig", "Spathi"):
        band = Band(756.0, 1080.0, approach_speed=60.0)
        group.update({
            "earlier": None,
            "former_band": replace(current, band=band),
            "base_band": replace(current, band=ship_behaviour("Utwig").band),
            "no_band": replace(current, band=None),
            "near": replace(current, band=Band(529.2, 756.0, approach_speed=60.0)),
            "far": replace(current, band=Band(982.8, 1404.0, approach_speed=60.0)),
            "face": replace(current, band=replace(band, inside=BAND_INSIDE_FACE)),
        })
    elif (ship, opponent) == ("Spathi", "Utwig"):
        current = replace(current, specials=())
        group["no_band"] = replace(current, band=None)
        group["approach"] = None
        for near, far in ((0, 1080), (0, 864), (0, 648), (0, 432), (320, 648)):
            name = f"band_{near}_{far}"
            group[name] = replace(
                current, recharge_battery=3, band=Band(near, far, approach_speed=math.inf,
                                                      run_offset=110, back_off=True),
            )
        returns = [f"aimed_{distance}" for distance in (432, 648, 864, 1080, 1350, 1620, 2160, 2700)]
        turns = [f"turn_{near}_{angle}" for near in (432, 648, 864) for angle in (120, 160)]
        delays = [f"delay_{seconds}" for seconds in (1, 2, 3, 4)]
        crossings = [f"cross_{distance}_{angle}" for distance in (864, 1350) for angle in (90, 135)]
        for name in (*returns, *turns, *delays, *crossings, "flank_648"):
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
    p.add_argument("--original-spathi-opponent", action="store_true",
                   help="Fly Utwig against Spathi's pre-screen no-band rule.")
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    if args.original_spathi_opponent and (args.ship, args.opponent) != ("Utwig", "Spathi"):
        p.error("--original-spathi-opponent requires Utwig Spathi")
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
        max_workers=args.workers, initializer=init_worker, initargs=(args.original_spathi_opponent,),
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
