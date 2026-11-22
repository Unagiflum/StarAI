"""Follow every shot in baseline arena fights to its end.

Phase 6B measures how the perception delay changes the defence of ships
that react to shots: each damaging projectile or laser a ship fires at its
opponent ends as one of

- ``hit``: it took crew;
- ``absorbed``: a damage shield (Yehat, Utwig) blocked it;
- ``teleported``: it reached the Arilou mid-teleport, or was within
  ``TELEPORT_NEAR`` of it when a teleport began and then missed;
- ``shot_down``: the target ship or one of its objects destroyed it;
- ``missed``: anything else (expired, outrun, planet, asteroid).

Run ``python -m src.training.baseline.shot_outcomes --help``.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from src.toroidal import wrapped_distance
from src.training.baseline import arena

OUTCOMES = ("hit", "absorbed", "teleported", "shot_down", "missed")
SHOT_TYPES = ("projectile", "laser")
# A shot this close to the Arilou when it starts a teleport counts as dodged
# by the teleport if it then misses.
TELEPORT_NEAR = 400.0

DEFAULT_SHOOTERS = ("Spathi", "Arilou", "Slylandro", "Vux")
DEFAULT_TARGETS = ("Arilou", "Yehat", "Utwig")


@dataclass
class _Shot:
    shooter: str
    ability: str
    target_ship: object
    target: str
    obj: object
    outcome: str | None = None
    near_teleport: bool = False


@dataclass
class ShotTally:
    """Outcome counts for one shooter's ability against one target."""

    shooter: str
    ability: str
    target: str
    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def share(self, outcome: str) -> float:
        return self.counts[outcome] / self.total if self.total else 0.0


def _root_owner(obj):
    owner = None
    parent = getattr(obj, "parent", None)
    seen = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        owner = parent
        parent = getattr(parent, "parent", None)
    return owner


def _damaging(obj) -> bool:
    damage = getattr(obj, "current_damage", 0) or 0
    return isinstance(damage, (int, float)) and damage > 0


class ShotTracker:
    """Classify every shot either ship fires at the other.

    Pass ``tracker.on_step`` as ``arena.run_match``'s ``on_step``.
    """

    def __init__(self) -> None:
        self.finished: list[_Shot] = []
        self._live: dict[int, _Shot] = {}
        self._wrapped: set[int] = set()
        self._world = None
        self._teleporting: dict[int, bool] = {}

    def on_step(self, simulation) -> None:
        if simulation.world is not self._world:
            self._watch_world(simulation)
        ships = (simulation.player1, simulation.player2)
        for ship in ships:
            self._watch_ship(ship)
        for ship in ships:
            self._note_teleport(ship)
        present = {id(obj) for obj in simulation.world.objects}
        for key, shot in list(self._live.items()):
            obj = shot.obj
            if key in present and getattr(obj, "currently_alive", True):
                continue
            del self._live[key]
            self._finish(shot)

    def finish(self) -> None:
        """End the fight: shots still in flight count as missed."""

        for shot in self._live.values():
            self._finish(shot)
        self._live.clear()

    def tallies(self) -> dict[tuple[str, str, str], ShotTally]:
        tallies: dict[tuple[str, str, str], ShotTally] = {}
        for shot in self.finished:
            key = (shot.shooter, shot.ability, shot.target)
            tally = tallies.setdefault(key, ShotTally(*key))
            tally.counts[shot.outcome] += 1
        return tallies

    def _watch_world(self, simulation) -> None:
        world = simulation.world
        self._world = world
        tracker = self

        def watched(original):
            @functools.wraps(original)
            def add(objects):
                original(objects)
                for obj in objects if isinstance(objects, list) else (objects,):
                    tracker._register(simulation, obj)

            return add

        original_add_all = world.add_all

        @functools.wraps(original_add_all)
        def add_all(objects):
            objects = list(objects)
            original_add_all(objects)
            for obj in objects:
                tracker._register(simulation, obj)

        world.add = watched(world.add)
        world.add_all = add_all

    def _register(self, simulation, obj) -> None:
        if getattr(obj, "type", None) not in SHOT_TYPES or not _damaging(obj):
            return
        shooter = _root_owner(obj)
        ships = (simulation.player1, simulation.player2)
        if shooter not in ships:
            return
        target = ships[1] if shooter is ships[0] else ships[0]
        self._live[id(obj)] = _Shot(
            shooter=str(getattr(shooter, "name", "")),
            ability=str(getattr(obj, "name", "")),
            target_ship=target,
            target=str(getattr(target, "name", "")),
            obj=obj,
        )

    def _watch_ship(self, ship) -> None:
        if id(ship) in self._wrapped:
            return
        self._wrapped.add(id(ship))
        original = ship.take_damage
        tracker = self

        @functools.wraps(original)
        def take_damage(damage, **kwargs):
            shielded = bool(kwargs.get("shieldable", True)) and bool(
                getattr(ship, "damage_shield_is_active", lambda: False)()
            )
            teleporting = bool(getattr(ship, "teleport_frame", 0))
            applied = original(damage, **kwargs)
            shot = tracker._live.get(id(kwargs.get("source")))
            if shot is not None and shot.target_ship is ship:
                if applied > 0:
                    shot.outcome = "hit"
                elif shot.outcome != "hit":
                    if shielded:
                        shot.outcome = "absorbed"
                    elif teleporting:
                        shot.outcome = "teleported"
            return applied

        ship.take_damage = take_damage

    def _note_teleport(self, ship) -> None:
        teleporting = bool(getattr(ship, "teleport_frame", 0))
        started = teleporting and not self._teleporting.get(id(ship), False)
        self._teleporting[id(ship)] = teleporting
        if not started:
            return
        origin = getattr(ship, "frozen_camera_position", None) or ship.position
        for shot in self._live.values():
            if shot.target_ship is ship and (
                wrapped_distance(origin, shot.obj.position) <= TELEPORT_NEAR
            ):
                shot.near_teleport = True

    def _finish(self, shot: _Shot) -> None:
        if shot.outcome is None:
            source = getattr(shot.obj, "_training_destroying_source", None)
            if source is not None and (
                source is shot.target_ship or _root_owner(source) is shot.target_ship
            ):
                shot.outcome = "shot_down"
            elif shot.near_teleport:
                shot.outcome = "teleported"
            else:
                shot.outcome = "missed"
        shot.obj = None
        shot.target_ship = None
        self.finished.append(shot)


def measure_match(match: arena.ArenaMatch, frames: int) -> list[tuple[str, str, str, str]]:
    """Fly one arena fight; return (shooter, ability, target, outcome) per shot."""

    tracker = ShotTracker()
    arena.run_match(match, frames=frames, on_step=tracker.on_step)
    tracker.finish()
    return [(shot.shooter, shot.ability, shot.target, shot.outcome) for shot in tracker.finished]


def _measure(arguments):
    match, frames = arguments
    return measure_match(match, frames)


def measure(
    matches: Sequence[arena.ArenaMatch],
    *,
    frames: int = arena.DEFAULT_FRAMES,
    workers: int = arena.DEFAULT_WORKERS,
    ship_directions: int = arena.DEFAULT_SHIP_DIRECTIONS,
    perception_delay: int = arena.DEFAULT_PERCEPTION_DELAY,
) -> dict[tuple[str, str, str], ShotTally]:
    work = [(match, int(frames)) for match in matches]
    if int(workers) <= 1:
        arena.apply_arena_settings(ship_directions, perception_delay)
        results = [_measure(item) for item in work]
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=arena._worker_init,
            initargs=(int(ship_directions), int(perception_delay)),
        ) as pool:
            results = list(pool.map(_measure, work, chunksize=1))
    tallies: dict[tuple[str, str, str], ShotTally] = {}
    for shots in results:
        for shooter, ability, target, outcome in shots:
            key = (shooter, ability, target)
            tallies.setdefault(key, ShotTally(*key)).counts[outcome] += 1
    return tallies


def format_tallies(tallies: dict[tuple[str, str, str], ShotTally]) -> str:
    header = f"{'shooter':10s} {'shot':13s} {'target':10s} {'shots':>6s} " + " ".join(
        f"{outcome:>10s}" for outcome in OUTCOMES
    )
    lines = [header]
    for key in sorted(tallies):
        tally = tallies[key]
        lines.append(
            f"{tally.shooter:10s} {tally.ability:13s} {tally.target:10s} {tally.total:6d} "
            + " ".join(f"{tally.share(outcome):10.1%}" for outcome in OUTCOMES)
        )
    return "\n".join(lines)


def _names(text: str, default: Sequence[str]) -> list[str]:
    return arena._ship_list(text, default)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.baseline.shot_outcomes",
        description="Follow every shot in baseline fights to its end.",
    )
    parser.add_argument(
        "--shooters", help=f"Comma-separated (default: {','.join(DEFAULT_SHOOTERS)})."
    )
    parser.add_argument(
        "--targets", help=f"Comma-separated (default: {','.join(DEFAULT_TARGETS)})."
    )
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per pairing.")
    parser.add_argument("--first-seed", type=int, default=0, help="First seed.")
    parser.add_argument("--frames", type=int, default=arena.DEFAULT_FRAMES)
    parser.add_argument(
        "--cells", help="Starting-condition cells: all, or some of SS,SW,WS,WW."
    )
    parser.add_argument("--workers", type=int, default=arena.DEFAULT_WORKERS)
    parser.add_argument(
        "--perception-delay",
        type=arena._perception_delay,
        default=arena.DEFAULT_PERCEPTION_DELAY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    shooters = _names(arguments.shooters, DEFAULT_SHOOTERS)
    targets = _names(arguments.targets, DEFAULT_TARGETS)
    seeds = range(arguments.first_seed, arguments.first_seed + arguments.seeds)
    cells = arena._cell_list(arguments.cells)
    matches = [
        match
        for shooter in shooters
        for match in arena.build_matches(
            [shooter], [target for target in targets if target != shooter], seeds, cells=cells
        )
    ]
    started = time.perf_counter()
    tallies = measure(
        matches,
        frames=arguments.frames,
        workers=arguments.workers,
        perception_delay=arguments.perception_delay,
    )
    # Only the evaluated ships' shots at the targets.
    tallies = {
        key: tally
        for key, tally in tallies.items()
        if tally.shooter in shooters and tally.target in targets
    }
    print(f"Perception delay: {arguments.perception_delay} frames")
    print(format_tallies(tallies))
    print(f"{len(matches)} fights in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
