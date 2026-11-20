"""Baseline-versus-baseline arena for comparing ship strategies.

Every fight puts two baseline-controlled ships at full crew in a fresh battle
and runs it until one side is destroyed or time runs out. Strategy variants
(see ``strategies.STRATEGY_VARIANTS``) fly against the same opponents with the
same seeds as the default rules, so their results compare pair by pair.
Opponents always fly their default rules.

Run from the repository root, for example::

    python -m src.training.baseline.arena --ships ZoqFotPik,Pkunk \\
        --opponents Spathi,Earthling,KzerZa --seeds 6 --variants all --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Iterable, Sequence

import src.const as const
from src.training.baseline.strategies import (
    DEFAULT_VARIANT,
    STRATEGY_VARIANTS,
    strategy_for,
)
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER

DEFAULT_OPPONENTS = ("Spathi", "Earthling", "KzerZa")
DEFAULT_SECONDS = 100.0
DEFAULT_SHIP_DIRECTIONS = 64

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_DRAW = "draw"
# Both ships destroyed by the evaluated ship's own self-destruct: a Shofixti
# glory blast that took the opponent with it.
OUTCOME_TRADE = "trade"
OUTCOME_TIMEOUT = "timeout"
_BOTH_DESTROYED = (OUTCOME_DRAW, OUTCOME_TRADE)


@dataclass(frozen=True, slots=True)
class ArenaMatch:
    """One fight to run: ``ship`` flying ``variant`` against ``opponent``."""

    ship: str
    opponent: str
    seed: int
    variant: str = DEFAULT_VARIANT


@dataclass(frozen=True, slots=True)
class ArenaResult:
    ship: str
    variant: str
    opponent: str
    seed: int
    outcome: str
    seconds: float
    crew_start: float
    crew_end: float
    opponent_crew_start: float
    opponent_crew_end: float
    # Satellite HP the opponent lost: set only when it fights with satellites
    # (the Chmmr), None otherwise.
    opponent_satellite_hp_lost: float | None = None

    @property
    def damage_dealt(self) -> float:
        """Share of the opponent's crew removed (1.0 for a kill)."""

        if self.outcome == OUTCOME_WIN or self.outcome in _BOTH_DESTROYED:
            return 1.0
        return _lost_share(self.opponent_crew_start, self.opponent_crew_end)

    @property
    def damage_taken(self) -> float:
        if self.outcome == OUTCOME_LOSS or self.outcome in _BOTH_DESTROYED:
            return 1.0
        return _lost_share(self.crew_start, self.crew_end)


def _lost_share(start: float, end: float) -> float:
    if start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (start - end) / start))


def apply_arena_settings(ship_directions: int = DEFAULT_SHIP_DIRECTIONS) -> None:
    """Apply the gameplay settings fights run under (once per process)."""

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    const.apply_game_settings(
        SimpleNamespace(
            asteroid_count=const.DEFAULT_GAMEPLAY["asteroid_count"],
            ship_directions=int(ship_directions),
            repeat_key_delay=const.DEFAULT_GAMEPLAY["repeat_key_delay"],
        )
    )


def _match_rng(match: ArenaMatch) -> random.Random:
    # The variant is left out so every variant meets the same battles.
    return random.Random(f"{match.ship}|{match.opponent}|{match.seed}")


_RESOURCES = None


def _arena_resources():
    """One headless asset cache per process, as training workers use.

    Without it ships fall back to placeholder sprites and collision masks,
    which changes every hull and weapon shape.
    """

    global _RESOURCES
    if _RESOURCES is None:
        from src.resources import HeadlessAssetManager

        _RESOURCES = HeadlessAssetManager()
    return _RESOURCES


def _destroyed(ship) -> bool:
    return not getattr(ship, "currently_alive", True) or ship.current_hp <= 0


def _has_satellites(ship_name: str) -> bool:
    from src.Objects.Ships.catalog import SHIP_DEFINITIONS

    return bool(getattr(SHIP_DEFINITIONS[ship_name], "satellite_count", None))


def _satellite_hp(simulation, owner) -> float:
    """Total HP of ``owner``'s live satellites."""

    return float(
        sum(
            obj.current_hp
            for obj in simulation.world.objects
            if getattr(obj, "name", None) == "ChmmrSatellite"
            and getattr(obj, "parent", None) is owner
            and getattr(obj, "currently_alive", False)
            and obj.current_hp > 0
        )
    )


def run_match(match: ArenaMatch, *, seconds: float = DEFAULT_SECONDS) -> ArenaResult:
    """Fly one fight between two baseline-controlled ships."""

    from src.Battle.battle import BattleSimulation
    from src.Battle.computer_control import guard_computer_controls
    from src.Objects.Ships.registry import create_ship
    from src.audio import NullAudioService
    from src.training.baseline.controller import BaselineController
    from src.training.contracts import action_for_index
    from src.training.observation import encode_observation
    from src.training.orchestration import _fully_arm_training_shofixti

    audio = NullAudioService()
    resources = _arena_resources()
    simulation = BattleSimulation(
        None,
        create_ship(match.ship, 1, resources=resources, audio_service=audio),
        create_ship(match.opponent, 2, resources=resources, audio_service=audio),
        audio_service=audio,
        rng=_match_rng(match),
        include_stars=False,
        resources=resources,
    )
    _fully_arm_training_shofixti(simulation.player1)
    _fully_arm_training_shofixti(simulation.player2)
    controllers = {
        1: BaselineController(match.ship, match.variant),
        2: BaselineController(match.opponent),
    }
    crew_start = float(simulation.player1.current_hp)
    opponent_crew_start = float(simulation.player2.current_hp)
    frame_limit = max(1, int(round(float(seconds) * const.FPS)))
    outcome = OUTCOME_TIMEOUT
    frames = frame_limit
    # Satellites appear on the first update and never regain HP, so the most
    # seen less the least seen is the HP the evaluated ship removed.
    track_satellites = _has_satellites(match.opponent)
    satellite_most = 0.0
    satellite_least = None
    for frame in range(frame_limit):
        ships = {1: simulation.player1, 2: simulation.player2}
        actions = {}
        for player, ship in ships.items():
            enemy = ships[3 - player]
            observation = encode_observation(
                ship,
                enemy,
                frame_id=simulation.frame_id,
                game_objects=simulation.world,
            )
            action = action_for_index(controllers[player].decide(observation))
            actions[player] = guard_computer_controls(action, ship, enemy)
        simulation.step(actions=actions)
        if track_satellites:
            satellite_hp = _satellite_hp(simulation, simulation.player2)
            satellite_most = max(satellite_most, satellite_hp)
            if satellite_least is None or satellite_hp < satellite_least:
                satellite_least = satellite_hp
        winner = simulation.winner()
        if winner is simulation.player1:
            outcome, frames = OUTCOME_WIN, frame + 1
            break
        if winner is simulation.player2:
            outcome, frames = OUTCOME_LOSS, frame + 1
            break
        aftermath = getattr(simulation, "aftermath", None)
        rebirth_pending = bool(aftermath is not None and aftermath.pending_rebirths)
        if (
            not rebirth_pending
            and _destroyed(simulation.player1)
            and _destroyed(simulation.player2)
        ):
            traded = bool(getattr(simulation.player1, "shofixti_self_destruct", False))
            outcome = OUTCOME_TRADE if traded else OUTCOME_DRAW
            frames = frame + 1
            break
    return ArenaResult(
        ship=match.ship,
        variant=match.variant,
        opponent=match.opponent,
        seed=match.seed,
        outcome=outcome,
        seconds=frames / float(const.FPS),
        crew_start=crew_start,
        crew_end=max(0.0, float(simulation.player1.current_hp)),
        opponent_crew_start=opponent_crew_start,
        opponent_crew_end=max(0.0, float(simulation.player2.current_hp)),
        opponent_satellite_hp_lost=(
            satellite_most - (satellite_least or 0.0) if track_satellites else None
        ),
    )


def build_matches(
    ships: Sequence[str],
    opponents: Sequence[str],
    seeds: int,
    variants: dict[str, Sequence[str]] | None = None,
) -> list[ArenaMatch]:
    """Every ship and variant against every opponent for ``seeds`` seeds."""

    variants = variants or {}
    matches = []
    for ship in ships:
        ship_variants = [DEFAULT_VARIANT, *variants.get(ship, ())]
        for variant in dict.fromkeys(ship_variants):
            strategy_for(ship, variant)  # Fail fast on unknown names.
            for opponent in opponents:
                for seed in range(int(seeds)):
                    matches.append(ArenaMatch(ship, opponent, seed, variant))
    return matches


def _worker_init(ship_directions: int) -> None:
    apply_arena_settings(ship_directions)


def _run_one(arguments: tuple[ArenaMatch, float]) -> ArenaResult:
    match, seconds = arguments
    return run_match(match, seconds=seconds)


def run_arena(
    matches: Sequence[ArenaMatch],
    *,
    seconds: float = DEFAULT_SECONDS,
    workers: int = 1,
    ship_directions: int = DEFAULT_SHIP_DIRECTIONS,
) -> Iterable[ArenaResult]:
    """Yield results in ``matches`` order, using worker processes if asked."""

    work = [(match, float(seconds)) for match in matches]
    if int(workers) <= 1:
        apply_arena_settings(ship_directions)
        for item in work:
            yield _run_one(item)
        return
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(int(ship_directions),),
    ) as pool:
        yield from pool.map(_run_one, work, chunksize=1)


@dataclass(slots=True)
class ArenaTally:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    trades: int = 0
    timeouts: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    win_seconds: float = 0.0
    seconds: float = 0.0
    satellite_fights: int = 0
    satellite_hp_lost: float = 0.0

    def add(self, result: ArenaResult) -> None:
        self.fights += 1
        if result.outcome == OUTCOME_WIN:
            self.wins += 1
            self.win_seconds += result.seconds
        elif result.outcome == OUTCOME_LOSS:
            self.losses += 1
        elif result.outcome == OUTCOME_DRAW:
            self.draws += 1
        elif result.outcome == OUTCOME_TRADE:
            self.trades += 1
        else:
            self.timeouts += 1
        self.damage_dealt += result.damage_dealt
        self.damage_taken += result.damage_taken
        self.seconds += result.seconds
        if result.opponent_satellite_hp_lost is not None:
            self.satellite_fights += 1
            self.satellite_hp_lost += result.opponent_satellite_hp_lost

    @property
    def mean_dealt(self) -> float:
        return self.damage_dealt / self.fights if self.fights else 0.0

    @property
    def mean_taken(self) -> float:
        return self.damage_taken / self.fights if self.fights else 0.0

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.fights if self.fights else 0.0

    @property
    def mean_satellite_hp_lost(self) -> float | None:
        if not self.satellite_fights:
            return None
        return self.satellite_hp_lost / self.satellite_fights

    @property
    def mean_win_seconds(self) -> float | None:
        return self.win_seconds / self.wins if self.wins else None

    def cell(self) -> str:
        text = (
            f"{self.wins}W {self.losses}L {self.draws}D {self.trades}X {self.timeouts}T "
            f"dealt {self.mean_dealt:4.0%} taken {self.mean_taken:4.0%} "
            f"fight {self.mean_seconds:3.0f}s"
        )
        satellite_hp = self.mean_satellite_hp_lost
        if satellite_hp is not None:
            text += f" sat HP {satellite_hp:4.1f}"
        return text


def tally(
    results: Iterable[ArenaResult],
) -> dict[tuple[str, str], dict[str, ArenaTally]]:
    """Group results by (ship, variant), then by opponent and ``"all"``."""

    table: dict[tuple[str, str], dict[str, ArenaTally]] = {}
    for result in results:
        rows = table.setdefault((result.ship, result.variant), {})
        rows.setdefault(result.opponent, ArenaTally()).add(result)
        rows.setdefault("all", ArenaTally()).add(result)
    return table


def format_report(
    table: dict[tuple[str, str], dict[str, ArenaTally]],
    opponents: Sequence[str],
) -> str:
    lines = []
    for (ship, variant), rows in table.items():
        overall = rows["all"]
        win_time = overall.mean_win_seconds
        lines.append(
            f"{ship} [{variant}]  {overall.cell()}"
            + (f"  mean win {win_time:.0f}s" if win_time is not None else "")
        )
        for opponent in opponents:
            if opponent in rows:
                lines.append(f"    vs {opponent:11s} {rows[opponent].cell()}")
    return "\n".join(lines)


def _ship_list(text: str | None, default: Sequence[str]) -> list[str]:
    if not text:
        return list(default)
    if text.strip().lower() == "all":
        return list(SHIP_TYPE_CATALOG_ORDER)
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in SHIP_TYPE_CATALOG_ORDER]
    if unknown:
        raise SystemExit(f"Unknown ships: {', '.join(unknown)}")
    return names


def _variant_map(entries: Sequence[str], ships: Sequence[str]) -> dict[str, list[str]]:
    """Parse ``all`` or ``SHIP=VARIANT`` entries into variants per ship."""

    variants: dict[str, list[str]] = {}
    for entry in entries:
        if entry.strip().lower() == "all":
            for ship in ships:
                variants.setdefault(ship, []).extend(STRATEGY_VARIANTS.get(ship, {}))
            continue
        ship, separator, variant = entry.partition("=")
        if not separator or not variant:
            raise SystemExit(f"Variant must be SHIP=VARIANT or all, not {entry!r}")
        variants.setdefault(ship.strip(), []).append(variant.strip())
    return variants


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.baseline.arena",
        description="Fight baseline strategies against each other.",
    )
    parser.add_argument(
        "--ships",
        help="Comma-separated ships to evaluate, or 'all' (default: all).",
    )
    parser.add_argument(
        "--opponents",
        help=(
            "Comma-separated opponents, or 'all' "
            f"(default: {','.join(DEFAULT_OPPONENTS)})."
        ),
    )
    parser.add_argument("--seeds", type=int, default=4, help="Fights per pairing.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Time limit per fight in game seconds.",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        default=[],
        help=(
            "Also fly SHIP=VARIANT next to the default rules; repeatable. "
            "'all' adds every variant of the chosen ships."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker processes.")
    parser.add_argument(
        "--directions",
        type=int,
        default=DEFAULT_SHIP_DIRECTIONS,
        choices=(16, 32, 64),
        help="Ship heading directions.",
    )
    parser.add_argument("--json", help="Also write every fight result to this file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    ships = _ship_list(arguments.ships, SHIP_TYPE_CATALOG_ORDER)
    opponents = _ship_list(arguments.opponents, DEFAULT_OPPONENTS)
    variants = _variant_map(arguments.variants, ships)
    try:
        matches = build_matches(ships, opponents, arguments.seeds, variants)
    except KeyError as error:
        raise SystemExit(error.args[0]) from None
    started = time.perf_counter()
    results = []
    for result in run_arena(
        matches,
        seconds=arguments.seconds,
        workers=arguments.workers,
        ship_directions=arguments.directions,
    ):
        results.append(result)
        print(
            f"\r{len(results)}/{len(matches)} fights",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr)
    print(format_report(tally(results), opponents))
    print(f"{len(results)} fights in {time.perf_counter() - started:.1f}s")
    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump([asdict(result) for result in results], handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
