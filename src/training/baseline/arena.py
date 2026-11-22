"""Baseline-versus-baseline arena for comparing ship strategies by harm.

Every fight runs a fixed number of frames (by default 1,200, the training
default) between two baseline-controlled ships. Either ship respawns when
destroyed, as in training rounds, so a fight always runs to the limit. Each
seed is flown in a 2 x 2 matrix of starting conditions: each side starts
strong (its normal starting crew, all satellites) or weak (a quarter of its
crew, one satellite), and respawns that way.

The harm tracker (``src/training/harm.py``) scores each fight from the
battle's events, with the Phase 5 formulas:

- harm dealt: the evaluated ship's damage and kill credit against the
  opponent;
- harm taken: its own damage taken and deaths;
- λ score: 0 when harm dealt is 0, otherwise harm dealt - λ x harm taken,
  with λ the ship's ``harm_taken_weight`` (``behaviours.SHIP_BEHAVIOURS``).

Settings rank by harm dealt first; the λ score only breaks near-ties
(``compare``). Strategy variants (``strategies.STRATEGY_VARIANTS``) fly the
same seeds and conditions as the default rules, so comparisons pair fight by
fight. Opponents always fly their default rules.

Run from the repository root, for example::

    python -m src.training.baseline.arena --ships Spathi --opponents all \\
        --seeds 20 --variant Spathi=approach
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import src.const as const
from src.training.baseline.behaviours import ship_behaviour
from src.training.baseline.strategies import (
    DEFAULT_VARIANT,
    STRATEGY_VARIANTS,
    strategy_for,
)
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER

DEFAULT_OPPONENTS = ("Spathi", "Earthling", "KzerZa")
DEFAULT_FRAMES = 1200
DEFAULT_SHIP_DIRECTIONS = 64
# The game's default, fixed here rather than read from the settings file so
# runs repeat.
DEFAULT_PERCEPTION_DELAY = const.DEFAULT_GAMEPLAY["perception_delay_frames"]
# One worker per logical CPU. On a 24-thread machine (8 performance cores
# with 2 threads each, 8 efficiency cores), 1,200-frame fights ran at 6,430
# frames per second with 24 workers, 6,069 with 20, 5,718 with 16 and 5,126
# with 12.
DEFAULT_WORKERS = os.cpu_count() or 1

STRONG = "strong"
WEAK = "weak"
CONDITIONS = (STRONG, WEAK)
# The 2 x 2 matrix: (evaluated ship, opponent).
CELLS = tuple((mine, theirs) for mine in CONDITIONS for theirs in CONDITIONS)
CELL_CODES = {(mine, theirs): mine[0].upper() + theirs[0].upper() for mine, theirs in CELLS}
WEAK_CREW_FRACTION = 0.25
WEAK_SATELLITES = 1
# A frame counts as still while the evaluated ship, alive, flies below this
# fraction of its top speed.
STILL_SPEED_FRACTION = 0.1
# One setting is clearly ahead of another on a measure when the paired 95%
# interval of the difference excludes zero and the gap is more than this
# share of the larger mean. Screening's lead group is the settings within
# this share of the highest harm dealt.
CLEAR_LEAD_FRACTION = 0.05
# Screening: start every setting on this many seeds, keep the better half,
# double the seeds, and repeat up to the maximum.
SCREEN_FIRST_SEEDS = 10
SCREEN_MAX_SEEDS = 100


@dataclass(frozen=True, slots=True)
class ArenaMatch:
    """One fight to run: ``ship`` flying ``variant`` against ``opponent``."""

    ship: str
    opponent: str
    seed: int
    variant: str = DEFAULT_VARIANT
    ship_condition: str = STRONG
    opponent_condition: str = STRONG


@dataclass(frozen=True, slots=True)
class ArenaResult:
    ship: str
    variant: str
    opponent: str
    seed: int
    ship_condition: str
    opponent_condition: str
    frames: int
    # λ for this matchup when the fight ran.
    harm_taken_weight: float
    # Harm, in the Phase 5 units, split into damage and kills.
    damage_dealt: float = 0.0
    kills_dealt: float = 0.0
    damage_taken: float = 0.0
    deaths_taken: float = 0.0
    # Diagnostics: opponent ships the evaluated ship destroyed, its own
    # deaths, and frames it spent alive below STILL_SPEED_FRACTION of top
    # speed.
    kills: int = 0
    deaths: int = 0
    still_frames: int = 0

    @property
    def harm_dealt(self) -> float:
        return self.damage_dealt + self.kills_dealt

    @property
    def harm_taken(self) -> float:
        return self.damage_taken + self.deaths_taken

    @property
    def score(self) -> float:
        """The λ score: nothing without harm dealt."""

        if self.harm_dealt <= 0.0:
            return 0.0
        return self.harm_dealt - self.harm_taken_weight * self.harm_taken

    @property
    def cell(self) -> str:
        return CELL_CODES[(self.ship_condition, self.opponent_condition)]

    @property
    def pair_key(self) -> tuple:
        """What two variants' fights share when they meet the same battle."""

        return (self.ship, self.opponent, self.seed, self.ship_condition, self.opponent_condition)


def apply_arena_settings(
    ship_directions: int = DEFAULT_SHIP_DIRECTIONS,
    perception_delay: int = DEFAULT_PERCEPTION_DELAY,
) -> None:
    """Apply the gameplay settings fights run under (once per process)."""

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    const.apply_game_settings(
        SimpleNamespace(
            asteroid_count=const.DEFAULT_GAMEPLAY["asteroid_count"],
            ship_directions=int(ship_directions),
            repeat_key_delay=const.DEFAULT_GAMEPLAY["repeat_key_delay"],
            perception_delay_frames=int(perception_delay),
        )
    )


def _battle_key(match: ArenaMatch) -> str:
    # The variant is left out so every variant meets the same battles.
    return (
        f"{match.ship}|{match.opponent}|{match.seed}|"
        f"{match.ship_condition}|{match.opponent_condition}"
    )


def _match_rng(match: ArenaMatch) -> random.Random:
    return random.Random(_battle_key(match))


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


def _alive(ship) -> bool:
    return bool(getattr(ship, "currently_alive", True)) and ship.current_hp > 0


def spawn_crew(ship, condition: str) -> int:
    start_hp = max(1, int(getattr(ship, "start_hp", getattr(ship, "current_hp", 1))))
    if condition == STRONG:
        return start_hp
    return max(1, math.ceil(start_hp * WEAK_CREW_FRACTION))


_SIMULATION_CLASS = None


def _arena_simulation_class():
    """A training-mode battle whose spawns follow the arena's conditions.

    Training mode respawns a destroyed ship; this subclass sets each new
    ship's crew and satellites from its side's condition instead of the
    battle's random stream, so every variant meets the same starts.
    """

    global _SIMULATION_CLASS
    if _SIMULATION_CLASS is not None:
        return _SIMULATION_CLASS
    from src.Battle.battle import BattleSimulation

    class ArenaSimulation(BattleSimulation):
        def __init__(self, *args, conditions: Mapping[int, str], spawn_key: str, **kwargs):
            self.arena_conditions = dict(conditions)
            self._arena_spawn_key = str(spawn_key)
            self._arena_lives = {1: 0, 2: 0}
            super().__init__(*args, **kwargs)

        def _prepare_training_spawn(self, ship):
            player = int(ship.player)
            life = self._arena_lives.get(player, 0)
            self._arena_lives[player] = life + 1
            condition = self.arena_conditions.get(player, STRONG)
            rng = random.Random(f"{self._arena_spawn_key}|{player}|{life}")
            ship.current_hp = spawn_crew(ship, condition)
            ship.currently_alive = True
            if getattr(ship, "name", None) == "Shofixti":
                ship.shofixti_arming_stage = getattr(ship, "ARMED", 2)
            spawn_satellites = getattr(ship, "spawn_satellites", None)
            if spawn_satellites is not None:
                spawn_satellites(rng=rng, randomized_health=False)
                satellites = list(ship.drain_spawned_objects())
                if condition == WEAK:
                    satellites = satellites[:WEAK_SATELLITES]
                self.world.add_all(satellites)

    _SIMULATION_CLASS = ArenaSimulation
    return ArenaSimulation


def new_arena_battle(match: ArenaMatch):
    """The match's training-mode battle and the event ledger it records to."""

    from src.Objects.Ships.registry import create_ship
    from src.audio import NullAudioService
    from src.training.event_ledger import BattleEventLedger

    audio = NullAudioService()
    resources = _arena_resources()
    ledger = BattleEventLedger()
    simulation = _arena_simulation_class()(
        None,
        create_ship(match.ship, 1, resources=resources, audio_service=audio),
        create_ship(match.opponent, 2, resources=resources, audio_service=audio),
        audio_service=audio,
        rng=_match_rng(match),
        include_stars=False,
        resources=resources,
        training_event_ledger=ledger,
        conditions={1: match.ship_condition, 2: match.opponent_condition},
        spawn_key=_battle_key(match),
    )
    # Explosions and the like are visual only.
    simulation.visual_effects_enabled = False
    return simulation, ledger


def run_match(match: ArenaMatch, *, frames: int = DEFAULT_FRAMES) -> ArenaResult:
    """Fly one fixed-length fight between two baseline-controlled ships."""

    from src.Battle.computer_control import guard_computer_controls
    from src.training.baseline.controller import BaselineController
    from src.training.baseline.observation_view import ObservationView
    from src.training.baseline.ship_specs import ship_spec
    from src.training.contracts import action_for_index
    from src.training.harm import (
        HARM_DAMAGE_ENEMY,
        HARM_DAMAGE_TAKEN,
        HARM_KILL_ENEMY,
        HARM_OWN_DEATH,
        HarmTracker,
    )
    from src.training.observation import build_observation_context, encode_observation

    simulation, ledger = new_arena_battle(match)
    controllers = {
        1: BaselineController(match.ship, match.variant),
        2: BaselineController(match.opponent),
    }
    flown = {1: simulation.player1, 2: simulation.player2}
    tracker = HarmTracker()
    still_speed = STILL_SPEED_FRACTION * ship_spec(match.ship).max_thrust
    still_frames = kills = deaths = 0
    frame_limit = max(1, int(frames))
    for _ in range(frame_limit):
        ships = {1: simulation.player1, 2: simulation.player2}
        for player, ship in ships.items():
            if ship is not flown[player]:
                # A respawned ship starts its rules afresh.
                controllers[player].reset()
                flown[player] = ship
        context = build_observation_context(ships[1], ships[2], simulation.world)
        actions = {}
        for player, ship in ships.items():
            enemy = ships[3 - player]
            observation = encode_observation(
                ship,
                enemy,
                frame_id=simulation.frame_id,
                game_objects=simulation.world,
                context=context,
                validate=False,
            )
            action = action_for_index(controllers[player].decide(observation))
            actions[player] = guard_computer_controls(action, ship, enemy)
            if player == 1 and _alive(ship):
                if ObservationView(observation).me.speed < still_speed:
                    still_frames += 1
        decision = SimpleNamespace(
            self_ship=ships[1],
            enemy_ship=ships[2],
            self_crew=float(ships[1].current_hp),
            enemy_crew=float(ships[2].current_hp),
            self_limpets=int(getattr(ships[1], "limpets_attached", 0) or 0),
            enemy_limpets=int(getattr(ships[2], "limpets_attached", 0) or 0),
        )
        event_start = len(ledger.events)
        simulation.step(actions=actions)
        tracker.contributions(decision, SimpleNamespace(events=ledger.events[event_start:]))
        died = set(getattr(simulation, "training_episode_deaths", ()))
        killers = set(getattr(simulation, "training_episode_kills", ()))
        deaths += int(1 in died)
        kills += int(2 in died and 1 in killers)
    totals = tracker.totals
    return ArenaResult(
        ship=match.ship,
        variant=match.variant,
        opponent=match.opponent,
        seed=match.seed,
        ship_condition=match.ship_condition,
        opponent_condition=match.opponent_condition,
        frames=frame_limit,
        harm_taken_weight=float(ship_behaviour(match.ship, match.opponent).harm_taken_weight),
        damage_dealt=float(totals[HARM_DAMAGE_ENEMY]),
        kills_dealt=float(totals[HARM_KILL_ENEMY]),
        damage_taken=float(totals[HARM_DAMAGE_TAKEN]),
        deaths_taken=float(totals[HARM_OWN_DEATH]),
        kills=kills,
        deaths=deaths,
        still_frames=still_frames,
    )


def build_matches(
    ships: Sequence[str],
    opponents: Sequence[str],
    seeds: int | Sequence[int],
    variants: Mapping[str, Sequence[str]] | None = None,
    *,
    cells: Sequence[tuple[str, str]] = CELLS,
) -> list[ArenaMatch]:
    """Every ship and variant against every opponent, seed and cell.

    ``seeds`` is a count (seeds 0 to n - 1) or the seeds themselves.
    """

    variants = variants or {}
    seed_list = list(range(int(seeds))) if isinstance(seeds, int) else [int(s) for s in seeds]
    matches = []
    for ship in ships:
        ship_variants = [DEFAULT_VARIANT, *variants.get(ship, ())]
        for variant in dict.fromkeys(ship_variants):
            strategy_for(ship, variant)  # Fail fast on unknown names.
            for opponent in opponents:
                for seed in seed_list:
                    for mine, theirs in cells:
                        matches.append(ArenaMatch(ship, opponent, seed, variant, mine, theirs))
    return matches


def _worker_init(ship_directions: int, perception_delay: int) -> None:
    apply_arena_settings(ship_directions, perception_delay)


def _run_one(arguments: tuple[ArenaMatch, int]) -> ArenaResult:
    match, frames = arguments
    return run_match(match, frames=frames)


def run_arena(
    matches: Sequence[ArenaMatch],
    *,
    frames: int = DEFAULT_FRAMES,
    workers: int = DEFAULT_WORKERS,
    ship_directions: int = DEFAULT_SHIP_DIRECTIONS,
    perception_delay: int = DEFAULT_PERCEPTION_DELAY,
) -> Iterable[ArenaResult]:
    """Yield results in ``matches`` order, using worker processes if asked."""

    work = [(match, int(frames)) for match in matches]
    if int(workers) <= 1:
        apply_arena_settings(ship_directions, perception_delay)
        for item in work:
            yield _run_one(item)
        return
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(int(ship_directions), int(perception_delay)),
    ) as pool:
        yield from pool.map(_run_one, work, chunksize=1)


def describe_match(match: ArenaMatch) -> str:
    cell = CELL_CODES[(match.ship_condition, match.opponent_condition)]
    return f"{match.ship} [{match.variant}] vs {match.opponent}, seed {match.seed} {cell}"


def progress_text(done: int, total: int, elapsed: float, now: ArenaMatch | None) -> str:
    """One status line: fights done of the batch, time, and the oldest
    fight still in flight."""

    share = done / total if total else 1.0
    text = f"{done}/{total} fights ({share:.0%})  {elapsed:.0f}s"
    if 0 < done < total:
        text += f", ~{elapsed * (total - done) / done:.0f}s left"
    if now is not None and done < total:
        text += f"  now: {describe_match(now)}"
    return text


def run_with_progress(
    matches: Sequence[ArenaMatch],
    *,
    frames: int = DEFAULT_FRAMES,
    workers: int = DEFAULT_WORKERS,
    ship_directions: int = DEFAULT_SHIP_DIRECTIONS,
    perception_delay: int = DEFAULT_PERCEPTION_DELAY,
    stream=None,
) -> list[ArenaResult]:
    """``run_arena`` with a status line on ``stream`` (stderr by default).

    On a terminal the line is rewritten in place after every fight;
    otherwise a line is written at each tenth of the batch.
    """

    stream = stream if stream is not None else sys.stderr
    live = bool(getattr(stream, "isatty", lambda: False)())
    total = len(matches)
    started = time.perf_counter()
    flown: list[ArenaResult] = []
    width = 0
    # Off a terminal, the next line is due once done reaches this many
    # tenths of the batch.
    tenths = 1
    for result in run_arena(
        matches,
        frames=frames,
        workers=workers,
        ship_directions=ship_directions,
        perception_delay=perception_delay,
    ):
        flown.append(result)
        done = len(flown)
        now = matches[done] if done < total else None
        text = progress_text(done, total, time.perf_counter() - started, now)
        if live:
            text = text[: max(20, shutil.get_terminal_size().columns - 1)]
            stream.write("\r" + text.ljust(width))
            width = max(width, len(text))
            stream.flush()
        elif done * 10 >= tenths * total:
            stream.write(text + "\n")
            stream.flush()
            while tenths * total <= done * 10:
                tenths += 1
    if live:
        stream.write("\n")
        stream.flush()
    return flown


@dataclass(slots=True)
class ArenaTally:
    fights: int = 0
    harm_dealt: float = 0.0
    harm_taken: float = 0.0
    score: float = 0.0
    kills: int = 0
    deaths: int = 0
    harmless_fights: int = 0
    still_frames: int = 0
    frames: int = 0

    def add(self, result: ArenaResult) -> None:
        self.fights += 1
        self.harm_dealt += result.harm_dealt
        self.harm_taken += result.harm_taken
        self.score += result.score
        self.kills += result.kills
        self.deaths += result.deaths
        self.harmless_fights += int(result.harm_dealt <= 0.0)
        self.still_frames += result.still_frames
        self.frames += result.frames

    @property
    def mean_dealt(self) -> float:
        return self.harm_dealt / self.fights if self.fights else 0.0

    @property
    def mean_taken(self) -> float:
        return self.harm_taken / self.fights if self.fights else 0.0

    @property
    def mean_score(self) -> float:
        return self.score / self.fights if self.fights else 0.0

    @property
    def harmless_share(self) -> float:
        return self.harmless_fights / self.fights if self.fights else 0.0

    @property
    def still_share(self) -> float:
        return self.still_frames / self.frames if self.frames else 0.0

    def cell(self) -> str:
        return (
            f"{self.fights:4d} fights  dealt {self.harm_dealt:7.2f} ({self.mean_dealt:5.3f})  "
            f"taken {self.harm_taken:7.2f} ({self.mean_taken:5.3f})  "
            f"score {self.score:7.2f} ({self.mean_score:6.3f})  "
            f"K{self.kills} D{self.deaths}  no harm {self.harmless_share:4.0%}  "
            f"still {self.still_share:4.0%}"
        )


def tally(
    results: Iterable[ArenaResult],
) -> dict[tuple[str, str], dict[str, ArenaTally]]:
    """Group results by (ship, variant), then by ``"all"``, by opponent, and
    by matrix cell (``"cell SS"`` and so on: my condition, then the
    opponent's)."""

    table: dict[tuple[str, str], dict[str, ArenaTally]] = {}
    for result in results:
        rows = table.setdefault((result.ship, result.variant), {})
        for key in ("all", result.opponent, f"cell {result.cell}"):
            rows.setdefault(key, ArenaTally()).add(result)
    return table


# Two-sided 95% critical values of Student's t by degrees of freedom; a
# count between entries uses the entry below it, which is conservative.
_T_CRITICAL = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447),
    (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228), (12, 2.179), (15, 2.131),
    (20, 2.086), (25, 2.060), (30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980),
)


def _t_critical(degrees: int) -> float:
    if degrees > 1000:
        return 1.960
    value = _T_CRITICAL[0][1]
    for limit, critical in _T_CRITICAL:
        if limit > degrees:
            break
        value = critical
    return value


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Mean of (other - base) over paired fights, with its 95% interval."""

    count: int
    base_mean: float
    other_mean: float
    mean: float
    low: float
    high: float

    @property
    def lead(self) -> int:
        """+1 when the other is clearly ahead, -1 when the base is, else 0."""

        larger = max(abs(self.base_mean), abs(self.other_mean))
        if self.count < 2 or abs(self.mean) <= CLEAR_LEAD_FRACTION * larger:
            return 0
        if self.low > 0.0:
            return 1
        if self.high < 0.0:
            return -1
        return 0

    def text(self) -> str:
        return f"{self.mean:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"


def paired_difference(base: Sequence[float], other: Sequence[float]) -> PairedDifference:
    differences = [b - a for a, b in zip(base, other)]
    count = len(differences)
    if count == 0:
        return PairedDifference(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = statistics.fmean(differences)
    half = 0.0
    if count > 1:
        half = _t_critical(count - 1) * statistics.stdev(differences) / math.sqrt(count)
    return PairedDifference(
        count=count,
        base_mean=statistics.fmean(base),
        other_mean=statistics.fmean(other),
        mean=mean,
        low=mean - half,
        high=mean + half,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """How ``other`` ranks against ``base`` over their shared fights."""

    dealt: PairedDifference
    score: PairedDifference
    # +1: other ranks higher; -1: base does; 0: equal.
    verdict: int
    reason: str

    def text(self) -> str:
        outcome = {1: "higher", -1: "lower", 0: "equal"}[self.verdict]
        return (
            f"{outcome} ({self.reason}; {self.dealt.count} paired fights): "
            f"harm dealt {self.dealt.text()}, lambda score {self.score.text()}"
        )


def compare(base: Iterable[ArenaResult], other: Iterable[ArenaResult]) -> Comparison:
    """Rank ``other`` against ``base``: harm dealt first, then the λ score.

    Fights pair by battle (``ArenaResult.pair_key``); unpaired ones are left
    out.
    """

    base_by_key = {result.pair_key: result for result in base}
    pairs = [
        (base_by_key[result.pair_key], result)
        for result in other
        if result.pair_key in base_by_key
    ]
    dealt = paired_difference([a.harm_dealt for a, _ in pairs], [b.harm_dealt for _, b in pairs])
    score = paired_difference([a.score for a, _ in pairs], [b.score for _, b in pairs])
    if dealt.lead:
        return Comparison(dealt, score, dealt.lead, "harm dealt")
    if score.lead:
        return Comparison(dealt, score, score.lead, "tied on harm dealt; lambda score")
    return Comparison(dealt, score, 0, "tied on harm dealt and lambda score")


def rank_order(tallies: Mapping[str, ArenaTally]) -> list[str]:
    """Settings in screening order: the lead group (within
    ``CLEAR_LEAD_FRACTION`` of the highest harm dealt) by λ score, then the
    rest by harm dealt."""

    if not tallies:
        return []
    top = max(entry.mean_dealt for entry in tallies.values())
    floor = top * (1.0 - CLEAR_LEAD_FRACTION)
    lead = [name for name, entry in tallies.items() if entry.mean_dealt >= floor]
    rest = [name for name in tallies if name not in lead]
    lead.sort(key=lambda name: (-tallies[name].mean_score, -tallies[name].mean_dealt))
    rest.sort(key=lambda name: -tallies[name].mean_dealt)
    return lead + rest


def screen(
    ship: str,
    variants: Sequence[str],
    opponents: Sequence[str],
    *,
    first_seed: int = 0,
    first_seeds: int = SCREEN_FIRST_SEEDS,
    max_seeds: int = SCREEN_MAX_SEEDS,
    cells: Sequence[tuple[str, str]] = CELLS,
    run=None,
    log=None,
) -> tuple[list[str], dict[str, list[ArenaResult]]]:
    """Drop settings as they fall behind; return the settings of the last
    round, best first, and every result flown.

    Every setting starts on ``first_seeds`` seeds. After each round:

    - settings whose fights all came out identical are merged into the one
      listed first (``variants`` order), as they fly the same way;
    - settings clearly behind the leader (``compare``) are dropped;
    - at most the better half (``rank_order``) is kept.

    The screen stops once one setting is left, without flying it further,
    or once ``max_seeds`` is reached. Otherwise it doubles the seeds,
    flying only the new ones.
    """

    run = run or (lambda matches: list(run_arena(matches)))
    alive = list(dict.fromkeys(variants))
    results: dict[str, list[ArenaResult]] = {name: [] for name in alive}
    flown = 0
    target = min(int(first_seeds), int(max_seeds))
    rounds = 0
    while True:
        seeds = range(first_seed + flown, first_seed + target)
        matches = [
            ArenaMatch(ship, opponent, seed, name, mine, theirs)
            for name in alive
            for opponent in opponents
            for seed in seeds
            for mine, theirs in cells
        ]
        rounds += 1
        if log is not None:
            log(
                f"Round {rounds}: {len(alive)} settings x seeds {seeds.start}-{seeds.stop - 1}"
                f" x {len(cells)} cells x {len(opponents)} opponents = {len(matches)} fights"
            )
        for result in run(matches):
            results[result.variant].append(result)
        flown = target
        alive, duplicates = _merge_identical(alive, results)
        tallies = {}
        for name in alive:
            entry = ArenaTally()
            for result in results[name]:
                entry.add(result)
            tallies[name] = entry
        order = rank_order(tallies)
        leader = order[0]
        behind = {
            name
            for name in order[1:]
            if compare(results[leader], results[name]).verdict < 0
        }
        if log is not None:
            log(f"{flown} seeds:")
            for name in order:
                note = "  (clearly behind)" if name in behind else ""
                log(f"    {name:20s} {tallies[name].cell()}{note}")
            for name, kept in duplicates.items():
                log(f"    {name} flies as {kept}; merged")
        keep = [name for name in order if name not in behind]
        alive = keep[: max(1, math.ceil(len(order) / 2))]
        if len(alive) <= 1 or flown >= max_seeds:
            return order, results
        target = min(max_seeds, flown * 2)


def _merge_identical(
    names: Sequence[str],
    results: Mapping[str, Sequence[ArenaResult]],
) -> tuple[list[str], dict[str, str]]:
    """Keep the first of each group of settings whose fights all came out
    identical; return the kept names and each merged name's keeper."""

    kept: dict[tuple, str] = {}
    merged: dict[str, str] = {}
    for name in names:
        signature = tuple(
            sorted(
                (
                    result.pair_key,
                    result.damage_dealt,
                    result.kills_dealt,
                    result.damage_taken,
                    result.deaths_taken,
                    result.kills,
                    result.deaths,
                    result.still_frames,
                )
                for result in results[name]
            )
        )
        if signature in kept:
            merged[name] = kept[signature]
        else:
            kept[signature] = name
    return list(kept.values()), merged


def format_report(
    table: Mapping[tuple[str, str], Mapping[str, ArenaTally]],
    opponents: Sequence[str],
    *,
    by_cell: bool = False,
) -> str:
    lines = []
    for (ship, variant), rows in table.items():
        lines.append(f"{ship} [{variant}]  {rows['all'].cell()}")
        for opponent in opponents:
            if opponent in rows:
                lines.append(f"    vs {opponent:11s} {rows[opponent].cell()}")
        if by_cell:
            for code in CELL_CODES.values():
                key = f"cell {code}"
                if key in rows:
                    lines.append(f"    cell {code}        {rows[key].cell()}")
    return "\n".join(lines)


def format_comparisons(results: Sequence[ArenaResult]) -> str:
    """Each variant against its ship's default rules."""

    grouped: dict[tuple[str, str], list[ArenaResult]] = {}
    for result in results:
        grouped.setdefault((result.ship, result.variant), []).append(result)
    lines = []
    for (ship, variant), own in grouped.items():
        base = grouped.get((ship, DEFAULT_VARIANT))
        if variant == DEFAULT_VARIANT or base is None:
            continue
        lines.append(f"{ship} [{variant}] vs [{DEFAULT_VARIANT}]: {compare(base, own).text()}")
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


def _cell_list(text: str | None) -> list[tuple[str, str]]:
    if not text or text.strip().lower() == "all":
        return list(CELLS)
    by_code = {code: cell for cell, code in CELL_CODES.items()}
    codes = [code.strip().upper() for code in text.split(",") if code.strip()]
    unknown = [code for code in codes if code not in by_code]
    if unknown:
        raise SystemExit(f"Unknown cells: {', '.join(unknown)} (use SS, SW, WS, WW)")
    return [by_code[code] for code in codes]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.baseline.arena",
        description="Fight baseline strategies against each other, scored by harm.",
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
    parser.add_argument("--seeds", type=int, default=4, help="Seeds per pairing.")
    parser.add_argument("--first-seed", type=int, default=0, help="First seed.")
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="Frames per fight.",
    )
    parser.add_argument(
        "--cells",
        help="Starting-condition cells: all, or some of SS,SW,WS,WW (mine, theirs).",
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
    parser.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Screen one ship's default and variants, dropping settings as they "
            f"fall behind, from {SCREEN_FIRST_SEEDS} seeds up to --seeds."
        ),
    )
    parser.add_argument("--by-cell", action="store_true", help="Report each matrix cell.")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="Worker processes."
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=DEFAULT_SHIP_DIRECTIONS,
        choices=(16, 32, 64),
        help="Ship heading directions.",
    )
    parser.add_argument(
        "--perception-delay",
        type=_perception_delay,
        default=DEFAULT_PERCEPTION_DELAY,
        help=(
            "Frames before a ship sees an enemy object "
            f"(0-{const.PERCEPTION_DELAY_MAX_FRAMES}, "
            f"default {DEFAULT_PERCEPTION_DELAY})."
        ),
    )
    parser.add_argument("--json", help="Also write every fight result to this file.")
    return parser.parse_args(argv)


def _perception_delay(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if not 0 <= value <= const.PERCEPTION_DELAY_MAX_FRAMES:
        raise argparse.ArgumentTypeError(
            f"must be from 0 to {const.PERCEPTION_DELAY_MAX_FRAMES}"
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    ships = _ship_list(arguments.ships, SHIP_TYPE_CATALOG_ORDER)
    opponents = _ship_list(arguments.opponents, DEFAULT_OPPONENTS)
    variants = _variant_map(arguments.variants, ships)
    cells = _cell_list(arguments.cells)
    started = time.perf_counter()

    def run(matches):
        return run_with_progress(
            matches,
            frames=arguments.frames,
            workers=arguments.workers,
            ship_directions=arguments.directions,
            perception_delay=arguments.perception_delay,
        )

    print(f"Perception delay: {arguments.perception_delay} frames")
    if arguments.screen:
        if len(ships) != 1:
            raise SystemExit("--screen takes one ship")
        ship = ships[0]
        names = [DEFAULT_VARIANT, *variants.get(ship, ())]
        try:
            for name in names:
                strategy_for(ship, name)
        except KeyError as error:
            raise SystemExit(error.args[0]) from None
        order, by_name = screen(
            ship,
            names,
            opponents,
            first_seed=arguments.first_seed,
            max_seeds=max(1, arguments.seeds),
            cells=cells,
            run=run,
            log=print,
        )
        results = [result for name in order for result in by_name[name]]
        print(f"Best: {order[0]}")
    else:
        try:
            matches = build_matches(
                ships,
                opponents,
                range(arguments.first_seed, arguments.first_seed + arguments.seeds),
                variants,
                cells=cells,
            )
        except KeyError as error:
            raise SystemExit(error.args[0]) from None
        results = run(matches)
    print(format_report(tally(results), opponents, by_cell=arguments.by_cell))
    comparisons = format_comparisons(results)
    if comparisons:
        print(comparisons)
    print(
        f"{len(results)} fights in {time.perf_counter() - started:.1f}s "
        f"at a perception delay of {arguments.perception_delay} frames"
    )
    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump([asdict(result) for result in results], handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
