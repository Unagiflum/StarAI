"""The harm reward scheme: lasting harm to the opponent and to the trainee.

Four components replace every classic one:

- ``Damage to enemy`` (D_enemy): crew, limpet and satellite damage the trainee
  causes the opponent.
- ``Enemy kill`` (K_enemy): opponent deaths and satellite kills the trainee
  causes.
- ``Damage taken`` (D_self): the same damage measured on the trainee, from any
  cause.
- ``Own death`` (K_self): the trainee's deaths and lost satellites.

For a ship of cost V, s = sqrt(V / Vmax) with Vmax the highest cost in the
catalog, HP its normal starting crew, satHP its normal satellite HP, and p 0.5
for the Pkunk (else 1):

    Damage = s * [0.5 * p * hp_damage / HP + L + 0.5 * sat_damage / satHP]
             / (1 + 0.5 * has_sats)
    Kill   = s * (ship_kills + sat_kills / satellite_count) / (1 + has_sats)

where each Pkunk death counts 0.5 and the n-th limpet on a ship adds
0.25 * 0.5 ** (n - 1) to L. The weights multiply these afterwards.

Crew is effective crew: crew aboard plus launched crew still away, so
launching and recovering a unit changes nothing and a unit lost for good is a
1-crew loss. Opponent crew damage counts only new lows the trainee causes;
the trainee's damage follows its crew level, so gains count back.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.training.event_ledger import (
    DEBUFF_LIMPET,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_HP_CHANGED,
    EVENT_OBJECT_REMOVED,
    EVENT_REBIRTH_COMPLETED,
    EVENT_SHIP_DIED,
    TrainingBattleEvent,
)

HARM_DAMAGE_ENEMY = "Damage to enemy"
HARM_KILL_ENEMY = "Enemy kill"
HARM_DAMAGE_TAKEN = "Damage taken"
HARM_OWN_DEATH = "Own death"
HARM_REWARD_COMPONENTS = (
    HARM_DAMAGE_ENEMY,
    HARM_KILL_ENEMY,
    HARM_DAMAGE_TAKEN,
    HARM_OWN_DEATH,
)
DEFAULT_HARM_REWARD_WEIGHTS = {
    HARM_DAMAGE_ENEMY: 1.024,
    HARM_KILL_ENEMY: 1.024,
    HARM_DAMAGE_TAKEN: 0.0,
    HARM_OWN_DEATH: 0.0,
}

CREW_DAMAGE_WEIGHT = 0.5
SATELLITE_DAMAGE_WEIGHT = 0.5
FIRST_LIMPET_WEIGHT = 0.25
LIMPET_WEIGHT_RATIO = 0.5
PKUNK_HARM_FACTOR = 0.5
CHMMR_SATELLITE_NAME = "ChmmrSatellite"


@dataclass(frozen=True)
class HarmProfile:
    """The normalising values of one ship type."""

    scale: float
    normal_crew: float
    satellite_hp: float = 0.0
    satellite_count: int = 0
    life_factor: float = 1.0

    @property
    def has_satellites(self) -> bool:
        return self.satellite_count > 0 and self.satellite_hp > 0.0

    @property
    def _damage_divisor(self) -> float:
        return 1.0 + 0.5 * float(self.has_satellites)

    @property
    def _kill_divisor(self) -> float:
        return 1.0 + float(self.has_satellites)

    def crew_damage(self, crew: float) -> float:
        return (
            self.scale
            * CREW_DAMAGE_WEIGHT
            * self.life_factor
            * float(crew)
            / self.normal_crew
            / self._damage_divisor
        )

    def limpet_damage(self, limpet_number: int) -> float:
        """Worth of the ``limpet_number``-th limpet on the ship (from 1)."""

        return (
            self.scale
            * FIRST_LIMPET_WEIGHT
            * LIMPET_WEIGHT_RATIO ** (int(limpet_number) - 1)
            / self._damage_divisor
        )

    def satellite_damage(self, hp: float) -> float:
        if not self.has_satellites:
            return 0.0
        return (
            self.scale
            * SATELLITE_DAMAGE_WEIGHT
            * float(hp)
            / self.satellite_hp
            / self._damage_divisor
        )

    def ship_kill(self) -> float:
        return self.scale * self.life_factor / self._kill_divisor

    def satellite_kill(self) -> float:
        if not self.has_satellites:
            return 0.0
        return self.scale / self.satellite_count / self._kill_divisor


_PROFILE_CACHE: dict[str, HarmProfile] = {}


def _catalog():
    from src.Objects.Ships.catalog import SHIP_DEFINITIONS

    return SHIP_DEFINITIONS


def _max_ship_cost() -> float:
    costs = [
        float(getattr(definition, "cost", 0) or 0)
        for definition in _catalog().values()
    ]
    return max(costs) if costs else 1.0


def harm_profile_for_name(name: str) -> HarmProfile | None:
    """The catalog profile of a ship type, or ``None`` if it isn't listed."""

    if name in _PROFILE_CACHE:
        return _PROFILE_CACHE[name]
    definition = _catalog().get(name)
    if definition is None:
        return None
    satellite_count = int(getattr(definition, "satellite_count", 0) or 0)
    satellite_hp = float(getattr(definition, "satellite_hp", 0) or 0)
    profile = HarmProfile(
        scale=math.sqrt(float(definition.cost) / _max_ship_cost()),
        normal_crew=max(1.0, float(definition.start_hp)),
        satellite_hp=satellite_count * satellite_hp,
        satellite_count=satellite_count,
        life_factor=PKUNK_HARM_FACTOR if name == "Pkunk" else 1.0,
    )
    _PROFILE_CACHE[name] = profile
    return profile


def harm_profile_for(ship) -> HarmProfile:
    name = str(getattr(ship, "name", "") or "")
    profile = harm_profile_for_name(name)
    if profile is not None:
        return profile
    # Ships outside the catalog (test doubles) count at full scale.
    crew = getattr(ship, "start_hp", None) or getattr(ship, "max_hp", None) or 1
    return HarmProfile(
        scale=1.0,
        normal_crew=max(1.0, float(crew)),
        life_factor=PKUNK_HARM_FACTOR if name == "Pkunk" else 1.0,
    )


@dataclass(frozen=True)
class HarmContribution:
    """One harm amount, placed on the effect frame unless ``relocate`` is set
    and the event's causal credit names an open origin frame."""

    component: str
    amount: float
    event: TrainingBattleEvent | None = None
    relocate: bool = False


@dataclass
class _ShipHarm:
    ship: Any
    profile: HarmProfile
    crew: float
    low: float
    limpets: int


def _root(obj):
    current = obj
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parent = getattr(current, "parent", None)
        if parent is None:
            return current
        current = parent
    return current


def _event_source(event: TrainingBattleEvent):
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    source = metadata.get("source")
    return source if source is not None else event.obj


def caused_by(event: TrainingBattleEvent, ship) -> bool:
    """Whether the event's damage source is ``ship`` or one of its objects."""

    if ship is None:
        return False
    source = _event_source(event)
    return source is not None and _root(source) is ship


def _is_satellite_event(event: TrainingBattleEvent) -> bool:
    target = event.target if event.event_type == EVENT_OBJECT_HP_CHANGED else event.obj
    return str(getattr(target, "name", "") or "") == CHMMR_SATELLITE_NAME


def _is_escrow_event(event: TrainingBattleEvent) -> bool:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    return bool(
        metadata.get("launched_crew_escrow_debit")
        or metadata.get("launched_crew_escrow_refund")
    )


class HarmTracker:
    """Per-round harm bookkeeping, keyed by ship.

    A respawned ship is a new object, so each life is tracked afresh; a Pkunk
    rebirth restarts tracking from the reborn crew.
    """

    def __init__(self):
        self._ships: dict[int, _ShipHarm] = {}
        self.totals = {component: 0.0 for component in HARM_REWARD_COMPONENTS}

    def state_for(self, ship) -> _ShipHarm | None:
        state = self._ships.get(id(ship))
        if state is not None and state.ship is ship:
            return state
        return None

    def _start(self, ship, crew: float, limpets: int) -> _ShipHarm:
        state = _ShipHarm(
            ship=ship,
            profile=harm_profile_for(ship),
            crew=float(crew),
            low=float(crew),
            limpets=max(0, int(limpets)),
        )
        self._ships[id(ship)] = state
        return state

    def _ensure(self, ship, crew, limpets, events) -> _ShipHarm | None:
        if ship is None:
            return None
        state = self.state_for(ship)
        if state is not None:
            return state
        if crew is None:
            # No pre-frame snapshot: undo this frame's crew events.
            crew = float(getattr(ship, "current_hp", 0) or 0) - sum(
                float(event.magnitude)
                for event in events
                if event.event_type == EVENT_CREW_CHANGED
                and event.target is ship
                and not _is_escrow_event(event)
            )
        if limpets is None:
            limpets = int(getattr(ship, "limpets_attached", 0) or 0)
        return self._start(ship, crew, limpets)

    def contributions(self, decision, outcome) -> list[HarmContribution]:
        trainee = decision.self_ship
        enemy = decision.enemy_ship
        events = tuple(outcome.events)
        trainee_state = self._ensure(
            trainee,
            getattr(decision, "self_crew", None),
            getattr(decision, "self_limpets", None),
            events,
        )
        enemy_state = self._ensure(
            enemy,
            getattr(decision, "enemy_crew", None),
            getattr(decision, "enemy_limpets", None),
            events,
        )
        found: list[HarmContribution] = []

        def emit(component, amount, event, relocate=False):
            if amount == 0.0:
                return
            found.append(HarmContribution(component, float(amount), event, relocate))
            self.totals[component] += float(amount)

        for event in events:
            kind = event.event_type
            if kind == EVENT_CREW_CHANGED:
                if _is_escrow_event(event):
                    continue
                delta = float(event.magnitude)
                if enemy_state is not None and event.target is enemy:
                    after = enemy_state.crew + delta
                    enemy_state.crew = after
                    if delta < 0.0:
                        new_low = max(0.0, enemy_state.low - after)
                        if new_low > 0.0 and caused_by(event, trainee):
                            emit(
                                HARM_DAMAGE_ENEMY,
                                enemy_state.profile.crew_damage(new_low),
                                event,
                                relocate=True,
                            )
                        enemy_state.low = min(enemy_state.low, after)
                elif trainee_state is not None and event.target is trainee:
                    trainee_state.crew += delta
                    emit(
                        HARM_DAMAGE_TAKEN,
                        trainee_state.profile.crew_damage(-delta),
                        event,
                        relocate=delta < 0.0 and caused_by(event, trainee),
                    )
            elif kind == EVENT_DEBUFF_APPLIED:
                metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
                if metadata.get("debuff_type") != DEBUFF_LIMPET:
                    continue
                if enemy_state is not None and event.target is enemy:
                    enemy_state.limpets += 1
                    if caused_by(event, trainee):
                        emit(
                            HARM_DAMAGE_ENEMY,
                            enemy_state.profile.limpet_damage(enemy_state.limpets),
                            event,
                            relocate=True,
                        )
                elif trainee_state is not None and event.target is trainee:
                    trainee_state.limpets += 1
                    emit(
                        HARM_DAMAGE_TAKEN,
                        trainee_state.profile.limpet_damage(trainee_state.limpets),
                        event,
                    )
            elif kind == EVENT_OBJECT_HP_CHANGED and _is_satellite_event(event):
                damage = max(0.0, -float(event.magnitude))
                if enemy_state is not None and event.owner is enemy:
                    if caused_by(event, trainee):
                        emit(
                            HARM_DAMAGE_ENEMY,
                            enemy_state.profile.satellite_damage(damage),
                            event,
                            relocate=True,
                        )
                elif trainee_state is not None and event.owner is trainee:
                    emit(
                        HARM_DAMAGE_TAKEN,
                        trainee_state.profile.satellite_damage(damage),
                        event,
                        relocate=caused_by(event, trainee),
                    )
            elif kind == EVENT_OBJECT_REMOVED and _is_satellite_event(event):
                if not event.destroyed:
                    continue
                if enemy_state is not None and event.owner is enemy:
                    if caused_by(event, trainee):
                        emit(
                            HARM_KILL_ENEMY,
                            enemy_state.profile.satellite_kill(),
                            event,
                            relocate=True,
                        )
                elif trainee_state is not None and event.owner is trainee:
                    emit(
                        HARM_OWN_DEATH,
                        trainee_state.profile.satellite_kill(),
                        event,
                        relocate=caused_by(event, trainee),
                    )
            elif kind == EVENT_SHIP_DIED:
                if enemy_state is not None and event.target is enemy:
                    if caused_by(event, trainee):
                        emit(
                            HARM_KILL_ENEMY,
                            enemy_state.profile.ship_kill(),
                            event,
                            relocate=True,
                        )
                elif trainee_state is not None and event.target is trainee:
                    emit(
                        HARM_OWN_DEATH,
                        trainee_state.profile.ship_kill(),
                        event,
                        relocate=caused_by(event, trainee),
                    )
            elif kind == EVENT_REBIRTH_COMPLETED:
                ship = event.target
                if ship is enemy or ship is trainee:
                    state = self._start(
                        ship,
                        float(getattr(ship, "current_hp", 0) or 0),
                        0,
                    )
                    if ship is enemy:
                        enemy_state = state
                    else:
                        trainee_state = state
        return found
