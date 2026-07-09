"""Typed training event ledger for reward reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVENT_OBJECT_SPAWNED = "object_spawned"
EVENT_OBJECT_REMOVED = "object_removed"
EVENT_CREW_CHANGED = "crew_changed"
EVENT_BATTERY_CHANGED = "battery_changed"
EVENT_DEBUFF_APPLIED = "debuff_applied"
EVENT_SHIP_DIED = "ship_died"
EVENT_REBIRTH_ATTEMPT = "rebirth_attempt"
EVENT_REBIRTH_COMPLETED = "rebirth_completed"

DEBUFF_LIMPET = "limpet"
DEBUFF_BOARDING_MARINE = "boarding_marine"
DEBUFF_CONFUSION = "confusion"
DEBUFF_DOGI_DRAIN = "dogi_drain"


@dataclass(frozen=True)
class TrainingBattleEvent:
    frame_id: int
    event_type: str
    actor: Any | None = None
    owner: Any | None = None
    target: Any | None = None
    obj: Any | None = None
    magnitude: float = 1.0
    ability_name: str | None = None
    action: str | None = None
    removal_reason: str | None = None
    destroyed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BattleEventLedger:
    """Frame-addressed append-only event sink used only by training."""

    def __init__(self):
        self.current_frame = 0
        self.events: list[TrainingBattleEvent] = []
        self._removed_object_ids: set[int] = set()

    def append(self, event_type: str, **fields) -> TrainingBattleEvent:
        event = TrainingBattleEvent(
            frame_id=int(fields.pop("frame_id", self.current_frame)),
            event_type=event_type,
            **fields,
        )
        self.events.append(event)
        return event

    def record_object_spawned(self, obj) -> TrainingBattleEvent | None:
        if not _is_reward_relevant_object(obj):
            return None
        return self.append(
            EVENT_OBJECT_SPAWNED,
            actor=_root_owner(obj),
            owner=_root_owner(obj),
            obj=obj,
            ability_name=_ability_name(obj),
            action=_action_for_object(obj),
        )

    def record_object_removed(
        self,
        obj,
        *,
        destroyed: bool,
        reason: str,
        actor=None,
    ) -> TrainingBattleEvent | None:
        if not _is_reward_relevant_object(obj) or id(obj) in self._removed_object_ids:
            return None
        self._removed_object_ids.add(id(obj))
        return self.append(
            EVENT_OBJECT_REMOVED,
            actor=actor,
            owner=_root_owner(obj),
            obj=obj,
            ability_name=_ability_name(obj),
            action=_action_for_object(obj),
            removal_reason=reason,
            destroyed=bool(destroyed),
        )

    def record_crew_changed(self, ship, delta: float, *, actor=None, source=None):
        if delta == 0:
            return None
        event = self.append(
            EVENT_CREW_CHANGED,
            actor=actor,
            owner=ship,
            target=ship,
            obj=source,
            magnitude=float(delta),
            ability_name=_ability_name(source),
            action=_action_for_object(source),
        )
        if delta < 0 and getattr(ship, "current_hp", 1) <= 0:
            self.append(
                EVENT_SHIP_DIED,
                actor=actor,
                owner=ship,
                target=ship,
                obj=source,
                ability_name=_ability_name(source),
                action=_action_for_object(source),
            )
        return event

    def record_battery_changed(self, ship, delta: float, *, actor=None, source=None):
        if delta == 0:
            return None
        return self.append(
            EVENT_BATTERY_CHANGED,
            actor=actor,
            owner=ship,
            target=ship,
            obj=source,
            magnitude=float(delta),
            ability_name=_ability_name(source),
            action=_action_for_object(source),
        )

    def record_debuff_applied(
        self,
        ship,
        debuff_type: str,
        *,
        actor=None,
        source=None,
        magnitude: float = 1.0,
    ):
        return self.append(
            EVENT_DEBUFF_APPLIED,
            actor=actor,
            owner=ship,
            target=ship,
            obj=source,
            magnitude=float(magnitude),
            ability_name=_ability_name(source),
            action=_action_for_object(source),
            metadata={"debuff_type": debuff_type},
        )


def ledger_for(obj) -> BattleEventLedger | None:
    seen = set()
    current = obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        ledger = getattr(current, "_training_event_ledger", None)
        if ledger is not None:
            return ledger
        current = getattr(current, "parent", None)
    return None


def bind_ledger(obj, ledger: BattleEventLedger | None) -> None:
    try:
        setattr(obj, "_training_event_ledger", ledger)
    except Exception:
        return


def record_spawned(obj) -> None:
    ledger = ledger_for(obj)
    if ledger is not None:
        ledger.record_object_spawned(obj)


def record_removed(obj, *, destroyed: bool, reason: str, actor=None) -> None:
    ledger = ledger_for(obj)
    if ledger is not None:
        ledger.record_object_removed(
            obj,
            destroyed=destroyed,
            reason=reason,
            actor=actor,
        )


def record_crew_changed(ship, delta: float, *, actor=None, source=None) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_crew_changed(ship, delta, actor=actor, source=source)


def record_battery_changed(ship, delta: float, *, actor=None, source=None) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_battery_changed(ship, delta, actor=actor, source=source)


def record_debuff_applied(
    ship,
    debuff_type: str,
    *,
    actor=None,
    source=None,
    magnitude: float = 1.0,
) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_debuff_applied(
            ship,
            debuff_type,
            actor=actor,
            source=source,
            magnitude=magnitude,
        )


def _is_reward_relevant_object(obj) -> bool:
    return getattr(obj, "type", None) in {"projectile", "special_object", "laser", "area"}


def _root_owner(obj):
    current = getattr(obj, "parent", None)
    if current is None:
        return None
    seen = set()
    while getattr(current, "parent", None) is not None and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "parent", None)
    return current


def _ability_name(obj) -> str | None:
    if obj is None:
        return None
    return str(getattr(obj, "name", getattr(obj, "projectile_name", None)) or "") or None


def _action_for_object(obj) -> str | None:
    action_number = getattr(obj, "action_number", None)
    if action_number in (1, 2, 3):
        return f"A{action_number}"
    name = _ability_name(obj)
    if not name:
        return None
    if name.endswith("A1"):
        return "A1"
    if name.endswith("A2"):
        return "A2"
    if name.endswith("A3"):
        return "A3"
    return None
