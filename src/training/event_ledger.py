"""Typed training event ledger for reward reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import src.const as const


EVENT_OBJECT_SPAWNED = "object_spawned"
EVENT_OBJECT_REMOVED = "object_removed"
EVENT_OBJECT_HP_CHANGED = "object_hp_changed"
EVENT_ACTION_USED = "action_used"
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
        self._crew_loss_totals: dict[int, float] = {}
        self._credited_crew_loss_totals: dict[int, float] = {}

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
        source=None,
    ) -> TrainingBattleEvent | None:
        if not _is_reward_relevant_object(obj) or id(obj) in self._removed_object_ids:
            return None
        source_owner = _root_owner(source) if source is not None else None
        self._removed_object_ids.add(id(obj))
        return self.append(
            EVENT_OBJECT_REMOVED,
            actor=actor if actor is not None else source_owner,
            owner=_root_owner(obj),
            obj=obj,
            ability_name=_ability_name(obj),
            action=_action_for_object(obj),
            removal_reason=reason,
            destroyed=bool(destroyed),
            metadata=_removal_source_metadata(source, source_owner),
        )

    def record_object_hp_changed(self, obj, damage: float, *, source=None):
        damage = max(0.0, float(damage))
        if damage <= 0.0 or getattr(obj, "name", None) != "ChmmrSatellite":
            return None
        source_owner = _root_owner(source) if source is not None else None
        return self.append(
            EVENT_OBJECT_HP_CHANGED,
            actor=source_owner,
            owner=_root_owner(obj),
            target=obj,
            obj=source,
            magnitude=-damage,
            ability_name=_ability_name(source),
            action=_action_for_object(source),
            metadata=_removal_source_metadata(source, source_owner),
        )

    def record_action_used(self, ship, action_number: int) -> TrainingBattleEvent:
        return self.append(
            EVENT_ACTION_USED,
            actor=ship,
            owner=ship,
            target=ship,
            action=f"A{int(action_number)}",
        )

    def record_crew_changed(self, ship, delta: float, *, actor=None, source=None):
        if delta == 0:
            return None
        source_credit = _source_reward_credit(source)
        event = self.append(
            EVENT_CREW_CHANGED,
            actor=actor,
            owner=ship,
            target=ship,
            obj=source,
            magnitude=float(delta),
            ability_name=_ability_name(source),
            action=_action_for_object(source),
            metadata={"source_credit": source_credit},
        )
        if delta < 0:
            self._record_crew_loss_credit(ship, -float(delta), source_credit)
        if delta < 0 and getattr(ship, "current_hp", 1) <= 0:
            self.append(
                EVENT_SHIP_DIED,
                actor=actor,
                owner=ship,
                target=ship,
                obj=source,
                ability_name=_ability_name(source),
                action=_action_for_object(source),
                metadata={"kill_credit": self._kill_credit_for_ship(ship)},
            )
        return event

    def _record_crew_loss_credit(
        self,
        ship,
        crew_lost: float,
        source_credit: float,
    ) -> None:
        ship_id = id(ship)
        loss = max(0.0, float(crew_lost))
        self._crew_loss_totals[ship_id] = (
            self._crew_loss_totals.get(ship_id, 0.0) + loss
        )
        self._credited_crew_loss_totals[ship_id] = (
            self._credited_crew_loss_totals.get(ship_id, 0.0)
            + loss * source_credit
        )

    def _kill_credit_for_ship(self, ship) -> float:
        ship_id = id(ship)
        total_loss = self._crew_loss_totals.get(ship_id, 0.0)
        if total_loss <= 0.0:
            return 1.0
        credited_loss = self._credited_crew_loss_totals.get(ship_id, 0.0)
        return _clamp01(credited_loss / total_loss)

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


def record_removed(
    obj,
    *,
    destroyed: bool,
    reason: str,
    actor=None,
    source=None,
) -> None:
    ledger = ledger_for(obj)
    if ledger is not None:
        ledger.record_object_removed(
            obj,
            destroyed=destroyed,
            reason=reason,
            actor=actor,
            source=source,
        )


def record_object_hp_changed(obj, damage: float, *, source=None) -> None:
    ledger = ledger_for(obj)
    if ledger is not None:
        ledger.record_object_hp_changed(obj, damage, source=source)


def record_action_used(ship, action_number: int) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_action_used(ship, action_number)


def record_crew_changed(ship, delta: float, *, actor=None, source=None) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_crew_changed(ship, delta, actor=actor, source=source)


def record_launched_crew_lost(
    unit,
    *,
    actor=None,
    source=None,
    magnitude: float = 1.0,
) -> None:
    parent = getattr(unit, "parent", None)
    if parent is None:
        return
    record_crew_changed(
        parent,
        -abs(float(magnitude)),
        actor=actor,
        source=source if source is not None else unit,
    )


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


def _removal_source_metadata(source, source_owner) -> dict[str, Any]:
    if source is None:
        return {}
    return {
        "source": source,
        "source_owner": source_owner,
        "source_type": getattr(source, "type", None),
        "source_ability_name": _ability_name(source),
        "source_action": _action_for_object(source),
    }


def _source_reward_credit(source) -> float:
    if source is None:
        return 1.0
    role = getattr(getattr(source, "collision_capabilities", None), "role", None)
    if getattr(role, "name", None) == "PLANET":
        return const.PLANET_KILL_CREDIT
    if _ability_name(source) == "DruugeA2":
        return const.DRUUGE_A2_KILL_CREDIT
    return 1.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
