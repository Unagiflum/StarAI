"""Typed training event ledger for reward reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import src.const as const
from src.training.causal_credit import (
    AbilityRewardCredit,
    CausalRewardDiagnostics,
    ORIGIN_KIND_AUTONOMOUS_FIRE,
    ORIGIN_KIND_LAUNCH,
    ORIGIN_KIND_PRESS,
    REWARD_MODE_LEGACY,
    REWARD_MODES,
    bind_reward_credit,
    full_weight_credit,
    inherit_reward_credit,
    new_trajectory_id,
    reward_credit_for,
    replace_release_half,
)


EVENT_OBJECT_SPAWNED = "object_spawned"
EVENT_OBJECT_REMOVED = "object_removed"
EVENT_OBJECT_HP_CHANGED = "object_hp_changed"
EVENT_ACTION_USED = "action_used"
EVENT_ACTION_RELEASED = "action_released"
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
        self._enemy_death_credited_crew_loss_totals: dict[int, float] = {}
        self.trainee_ship = None
        self.active_trajectory_id: str | None = None
        self.current_decision_frame: int | None = None
        self.current_action_index: int | None = None
        self.enemy_death_count = 0
        self._spawn_sequence = 0
        self.diagnostics = CausalRewardDiagnostics()
        self.reward_mode = REWARD_MODE_LEGACY

    def start_reward_trajectory(
        self,
        trainee_ship,
        *,
        trajectory_id: str | None = None,
    ) -> str:
        """Open a unique trainee-life reward trajectory."""

        self.trainee_ship = trainee_ship
        self.active_trajectory_id = str(trajectory_id or new_trajectory_id())
        self.current_decision_frame = None
        self.current_action_index = None
        self.enemy_death_count = 0
        return self.active_trajectory_id

    def close_reward_trajectory(self) -> str | None:
        trajectory_id = self.active_trajectory_id
        self.active_trajectory_id = None
        self.current_decision_frame = None
        self.current_action_index = None
        return trajectory_id

    def begin_decision(
        self,
        trainee_ship,
        frame_index: int,
        action_index: int,
        *,
        reward_mode: str | None = None,
    ) -> None:
        """Expose the staged parent decision before simulation updates run."""

        if self.active_trajectory_id is None or self.trainee_ship is not trainee_ship:
            self.start_reward_trajectory(trainee_ship)
        self.current_decision_frame = int(frame_index)
        self.current_action_index = int(action_index)
        if reward_mode is not None:
            if reward_mode not in REWARD_MODES:
                raise ValueError("unsupported reward mode")
            self.reward_mode = str(reward_mode)

    def credit_is_open(self, credit: AbilityRewardCredit | None) -> bool:
        return bool(
            credit is not None
            and self.active_trajectory_id is not None
            and credit.trajectory_id == self.active_trajectory_id
        )

    def bind_committed_action(self, ship, action_number: int, spawned_objects) -> None:
        if (
            ship is not self.trainee_ship
            or self.active_trajectory_id is None
            or self.current_decision_frame is None
        ):
            return
        kind = ORIGIN_KIND_PRESS if int(action_number) == 1 else ORIGIN_KIND_LAUNCH
        credit = full_weight_credit(
            self.active_trajectory_id,
            self.current_decision_frame,
            kind=kind,
        )
        stamp = self._next_spawn_stamp(self.current_decision_frame)
        for obj in tuple(spawned_objects):
            bind_reward_credit(obj, credit)
            _bind_spawn_stamp(obj, stamp)
            obj._training_origin_enemy_death_count = self.enemy_death_count

    def bind_autonomous_fire(self, obj, root_parent) -> AbilityRewardCredit | None:
        if (
            root_parent is not self.trainee_ship
            or self.active_trajectory_id is None
            or self.current_decision_frame is None
        ):
            return None
        credit = full_weight_credit(
            self.active_trajectory_id,
            self.current_decision_frame,
            kind=ORIGIN_KIND_AUTONOMOUS_FIRE,
        )
        bound = bind_reward_credit(obj, credit)
        if bound is not None:
            _bind_spawn_stamp(
                obj,
                self._next_spawn_stamp(self.current_decision_frame),
            )
            obj._training_origin_enemy_death_count = self.enemy_death_count
        return bound

    def _next_spawn_stamp(self, frame_index: int | None = None) -> tuple[int, int]:
        self._spawn_sequence += 1
        frame = self.current_frame if frame_index is None else frame_index
        return int(frame), self._spawn_sequence

    def resolve_event_credit(
        self,
        event: TrainingBattleEvent,
        *,
        component: str,
        expected: bool = False,
    ) -> AbilityRewardCredit | None:
        """Resolve an immutable event-time credit and update shadow diagnostics."""

        ability = event.metadata.get("source_ability_name") or event.ability_name or "unknown"
        credit = causal_credit_for_event(event)
        if credit is None:
            if expected:
                self.diagnostics.missing_provenance[str(ability)] += 1
            return None
        if not self.credit_is_open(credit):
            self.diagnostics.closed_trajectory_rejections[str(ability)] += 1
            return None
        self.diagnostics.routed_events[(str(component), str(ability))] += 1
        origin_deaths = int(event.metadata.get("source_origin_enemy_death_count", 0))
        effect_deaths = int(event.metadata.get("effect_enemy_death_count", 0))
        if effect_deaths > origin_deaths:
            self.diagnostics.cross_enemy_death_effects[str(ability)] += 1
        return credit

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
        if (
            reward_credit_for(obj) is not None
            and _root_owner(obj) is self.trainee_ship
            and spawn_stamp_for(obj) is None
        ):
            _bind_spawn_stamp(obj, self._next_spawn_stamp())
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
            metadata=self._source_metadata(source, source_owner),
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
            metadata=self._source_metadata(source, source_owner),
        )

    def record_action_used(self, ship, action_number: int) -> TrainingBattleEvent:
        return self.append(
            EVENT_ACTION_USED,
            actor=ship,
            owner=ship,
            target=ship,
            action=f"A{int(action_number)}",
        )

    def record_action_released(
        self,
        ship,
        affected_objects,
    ) -> TrainingBattleEvent | None:
        affected = tuple(obj for obj in affected_objects if obj is not None)
        if not affected:
            return None
        if (
            self.reward_mode != REWARD_MODE_LEGACY
            and ship is self.trainee_ship
            and self.current_decision_frame is not None
        ):
            for obj in affected:
                credit = reward_credit_for(obj)
                if self.credit_is_open(credit):
                    bind_reward_credit(
                        obj,
                        replace_release_half(
                            credit,
                            release_frame=self.current_decision_frame,
                        ),
                    )
        return self.append(
            EVENT_ACTION_RELEASED,
            actor=ship,
            owner=ship,
            target=ship,
            action="A1",
            metadata={
                "affected_objects": affected,
                "action_index": self.current_action_index,
                "trajectory_id": self.active_trajectory_id,
            },
        )

    def record_crew_changed(
        self,
        ship,
        delta: float,
        *,
        actor=None,
        source=None,
        metadata: dict[str, Any] | None = None,
    ):
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
            metadata={
                **self._source_metadata(source, _root_owner(source)),
                "source_credit": source_credit,
                **(metadata or {}),
            },
        )
        if delta < 0:
            self._record_enemy_death_reward_credit(
                ship,
                -float(delta),
                _source_enemy_death_reward_factor(source),
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
                metadata={
                    **self._source_metadata(source, _root_owner(source)),
                    "enemy_death_reward_credit": (
                        self._enemy_death_reward_credit_for_ship(ship)
                    )
                },
            )
            if ship is not self.trainee_ship:
                self.enemy_death_count += 1
        return event

    def _record_enemy_death_reward_credit(
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
        self._enemy_death_credited_crew_loss_totals[ship_id] = (
            self._enemy_death_credited_crew_loss_totals.get(ship_id, 0.0)
            + loss * source_credit
        )

    def _enemy_death_reward_credit_for_ship(self, ship) -> float:
        ship_id = id(ship)
        total_loss = self._crew_loss_totals.get(ship_id, 0.0)
        if total_loss <= 0.0:
            return 1.0
        credited_loss = self._enemy_death_credited_crew_loss_totals.get(
            ship_id,
            0.0,
        )
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
            metadata={
                **self._source_metadata(source, _root_owner(source)),
                "debuff_type": debuff_type,
            },
        )

    def _source_metadata(self, source, source_owner) -> dict[str, Any]:
        metadata = _removal_source_metadata(source, source_owner)
        credit = reward_credit_for(source)
        if credit is not None:
            metadata["reward_credit"] = credit
            metadata["source_origin_enemy_death_count"] = int(
                getattr(source, "_training_origin_enemy_death_count", 0)
            )
            metadata["effect_enemy_death_count"] = self.enemy_death_count
        stamp = spawn_stamp_for(source)
        if stamp is not None:
            metadata["source_spawn_stamp"] = stamp
        return metadata


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


_SPAWN_STAMP_ATTRIBUTE = "_training_spawn_stamp"


def spawn_stamp_for(obj) -> tuple[int, int] | None:
    stamp = getattr(obj, _SPAWN_STAMP_ATTRIBUTE, None)
    if (
        isinstance(stamp, tuple)
        and len(stamp) == 2
        and all(isinstance(value, int) for value in stamp)
    ):
        return stamp
    return None


def _bind_spawn_stamp(obj, stamp: tuple[int, int]) -> tuple[int, int] | None:
    if obj is None:
        return None
    try:
        setattr(obj, _SPAWN_STAMP_ATTRIBUTE, stamp)
    except Exception:
        return None
    return stamp


def bind_committed_action(ship, action_number: int, spawned_objects) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.bind_committed_action(ship, action_number, spawned_objects)


def inherit_credit(child, source) -> AbilityRewardCredit | None:
    credit = inherit_reward_credit(child, source)
    if credit is not None:
        try:
            child._training_origin_enemy_death_count = int(
                getattr(source, "_training_origin_enemy_death_count", 0)
            )
        except Exception:
            pass
    return credit


def bind_autonomous_fire(obj, root_parent) -> AbilityRewardCredit | None:
    ledger = ledger_for(root_parent)
    if ledger is None:
        return None
    return ledger.bind_autonomous_fire(obj, root_parent)


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


def record_action_released(ship, affected_objects) -> None:
    ledger = ledger_for(ship)
    if ledger is not None:
        ledger.record_action_released(ship, affected_objects)


def causal_credit_for_event(event: TrainingBattleEvent) -> AbilityRewardCredit | None:
    credit = event.metadata.get("reward_credit")
    if isinstance(credit, AbilityRewardCredit):
        return credit
    if event.event_type == EVENT_OBJECT_REMOVED:
        return reward_credit_for(event.metadata.get("source"))
    return reward_credit_for(event.obj)


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
    ledger = ledger_for(unit) or ledger_for(parent)
    if ledger is None:
        return
    ledger.record_crew_changed(
        parent,
        -abs(float(magnitude)),
        actor=actor,
        source=source if source is not None else unit,
        metadata={
            "launched_crew_loss": True,
            "launched_unit": unit,
            "launched_unit_ability_name": _ability_name(unit),
            "launched_unit_credit": reward_credit_for(unit),
            "launched_unit_spawn_stamp": spawn_stamp_for(unit),
        },
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


def damage_source_owner(source):
    """Return the player ship ultimately responsible for a damage source."""
    if source is None:
        return None
    current = source
    seen = set()
    while getattr(current, "parent", None) is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.parent
    if getattr(current, "player", None) is None:
        return None
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
        return const.PLANET_CREW_LOSS_REWARD_FACTOR
    source_name = _ability_name(source)
    if source_name == "DruugeA2":
        return const.DRUUGE_A2_CREW_LOSS_REWARD_FACTOR
    if source_name == "ShofixtiA2":
        return const.SHOFIXTI_A2_CREW_LOSS_REWARD_FACTOR
    return 1.0


def _source_enemy_death_reward_factor(source) -> float:
    if source is None:
        return 1.0
    role = getattr(getattr(source, "collision_capabilities", None), "role", None)
    if getattr(role, "name", None) == "PLANET":
        return const.PLANET_ENEMY_DEATH_REWARD_FACTOR
    source_name = _ability_name(source)
    if source_name == "DruugeA2":
        return const.DRUUGE_A2_ENEMY_DEATH_REWARD_FACTOR
    if source_name == "ShofixtiA2":
        return const.SHOFIXTI_A2_ENEMY_DEATH_REWARD_FACTOR
    return 1.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
