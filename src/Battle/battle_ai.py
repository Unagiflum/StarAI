"""Optional trained-AI runtime for battle mode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random
from typing import Any

import src.const as const
from src.toroidal import wrapped_delta
from src.training.contracts import action_for_index
from src.training.model_loader import (
    InferenceModelCache,
    load_inference_model,
)
from src.training.model_registry import (
    TrainingModelRepository,
    TrainingModelSlot,
    model_slot_has_checkpoint,
    model_basename,
)
from src.training.observation import encode_observation
from src.training.replay import select_action_epsilon_greedy


@dataclass(frozen=True)
class BattleAIModel:
    model: Any
    slot: TrainingModelSlot
    label: str


class ModelLoadFailure(RuntimeError):
    """Raised when a stored battle AI model cannot be loaded."""


def controls_for_action_index(action_index: int) -> dict[str, bool]:
    action = action_for_index(int(action_index))
    return {
        "forward": action.thrust,
        "left": action.turn_left,
        "right": action.turn_right,
        "action1": action.a1,
        "action2": action.a2,
    }


class TrainedModelController:
    def __init__(self, player: int, model):
        self.player = int(player)
        self.model = model

    def actions_for_frame(self, simulation) -> dict[str, bool]:
        self_ship, enemy_ship = _ships_for_player(simulation, self.player)
        observation = encode_observation(
            self_ship,
            enemy_ship,
            frame_id=getattr(simulation, "frame_id", None),
            game_objects=getattr(simulation, "world", None),
        )
        selection = select_action_epsilon_greedy(
            self.model,
            observation,
            epsilon=0.0,
        )
        return controls_for_action_index(selection.action_index)


class FallbackController:
    def __init__(self, player: int, *, rng=None):
        self.player = int(player)
        self.rng = rng or random
        self.action1_held = False
        self.action2_held = False

    def actions_for_frame(self, simulation) -> dict[str, bool]:
        self_ship, enemy_ship = _ships_for_player(simulation, self.player)
        left, right = _turn_toward_target(self_ship, enemy_ship)
        self._update_button_state("action1_held", 1.0 / const.FPS)
        self._update_button_state("action2_held", 1.0 / (2.0 * const.FPS))
        return {
            "forward": True,
            "left": left,
            "right": right,
            "action1": self.action1_held,
            "action2": self.action2_held,
        }

    def _update_button_state(self, attribute: str, probability: float) -> None:
        if self.rng.random() < probability:
            setattr(self, attribute, not getattr(self, attribute))


class BattleAIManager:
    def __init__(
        self,
        ai_enabled: Mapping[int, bool],
        repository: TrainingModelRepository | None = None,
        rng=None,
        model_cache: InferenceModelCache | None = None,
    ):
        self.ai_enabled = {
            1: bool(ai_enabled.get(1, ai_enabled.get("1", False))),
            2: bool(ai_enabled.get(2, ai_enabled.get("2", False))),
        }
        self.repository = repository or TrainingModelRepository(
            const.DEFAULT_MODELS_PATH,
            const.MODELS_PATH,
        )
        self.rng = rng or random
        self.model_cache = model_cache or InferenceModelCache()
        self._controllers: dict[int, Any] = {}
        self._labels: dict[int, str] = {}
        self.load_failures: dict[int, tuple[str, ...]] = {}

    def bind_round(self, simulation) -> None:
        self._controllers.clear()
        self._labels.clear()
        self.load_failures.clear()
        for player in (1, 2):
            if not self.is_ai_player(player):
                continue
            ship, _ = _ships_for_player(simulation, player)
            loaded, failures = self._resolve_model(str(getattr(ship, "name", "")))
            self.load_failures[player] = tuple(failures)
            if loaded is None:
                self._controllers[player] = FallbackController(player, rng=self.rng)
                self._labels[player] = "None found"
            else:
                self._controllers[player] = TrainedModelController(player, loaded.model)
                self._labels[player] = loaded.label

    def actions_for_frame(self, simulation) -> dict[int, dict[str, bool]]:
        actions = {}
        for player, controller in self._controllers.items():
            try:
                actions[player] = controller.actions_for_frame(simulation)
            except Exception:
                fallback = FallbackController(player, rng=self.rng)
                self._controllers[player] = fallback
                self._labels[player] = "None found"
                actions[player] = fallback.actions_for_frame(simulation)
        return actions

    def is_ai_player(self, player: int) -> bool:
        return bool(self.ai_enabled.get(int(player), False))

    def label_for_player(self, player: int) -> str | None:
        return self._labels.get(int(player)) if self.is_ai_player(int(player)) else None

    def _resolve_model(self, ship_name: str) -> tuple[BattleAIModel | None, list[str]]:
        slots = self.repository.slots_for_ship(ship_name)
        default_slots = [slot for slot in slots if _is_default_slot(slot)]
        remaining_slots = [slot for slot in slots if slot not in default_slots]
        failures = []
        for slot in (*default_slots, *remaining_slots):
            if not _slot_has_candidate_weights(slot):
                continue
            loaded = self.model_cache.entry_for(slot.ship, slot.slot)
            if loaded is None:
                error = self.model_cache.error_for(slot.ship, slot.slot)
                if error:
                    failures.append(f"{slot.ship}-{slot.slot:02d}: {error}")
                continue
            return BattleAIModel(
                model=loaded.model,
                slot=loaded.slot,
                label=model_basename(loaded.slot.ship, loaded.slot.slot),
            ), failures
        return None, failures


def load_battle_ai_model(slot: TrainingModelSlot) -> BattleAIModel:
    try:
        loaded = load_inference_model(slot)
    except Exception as exc:
        raise ModelLoadFailure(str(exc)) from exc
    return BattleAIModel(
        model=loaded.model,
        slot=slot,
        label=model_basename(slot.ship, slot.slot),
    )


def _is_default_slot(slot: TrainingModelSlot) -> bool:
    return bool(slot.is_bundled or slot.description.strip().lower() == "default")


def _slot_has_candidate_weights(slot: TrainingModelSlot) -> bool:
    return model_slot_has_checkpoint(slot)


def _ships_for_player(simulation, player: int):
    if int(player) == 1:
        return simulation.player1, simulation.player2
    return simulation.player2, simulation.player1


def _turn_toward_target(ship, target) -> tuple[bool, bool]:
    dx, dy = wrapped_delta(_position(ship), _position(target))
    if dx == 0 and dy == 0:
        return False, False
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
    rotation = float(getattr(ship, "rotation", 0.0)) % 360.0
    diff = (target_angle - rotation + 540.0) % 360.0 - 180.0
    if abs(diff) <= const.TURN_ANGLE / 2.0:
        return False, False
    return diff < 0.0, diff > 0.0


def _position(obj) -> tuple[float, float]:
    value = getattr(obj, "position", (0.0, 0.0))
    return float(value[0]), float(value[1])
