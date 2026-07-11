"""Typed validation and commit values for ship actions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

ActionEffect = Callable[[], None]


class ActionOutput(Enum):
    """Shape retained by the temporary perform_action compatibility API."""

    NONE = "none"
    SINGLE = "single"
    MANY = "many"


@dataclass(frozen=True)
class ActionPlan:
    """A fully validated action whose mutations have not been committed."""

    action_number: int
    valid: bool
    spawned_objects: tuple[Any, ...] = ()
    energy_change: int = 0
    resets_energy_wait: bool = True
    crew_change: int = 0
    crew_change_source: Any | None = None
    cooldown_frames: int = 0
    cooldown_committed: bool = False
    side_effects: tuple[ActionEffect, ...] = ()
    launch_sound: Any | None = None
    use_first_object_sound: bool = True
    output: ActionOutput = ActionOutput.NONE

    @classmethod
    def invalid(cls, action_number: int) -> "ActionPlan":
        return cls(action_number=action_number, valid=False)


@dataclass(frozen=True)
class ActionResult:
    """The stable, ordered outcome of committing an ActionPlan."""

    valid: bool
    spawned_objects: tuple[Any, ...] = ()
    energy_change: int = 0
    crew_change: int = 0
    cooldown_action: int | None = None
    cooldown_frames: int = 0
    side_effects: tuple[ActionEffect, ...] = ()
    launch_sound_played: bool = False
    output: ActionOutput = ActionOutput.NONE

    @classmethod
    def invalid(cls) -> "ActionResult":
        return cls(valid=False)

    def compatibility_value(self):
        """Return the legacy object/list/None result expected by old callers."""
        if not self.valid or self.output is ActionOutput.NONE:
            return None
        if self.output is ActionOutput.SINGLE:
            return self.spawned_objects[0] if self.spawned_objects else None
        return list(self.spawned_objects)
