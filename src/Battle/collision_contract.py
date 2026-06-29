"""Shared contracts for collision dispatch and response handling."""

from collections.abc import Callable, MutableSequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class CollisionOutcome(Enum):
    """Explicit result of evaluating one ordered collision pair."""

    IGNORED = auto()
    RESOLVED = auto()
    CONSUMED_FIRST = auto()
    CONSUMED_SECOND = auto()
    CONSUMED_BOTH = auto()

    @property
    def handled(self) -> bool:
        return self is not CollisionOutcome.IGNORED

    @property
    def first_consumed(self) -> bool:
        return self in (
            CollisionOutcome.CONSUMED_FIRST,
            CollisionOutcome.CONSUMED_BOTH,
        )

    @property
    def second_consumed(self) -> bool:
        return self in (
            CollisionOutcome.CONSUMED_SECOND,
            CollisionOutcome.CONSUMED_BOTH,
        )

    def __bool__(self) -> bool:
        """Preserve truth checks while callers migrate from boolean results."""
        return self.handled

    def reversed(self) -> "CollisionOutcome":
        """Express this result after reversing the ordered collision pair."""
        if self is CollisionOutcome.CONSUMED_FIRST:
            return CollisionOutcome.CONSUMED_SECOND
        if self is CollisionOutcome.CONSUMED_SECOND:
            return CollisionOutcome.CONSUMED_FIRST
        return self


@dataclass(frozen=True)
class CollisionEnvironment:
    ships: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CollisionContext:
    """Services and world state shared while resolving collision pairs."""

    effects: MutableSequence[Any]
    environment: Any = field(default_factory=CollisionEnvironment)
    object_on_screen_policy: Callable[[Any, tuple[Any, ...]], bool] | None = None


def collision_context(
    context_or_effects,
    environment=None,
    *,
    object_on_screen_policy=None,
) -> CollisionContext:
    """Coerce the legacy effects/environment arguments during migration."""
    if isinstance(context_or_effects, CollisionContext):
        return context_or_effects
    return CollisionContext(
        effects=context_or_effects,
        environment=environment or CollisionEnvironment(),
        object_on_screen_policy=object_on_screen_policy,
    )
