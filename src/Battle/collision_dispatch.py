"""Role-based registration and dispatch for collision pair handlers."""

from collections.abc import Callable
from typing import Any

from src.Battle.collision_contract import CollisionContext, CollisionOutcome
from src.collision_capabilities import CollisionRole


CollisionPairHandler = Callable[
    [Any, Any, CollisionContext],
    CollisionOutcome,
]


class CollisionPairRegistry:
    """Map ordered collision-role pairs to response handlers."""

    def __init__(self):
        self._handlers: dict[
            tuple[CollisionRole, CollisionRole],
            CollisionPairHandler,
        ] = {}

    def register(
        self,
        first_role: CollisionRole,
        second_role: CollisionRole,
        handler: CollisionPairHandler,
        *,
        bidirectional: bool = True,
    ) -> None:
        """Register a handler while preserving the pair's incoming object order."""
        keys = [(first_role, second_role)]
        if bidirectional and first_role is not second_role:
            keys.append((second_role, first_role))

        for registered_first, registered_second in keys:
            if (registered_first, registered_second) in self._handlers:
                raise ValueError(
                    f"collision handler already registered for "
                    f"{registered_first.name} x {registered_second.name}"
                )

        for key in keys:
            self._handlers[key] = handler

    def handler_for(
        self,
        first_role: CollisionRole,
        second_role: CollisionRole,
    ) -> CollisionPairHandler | None:
        return self._handlers.get((first_role, second_role))

    def dispatch(self, first, second, context: CollisionContext) -> CollisionOutcome:
        first_role = _collision_role(first)
        second_role = _collision_role(second)
        handler = self.handler_for(first_role, second_role)
        if handler is None:
            return CollisionOutcome.IGNORED
        return handler(first, second, context)


def _collision_role(obj) -> CollisionRole:
    capabilities = getattr(obj, "collision_capabilities", None)
    return getattr(capabilities, "role", CollisionRole.NONE)
