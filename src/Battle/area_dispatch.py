"""Role-based policy dispatch for area-effect targets."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.collision_capabilities import CollisionRole


AreaEligibilityPolicy = Callable[[Any, Any], bool]
AreaImpactPolicy = Callable[[Any, Any, list, Any, float, float], float]


@dataclass(frozen=True)
class AreaTargetPolicy:
    is_eligible: AreaEligibilityPolicy
    apply_damage: AreaImpactPolicy


class AreaTargetRegistry:
    """Map target collision roles to area eligibility and damage policy."""

    def __init__(self):
        self._policies: dict[CollisionRole, AreaTargetPolicy] = {}

    def register(
        self,
        role: CollisionRole,
        *,
        is_eligible: AreaEligibilityPolicy,
        apply_damage: AreaImpactPolicy,
    ) -> None:
        if role in self._policies:
            raise ValueError(f"area target policy already registered for {role.name}")
        self._policies[role] = AreaTargetPolicy(is_eligible, apply_damage)

    def policy_for(self, target) -> AreaTargetPolicy | None:
        capabilities = getattr(target, "collision_capabilities", None)
        role = getattr(capabilities, "role", CollisionRole.NONE)
        return self._policies.get(role)

    def is_eligible(self, source, target) -> bool:
        policy = self.policy_for(target)
        return bool(policy and policy.is_eligible(source, target))

    def apply_damage(
        self,
        source,
        target,
        effects,
        delta,
        distance,
        damage,
    ) -> float:
        policy = self.policy_for(target)
        if policy is None:
            return 0
        return policy.apply_damage(
            source,
            target,
            effects,
            delta,
            distance,
            damage,
        )
