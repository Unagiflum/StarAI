"""Role-based policy dispatch for laser targets."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.collision_capabilities import CollisionRole


LaserEligibilityPolicy = Callable[[Any, Any, bool], bool]
LaserImpactPolicy = Callable[..., None]


@dataclass(frozen=True)
class LaserTargetPolicy:
    is_eligible: LaserEligibilityPolicy
    apply_impact: LaserImpactPolicy


class LaserTargetRegistry:
    """Map target collision roles to laser eligibility and impact policy."""

    def __init__(self):
        self._policies: dict[CollisionRole, LaserTargetPolicy] = {}

    def register(
        self,
        role: CollisionRole,
        *,
        is_eligible: LaserEligibilityPolicy,
        apply_impact: LaserImpactPolicy,
    ) -> None:
        if role in self._policies:
            raise ValueError(f"laser target policy already registered for {role.name}")
        self._policies[role] = LaserTargetPolicy(is_eligible, apply_impact)

    def policy_for(self, target) -> LaserTargetPolicy | None:
        capabilities = getattr(target, "collision_capabilities", None)
        role = getattr(capabilities, "role", CollisionRole.NONE)
        return self._policies.get(role)

    def is_eligible(self, laser, target, *, explicit: bool = False) -> bool:
        policy = self.policy_for(target)
        return bool(policy and policy.is_eligible(laser, target, explicit))

    def apply_impact(
        self,
        target,
        effects,
        normal,
        damage,
        contact,
        *,
        source=None,
    ) -> None:
        policy = self.policy_for(target)
        if policy is None:
            return
        if source is None:
            policy.apply_impact(target, effects, normal, damage, contact)
        else:
            policy.apply_impact(
                target,
                effects,
                normal,
                damage,
                contact,
                source=source,
            )
