"""Training-only causal provenance contracts for delayed rewards."""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


REWARD_MODE_LEGACY = "legacy"
REWARD_MODE_SHADOW = "shadow"
REWARD_MODE_CAUSAL = "causal"
REWARD_MODES = frozenset(
    {REWARD_MODE_LEGACY, REWARD_MODE_SHADOW, REWARD_MODE_CAUSAL}
)

ORIGIN_KIND_PRESS = "press"
ORIGIN_KIND_RELEASE = "release"
ORIGIN_KIND_LAUNCH = "launch"
ORIGIN_KIND_AUTONOMOUS_FIRE = "autonomous_fire"

_WEIGHT_TOLERANCE = 1e-6
_CREDIT_ATTRIBUTE = "_training_reward_credit"


@dataclass(frozen=True)
class RewardOrigin:
    """One weighted staged decision responsible for a future effect."""

    trajectory_id: str
    frame_index: int
    weight: float = 1.0
    kind: str = ORIGIN_KIND_LAUNCH

    def __post_init__(self) -> None:
        if not str(self.trajectory_id):
            raise ValueError("trajectory_id must not be empty")
        if int(self.frame_index) < 0:
            raise ValueError("frame_index cannot be negative")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("origin weight must be finite and non-negative")
        if not str(self.kind):
            raise ValueError("origin kind must not be empty")


@dataclass(frozen=True)
class AbilityRewardCredit:
    """Immutable causal provenance carried by a gameplay ability."""

    trajectory_id: str
    origins: tuple[RewardOrigin, ...]

    def __post_init__(self) -> None:
        if not str(self.trajectory_id):
            raise ValueError("trajectory_id must not be empty")
        origins = tuple(self.origins)
        if not origins:
            raise ValueError("ability reward credit requires at least one origin")
        if any(origin.trajectory_id != self.trajectory_id for origin in origins):
            raise ValueError("every origin must belong to the credit trajectory")
        total = sum(float(origin.weight) for origin in origins)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_WEIGHT_TOLERANCE):
            raise ValueError("origin weights must sum to 1.0")


@dataclass
class CausalRewardDiagnostics:
    """Counters collected without changing gameplay or legacy replay targets."""

    routed_events: Counter[tuple[str, str]] = field(default_factory=Counter)
    missing_provenance: Counter[str] = field(default_factory=Counter)
    cross_enemy_death_effects: Counter[str] = field(default_factory=Counter)
    closed_trajectory_rejections: Counter[str] = field(default_factory=Counter)
    launched_crew_loss_routes: Counter[str] = field(default_factory=Counter)
    peak_staged_frames: int = 0
    peak_staged_bytes: int = 0
    finalized_trajectory_lengths: list[int] = field(default_factory=list)
    finalization_seconds: list[float] = field(default_factory=list)
    shadow_comparison_count: int = 0
    shadow_comparisons: list[Any] = field(default_factory=list)
    last_shadow_comparison: Any | None = None


def new_trajectory_id() -> str:
    """Return an identifier safe across simulations, replacements, and workers."""

    return uuid.uuid4().hex


def full_weight_credit(
    trajectory_id: str,
    frame_index: int,
    *,
    kind: str = ORIGIN_KIND_LAUNCH,
) -> AbilityRewardCredit:
    origin = RewardOrigin(
        trajectory_id=str(trajectory_id),
        frame_index=int(frame_index),
        weight=1.0,
        kind=str(kind),
    )
    return AbilityRewardCredit(str(trajectory_id), (origin,))


def weighted_credit(
    trajectory_id: str,
    origins: Iterable[RewardOrigin],
) -> AbilityRewardCredit:
    return AbilityRewardCredit(str(trajectory_id), tuple(origins))


def reward_credit_for(obj: Any | None) -> AbilityRewardCredit | None:
    if obj is None:
        return None
    credit = getattr(obj, _CREDIT_ATTRIBUTE, None)
    return credit if isinstance(credit, AbilityRewardCredit) else None


def bind_reward_credit(
    obj: Any | None,
    credit: AbilityRewardCredit | None,
) -> AbilityRewardCredit | None:
    if obj is None or credit is None:
        return None
    try:
        setattr(obj, _CREDIT_ATTRIBUTE, credit)
    except Exception:
        return None
    return credit


def inherit_reward_credit(
    child: Any | None,
    source: Any | None,
) -> AbilityRewardCredit | None:
    """Copy immutable credit without relying on the child's gameplay parent."""

    return bind_reward_credit(child, reward_credit_for(source))


def replace_release_half(
    credit: AbilityRewardCredit,
    *,
    release_frame: int,
) -> AbilityRewardCredit:
    """Return future 50/50 press/latest-release credit for a live ability."""

    press_origin = next(
        (origin for origin in credit.origins if origin.kind == ORIGIN_KIND_PRESS),
        credit.origins[0],
    )
    return weighted_credit(
        credit.trajectory_id,
        (
            RewardOrigin(
                credit.trajectory_id,
                press_origin.frame_index,
                0.5,
                ORIGIN_KIND_PRESS,
            ),
            RewardOrigin(
                credit.trajectory_id,
                int(release_frame),
                0.5,
                ORIGIN_KIND_RELEASE,
            ),
        ),
    )
