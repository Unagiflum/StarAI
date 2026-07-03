"""Integer credit accounting for quantized turning."""

import src.const as const


def initialize_turn_credits(owner, *, full):
    """Initialize a bounded turn-credit bucket on ``owner``."""
    capacity = const.DIRECTIONS_MULTIPLIER
    owner.turn_credit_capacity = capacity
    owner.turn_credits = capacity if full else 0
    owner.turn_credit_progress = 0
    owner._turn_credit_cooldown = None


def accrue_turn_credits(owner, turn_wait):
    """Accrue credits at ``capacity / (turn_wait + 1)`` per physics frame.

    Fractional progress is retained between spends. If a status or form change
    alters the cooldown, the old partial interval is discarded rather than
    being retroactively valued at the new rate.
    """
    capacity = owner.turn_credit_capacity
    cooldown = const.cooldown_frames(turn_wait)

    if owner._turn_credit_cooldown != cooldown:
        owner.turn_credit_progress = 0
        owner._turn_credit_cooldown = cooldown

    if owner.turn_credits >= capacity:
        owner.turn_credit_progress = 0
        return 0

    owner.turn_credit_progress += capacity
    earned, owner.turn_credit_progress = divmod(
        owner.turn_credit_progress, cooldown
    )
    if earned <= 0:
        return 0

    previous = owner.turn_credits
    owner.turn_credits = min(capacity, previous + earned)
    if owner.turn_credits == capacity:
        # A full token bucket cannot bank hidden fractional credit.
        owner.turn_credit_progress = 0
    return owner.turn_credits - previous


def spend_turn_credits(owner, requested):
    """Spend and return up to ``requested`` available credits."""
    spent = min(max(0, int(requested)), owner.turn_credits)
    owner.turn_credits -= spent
    return spent
