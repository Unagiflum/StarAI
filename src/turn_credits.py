"""Integer credit accounting for quantized turning."""

import src.const as const


def initialize_turn_credits(owner, *, full):
    """Initialize bounded turn storage and its fixed-point spend cadence."""
    capacity = const.DIRECTIONS_MULTIPLIER
    cooldown = const.cooldown_frames(owner.turn_wait)
    owner.turn_credit_capacity = capacity
    owner.turn_credits = capacity if full else 0
    owner.turn_credit_progress = 0
    owner._turn_credit_cooldown = cooldown
    owner.turn_spend_progress = cooldown if full else 0


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
        # Match credit accrual's status/form-change semantics: discard the old
        # fractional interval, but keep an immediately usable stored credit
        # immediately usable at the new rate.
        owner.turn_spend_progress = cooldown if owner.turn_credits else 0

    # A fine turn costs ``cooldown`` units and each frame earns ``capacity``.
    # Stop accruing once at least one turn is ready so an idle ship cannot bank
    # a burst. The extra capacity-minus-one units retain the fractional
    # overflow that makes rates such as four turns per ten frames exact.
    if owner.turn_spend_progress < cooldown:
        owner.turn_spend_progress = min(
            cooldown + capacity - 1,
            owner.turn_spend_progress + capacity,
        )

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


def available_turn_credits(owner):
    """Return stored credits that may be spent at the configured cadence."""
    cooldown = owner._turn_credit_cooldown
    if cooldown is None or cooldown <= 0:
        return 0
    return min(owner.turn_credits, owner.turn_spend_progress // cooldown)


def spend_turn_credits(owner, requested):
    """Spend and return credits currently released by the turn cadence."""
    spent = min(max(0, int(requested)), available_turn_credits(owner))
    owner.turn_credits -= spent
    owner.turn_spend_progress -= spent * owner._turn_credit_cooldown
    return spent
