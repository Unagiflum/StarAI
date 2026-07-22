"""Training action adapters for the shared computer-control safety rules."""

from __future__ import annotations

from src.Battle.computer_control import guard_computer_controls
from src.training.contracts import ACTION_INDEX_TABLE, action_for_index
from src.training.replay_contracts import ActionSelection


_ACTION_INDEX_BY_MASK = {
    action.mask: index for index, action in enumerate(ACTION_INDEX_TABLE)
}


def guard_training_selection(selection: ActionSelection, ship, enemy) -> ActionSelection:
    """Return a selection whose recorded action matches the guarded controls."""

    action = action_for_index(int(selection.action_index))
    guarded = guard_computer_controls(action, ship, enemy)
    if guarded.mask == action.mask:
        return selection
    return ActionSelection(
        action_index=_ACTION_INDEX_BY_MASK[guarded.mask],
        exploratory=selection.exploratory,
        action_values=selection.action_values,
    )
