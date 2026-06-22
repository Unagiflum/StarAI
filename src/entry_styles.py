"""Data-only visual styles for battle entry trails."""

from dataclasses import dataclass

import src.const as const


@dataclass(frozen=True)
class EntryTrailStyle:
    angles: tuple[float, ...] = (0,)
    spacing: float = const.ENTRY_TRAIL_SPACING


STANDARD_ENTRY_TRAIL = EntryTrailStyle()
