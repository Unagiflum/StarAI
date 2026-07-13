"""Frame-local toroidal broad phase for battle collisions.

The index is deliberately not authoritative.  It only rejects objects whose
radius-expanded swept bounds cannot share a grid cell.  Collision policies,
pixel masks, and swept narrow-phase sampling remain in ``collision_geometry``
and ``collision_responses``.

Ordering and mutation invariants
--------------------------------
* Entries carry their authoritative world index.  Queries deduplicate by
  identity and sort by that index; hash/set iteration never affects gameplay.
* Bounds use the shortest toroidal displacement from ``previous_position`` to
  ``position`` and the object's *current* dimensions and mask canvas.
* Seam crossings are represented as short unwrapped intervals and cell
  coordinates are reduced modulo the toroidal grid.  They do not turn into
  full-arena bounds.
* The owner creates a new index after object updates on every frame.  If a
  collision response moves an indexed object, ``update`` must be called before
  later queries that can observe that movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import src.const as const
from src.Battle.collision_geometry import (
    collision_size,
    get_collision_mask,
    sweep_previous_position,
)
from src.toroidal import wrapped_delta


# 128 pixels is representative of ships, asteroids, and larger projectiles.
# It is intentionally unrelated to SPEED_LIMIT.  The effective cell width is
# adjusted slightly so an integer number of cells exactly tiles the arena.
DEFAULT_CELL_SIZE = 128.0


@dataclass
class SpatialIndexMetrics:
    """Optional counters used by diagnostics and the collision benchmark."""

    object_count: int = 0
    occupied_cells: int = 0
    memberships: int = 0
    queries: int = 0
    returned_candidates: int = 0


@dataclass
class _Entry:
    obj: Any
    order: int
    categories: frozenset[str]
    cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)


class ToroidalSpatialIndex:
    """Uniform grid containing current, swept object bounds."""

    def __init__(
        self,
        objects: Iterable[Any] = (),
        *,
        categories: Mapping[int, Iterable[str]] | None = None,
        cell_size: float = DEFAULT_CELL_SIZE,
        arena_size: float = const.ARENA_SIZE,
        metrics: SpatialIndexMetrics | None = None,
    ):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if arena_size <= 0:
            raise ValueError("arena_size must be positive")

        self.arena_size = float(arena_size)
        self.cells_per_axis = max(1, int(math.ceil(arena_size / cell_size)))
        self.cell_size = self.arena_size / self.cells_per_axis
        self.metrics = metrics if metrics is not None else SpatialIndexMetrics()
        self._metrics_enabled = metrics is not None
        self._cells: dict[tuple[int, int], list[_Entry]] = {}
        self._entries: dict[int, _Entry] = {}
        self._membership_count = 0

        category_map = categories or {}
        for order, obj in enumerate(objects):
            if categories is not None and id(obj) not in category_map:
                continue
            if not _has_position(obj):
                continue
            if id(obj) in self._entries:
                # One membership is sufficient; ordered phase lists still
                # preserve duplicate references if a caller supplied them.
                continue
            self.insert(obj, order, category_map.get(id(obj), ()))

    def __len__(self) -> int:
        return len(self._entries)

    def insert(self, obj: Any, order: int, categories: Iterable[str] = ()) -> None:
        """Insert one object using its current geometry and swept path."""
        identity = id(obj)
        if identity in self._entries:
            raise ValueError("object is already indexed")

        entry = _Entry(obj, order, frozenset(categories))
        entry.cells = self._object_cells(obj)
        self._entries[identity] = entry
        self._membership_count += len(entry.cells)
        for cell in entry.cells:
            self._cells.setdefault(cell, []).append(entry)
        self._refresh_metrics()

    def update(self, obj: Any) -> bool:
        """Refresh one object's membership after a response moves or reshapes it."""
        entry = self._entries.get(id(obj))
        if entry is None:
            return False

        new_cells = self._object_cells(obj)
        if new_cells == entry.cells:
            return False

        old_cells = entry.cells
        for cell in old_cells:
            members = self._cells[cell]
            members.remove(entry)
            if not members:
                del self._cells[cell]

        self._membership_count += len(new_cells) - len(old_cells)
        entry.cells = new_cells
        for cell in new_cells:
            members = self._cells.setdefault(cell, [])
            # Updates can add an earlier world object to a populated cell.
            # Maintain world order locally so even unsorted diagnostic reads
            # remain deterministic.
            insertion_index = len(members)
            while insertion_index and members[insertion_index - 1].order > entry.order:
                insertion_index -= 1
            members.insert(insertion_index, entry)
        self._refresh_metrics()
        return True

    def contains(self, obj: Any) -> bool:
        return id(obj) in self._entries

    def candidates_for(
        self,
        obj: Any,
        *,
        categories: Iterable[str] | None = None,
        include_self: bool = False,
    ) -> list[Any]:
        """Return co-cell candidates in authoritative world order."""
        entry = self._entries.get(id(obj))
        if entry is None:
            return []
        return self._objects_in_cells(
            entry.cells,
            categories=categories,
            excluded_identity=None if include_self else id(obj),
        )

    def query_radius(
        self,
        position,
        query_radius: float,
        *,
        categories: Iterable[str] | None = None,
    ) -> list[Any]:
        """Conservatively query a toroidal circle through its bounding square.

        Indexed targets already include their own current extent, so callers
        pass only the effect radius.  The narrow phase still checks distance or
        masks exactly.
        """
        if not math.isfinite(query_radius) or query_radius < 0:
            raise ValueError("query_radius must be finite and non-negative")
        x, y = position
        cells = self._cells_for_bounds(
            x - query_radius,
            x + query_radius,
            y - query_radius,
            y + query_radius,
        )
        return self._objects_in_cells(cells, categories=categories)

    def query_segments(
        self,
        segments,
        *,
        width: float = 1.0,
        categories: Iterable[str] | None = None,
    ) -> list[Any]:
        """Return candidates along wrapped line segments using a 2-D DDA.

        The traversed cells are expanded for half the beam width.  Target
        extent is accounted for by target insertion, avoiding a global
        maximum-target-size expansion.
        """
        if not math.isfinite(width) or width < 0:
            raise ValueError("width must be finite and non-negative")

        cells: set[tuple[int, int]] = set()
        expansion = int(math.ceil((width / 2.0) / self.cell_size))
        for start, end in segments:
            delta = wrapped_delta(start, end)
            wrapped_end = (start[0] + delta[0], start[1] + delta[1])
            for cell_x, cell_y in self._segment_cells(start, wrapped_end):
                for offset_x in range(-expansion, expansion + 1):
                    for offset_y in range(-expansion, expansion + 1):
                        cells.add(
                            (
                                (cell_x + offset_x) % self.cells_per_axis,
                                (cell_y + offset_y) % self.cells_per_axis,
                            )
                        )
        return self._objects_in_cells(cells, categories=categories)

    def shares_cell(self, first: Any, second: Any) -> bool:
        """Diagnostic predicate used by differential/shadow validation."""
        first_entry = self._entries.get(id(first))
        second_entry = self._entries.get(id(second))
        if first_entry is None or second_entry is None:
            return False
        second_cells = set(second_entry.cells)
        return any(cell in second_cells for cell in first_entry.cells)

    def _object_cells(self, obj: Any) -> tuple[tuple[int, int], ...]:
        previous = sweep_previous_position(obj)
        current = obj.position
        delta = wrapped_delta(previous, current)
        end_x = previous[0] + delta[0]
        end_y = previous[1] + delta[1]
        extent_x, extent_y = collision_extents(obj)
        return tuple(
            self._cells_for_bounds(
                min(previous[0], end_x) - extent_x,
                max(previous[0], end_x) + extent_x,
                min(previous[1], end_y) - extent_y,
                max(previous[1], end_y) + extent_y,
            )
        )

    def _cells_for_bounds(self, left, right, top, bottom):
        x_cells = self._axis_cells(left, right)
        y_cells = self._axis_cells(top, bottom)
        return tuple((x, y) for x in x_cells for y in y_cells)

    def _axis_cells(self, low: float, high: float) -> tuple[int, ...]:
        if high < low:
            low, high = high, low
        if high - low >= self.arena_size:
            return tuple(range(self.cells_per_axis))

        first = math.floor(low / self.cell_size)
        last = math.floor(high / self.cell_size)
        # A short seam-crossing interval can map both ends to wrapped cells.
        # dict.fromkeys preserves traversal order while removing any duplicate
        # caused by modulo reduction.
        return tuple(
            dict.fromkeys(
                index % self.cells_per_axis
                for index in range(first, last + 1)
            )
        )

    def _segment_cells(self, start, end):
        """Yield all grid cells crossed by one unwrapped segment."""
        x0, y0 = start
        x1, y1 = end
        cell_x = math.floor(x0 / self.cell_size)
        cell_y = math.floor(y0 / self.cell_size)
        end_cell_x = math.floor(x1 / self.cell_size)
        end_cell_y = math.floor(y1 / self.cell_size)

        yield cell_x % self.cells_per_axis, cell_y % self.cells_per_axis
        if cell_x == end_cell_x and cell_y == end_cell_y:
            return

        dx = x1 - x0
        dy = y1 - y0
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        infinity = float("inf")

        if step_x:
            next_x = (cell_x + (1 if step_x > 0 else 0)) * self.cell_size
            t_max_x = (next_x - x0) / dx
            t_delta_x = self.cell_size / abs(dx)
        else:
            t_max_x = t_delta_x = infinity

        if step_y:
            next_y = (cell_y + (1 if step_y > 0 else 0)) * self.cell_size
            t_max_y = (next_y - y0) / dy
            t_delta_y = self.cell_size / abs(dy)
        else:
            t_max_y = t_delta_y = infinity

        while cell_x != end_cell_x or cell_y != end_cell_y:
            if t_max_x < t_max_y:
                cell_x += step_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_x:
                cell_y += step_y
                t_max_y += t_delta_y
            else:
                # At a grid corner the mathematical segment touches both side
                # cells.  Yield them as well as the diagonal cell so zero-width
                # and boundary-aligned queries remain conservative.
                if step_x:
                    yield (
                        (cell_x + step_x) % self.cells_per_axis,
                        cell_y % self.cells_per_axis,
                    )
                if step_y:
                    yield (
                        cell_x % self.cells_per_axis,
                        (cell_y + step_y) % self.cells_per_axis,
                    )
                cell_x += step_x
                cell_y += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            yield cell_x % self.cells_per_axis, cell_y % self.cells_per_axis

    def _objects_in_cells(
        self,
        cells,
        *,
        categories: Iterable[str] | None,
        excluded_identity: int | None = None,
    ) -> list[Any]:
        if self._metrics_enabled:
            self.metrics.queries += 1
        category_filter = frozenset(categories or ())
        found: dict[int, _Entry] = {}
        for cell in cells:
            for entry in self._cells.get(cell, ()):
                identity = id(entry.obj)
                if identity == excluded_identity:
                    continue
                if category_filter and entry.categories.isdisjoint(category_filter):
                    continue
                found[identity] = entry

        ordered = sorted(found.values(), key=lambda entry: entry.order)
        if self._metrics_enabled:
            self.metrics.returned_candidates += len(ordered)
        return [entry.obj for entry in ordered]

    def _refresh_metrics(self) -> None:
        if not self._metrics_enabled:
            return
        self.metrics.object_count = len(self._entries)
        self.metrics.occupied_cells = len(self._cells)
        self.metrics.memberships = self._membership_count


def collision_extents(obj: Any) -> tuple[float, float]:
    """Return conservative current half-extents for spatial insertion."""
    # Lightweight collision doubles and non-colliding area emitters can be
    # only partially initialized.  Index construction must not demand more
    # geometry than the legacy full scan did.
    try:
        size = collision_size(obj)
    except (AttributeError, IndexError, TypeError):
        size = getattr(obj, "size", (0.0, 0.0)) or (0.0, 0.0)
    width = float(size[0]) if len(size) > 0 else 0.0
    height = float(size[1]) if len(size) > 1 else width

    try:
        mask = get_collision_mask(obj)
    except (AttributeError, IndexError, TypeError):
        mask = None
    if mask is not None:
        mask_width, mask_height = mask.get_size()
        width = max(width, float(mask_width))
        height = max(height, float(mask_height))

    circular_radius = max(width, height) / 2.0
    return max(width / 2.0, circular_radius), max(height / 2.0, circular_radius)


def _has_position(obj: Any) -> bool:
    position = getattr(obj, "position", None)
    return position is not None and len(position) >= 2
