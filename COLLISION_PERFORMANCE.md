# Collision broad phase

`src/Battle/collision_spatial_index.py` implements the frame-local broad phase
used by `handle_collisions()`. The grid is a candidate generator only. Role
eligibility, swept sampling, pixel masks, and response handlers remain the
authoritative collision tests.

## Invariants

- The index is built after object updates and discarded after the collision
  frame. `World` remains the only authoritative ordered object store; no
  persistent secondary index needs synchronization.
- An object's cells cover the shortest toroidal sweep from its actual
  `previous_position` to `position`, expanded by its current logical dimensions
  and mask canvas. Cell size defaults to 128 arena pixels and is unrelated to
  `SPEED_LIMIT`.
- Seam-crossing paths are kept as short unwrapped intervals, then cell
  coordinates are reduced modulo the arena grid. A seam crossing therefore
  covers edge cells, not the whole arena.
- Queries deduplicate by object identity and return candidates by stable world
  index. Physical dispatch filters that result through the original phase group
  order, preserving phase order, unique pairs, stop-after-handled behavior, and
  relative second-object order.
- Physical responses may separate either participant. Both participants are
  reindexed after a handled response; position changes are also detected after
  ignored responses. If the outer participant changes cells, its unvisited
  suffix is queried again. Objects already passed in the original second list
  are never revisited. Responses that move an unrelated third object are not a
  supported response contract.
- Animated objects are indexed from current post-update masks and dimensions.
  Geometry values are cached only within the collision frame and invalidated
  after authoritative responses.

Laser queries traverse every wrapped collision segment with a 2-D DDA and
expand traversed cells by half the configured beam width. Target extent is
already represented by target membership. Exact `laser_hit_info()` still
determines pixel hits and hit distance.

Area emitters expose `maximum_area_damage_radius()` when they have a finite
bound. Radial abilities use their configured `range`; mask-shaped abilities use
their current mask radius. Emitters without a finite bound retain the full
ordered scan.

## Diagnostics and benchmark

`handle_collisions(..., broad_phase="brute_force")` selects the reference path.
`shadow_validate=True` checks omitted physical pairs, laser targets, and bounded
area targets against the exact brute-force geometry; it is disabled by default.

Run the collision-only benchmark with:

```text
python benchmark_collision.py --counts 50,150,300 --warmups 2 --iterations 7
```

It rebuilds every mutating scene outside the timed region, covers sparse and
dense projectile, special-object, laser, and area-damage workloads, and reports
possible pairs, spatial candidate counts, median time, and p95 time for both
paths.
