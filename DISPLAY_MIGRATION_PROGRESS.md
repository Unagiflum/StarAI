# Battle Display Migration Progress

## Phase 1: Extract a Side-Effect-Light Battle Controller

Status: Complete

Scope implemented:

- Added `BattleDrawLayout` for explicit arena and play HUD rectangles.
- Added `BattleDrawOptions` for draw-time options such as HUDs, instructions,
  pause overlay, entry trails, and interpolation.
- Added `BattleDrawController.draw()` as the shared no-flip battle drawing API.
- Kept normal play behavior behind `draw_battle()`, which now clears the screen,
  delegates to the controller, and calls `pygame.display.flip()` once.
- Preserved current play-mode HUD placement, viewport clipping, skipped star
  drawing inside HUD viewports, instruction hints, pause overlay, arena drawing,
  crosshair settings, and planet gravity marker settings.
- Left training display behavior unchanged; it still uses the existing
  `draw_battle_arena()` and training-specific HUD helpers.

Tests added or updated:

- Shared controller does not call `pygame.display.flip()`.
- `draw_battle()` wrapper calls `pygame.display.flip()` exactly once.
- Shared controller draws the supplied play arena and HUD rectangles.
- Existing play HUD layout expectations still pass.

Verification:

- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_star_field_renderer`
- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest discover tests`

Result: All verification commands passed.

## Next Phase

## Phase 2: Make HUD Rendering Rectangle-Based

Status: Complete

Scope implemented:

- Removed the remaining unused fixed `P1_X`/`P2_X` HUD placement constants from
  the battle renderer.
- Split shared play HUD drawing into a per-player helper that receives an
  explicit destination `pygame.Rect`.
- Preserved the existing play layout behavior by continuing to route
  `draw_battle()` through `create_play_battle_layout()`.
- Added clipping around each supplied HUD rectangle so arbitrary caller
  rectangles cannot be painted past by oversized HUD content.
- Kept the full shared HUD feature path together for live and placeholder HUDs:
  player color overlay, status bars, ship viewport, boarded marine icons,
  limpet count, special indicator, and dead/no-ship placeholders.
- Left training display behavior unchanged; training still uses its existing
  custom display-on HUDs until the later shared-renderer migration phase.

Tests added or updated:

- HUD panels render into arbitrary caller-supplied rectangles.
- HUD rendering is clipped to the supplied rect when the rect is shorter than
  the full HUD content.
- Live HUD features still render through the shared controller in an arbitrary
  HUD destination.
- Existing play-mode HUD output still appears in the expected screen regions.

Verification:

- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest discover tests`

Result: All verification commands passed.

## Phase 3: Adjust Training Layout for Full-Size HUDs

Status: Complete

Scope implemented:

- Derived the training HUD height from the shared battle HUD dimensions:
  marine icon region, 200px ship viewport, and HUD bottom padding.
- Moved training HUD rectangles flush with the screen bottom by making
  `HUD_TOP` depend on the shared full HUD height.
- Kept the existing display, start/stop, and back control rows in place because
  they already fit above the larger bottom HUD band with a small gap.
- Preserved the right-side arena/log rectangle used by display-on battle
  rendering and display-off batch logs.
- Left training display rendering behavior unchanged; training still uses its
  custom display-on battle/HUD path until Phase 4.

Tests added or updated:

- Training HUD rectangles contain the full shared HUD height and sit flush with
  the screen bottom.
- Training HUD rectangles are wide enough for the shared status bars and ship
  viewport.
- Bottom Display, Start/Stop, and Back controls do not overlap either HUD
  rectangle.
- Display-off log region remains the training arena rectangle and does not
  overlap HUD rectangles.

Verification:

- `.\.venv\Scripts\python.exe -m unittest tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_training_orchestration tests.test_training_session tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest discover tests`

Result: All verification commands passed.

## Phase 4: Migrate Training Display to the Shared Controller

Status: Complete

Scope implemented:

- Updated `BattleDrawController` so it can draw arena content into a caller
  supplied non-play arena rectangle while preserving the existing native
  `SCREEN_LEFT` assumptions in object draw methods.
- Replaced training display-on arena drawing with `BattleDrawController.draw()`
  using the training arena rectangle and `BattleDrawOptions(draw_huds=False)`.
- Replaced the active training display-on HUD path with
  `BattleDrawController.draw()` using the training HUD rectangles and
  `BattleDrawOptions(draw_arena=False)`.
- Kept training menu screen ownership unchanged: the menu still draws
  background, controls, modals, notices, and performs the single final
  `pygame.display.flip()`.
- Preserved display-off behavior; batch logs still draw into the training arena
  region without requiring a battle render.
- Left obsolete training-specific HUD helper functions in place for the Phase 6
  cleanup pass, but removed them from the active display-on path.

Tests added or updated:

- Shared controller draws arena content into a non-play arena rectangle.
- Training display-on arena rendering calls the shared controller with the
  training arena rectangle and does not flip.
- Training display-on HUD rendering calls the shared controller with both
  training HUD rectangles.
- Training display-on HUD rendering uses shared live HUD features.

Verification:

- `.\.venv\Scripts\python.exe -m unittest tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_training_orchestration tests.test_training_session tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest discover tests`

Result: All verification commands passed.

## Phase 5: Migrate Stars to Display Ownership

Status: Complete

Scope implemented:

- Added `DisplayStarField` as the display-owned star collection for shared
  battle rendering.
- Created display stars with an isolated deterministic RNG seed, separate from
  battle simulation RNG and training RNG.
- Updated the shared world renderer to draw stars from the display-owned star
  field instead of `RenderSnapshot.stars`.
- Updated normal play mode to create and pass a `DisplayStarField` to
  `draw_battle()`.
- Updated Training UI display mode to create and pass the same
  display-owned star-field type to the shared controller.
- Stopped `initialize_battle()` from inserting `Star` objects into the
  gameplay `World`; the legacy `include_stars` argument is retained as a
  compatibility no-op.
- Audited collision ownership by adding coverage that stars do not receive
  spatial collision categories.

Tests added or updated:

- Display-owned stars are deterministic and owned by `DisplayStarField`.
- Display star drawing uses the display-owned collection rather than world
  snapshot stars.
- Battle initialization keeps stars out of `World` objects.
- The legacy `include_stars=True` path no longer creates gameplay stars.
- Stars are not spatial collision candidates.

Verification:

- `.\.venv\Scripts\python.exe -m unittest tests.test_star_field_renderer tests.test_world tests.test_match_ui tests.test_train_ai_ui`
- `.\.venv\Scripts\python.exe -m unittest tests.test_training_orchestration tests.test_training_session tests.test_environment tests.test_training_observation tests.test_training_replay`
- `$env:PYTHONPATH='tests'; .\.venv\Scripts\python.exe -m unittest tests.test_collision_pipeline tests.test_collision_spatial_index tests.test_collision_spatial_pipeline tests.test_collision_contract tests.test_collision_dispatch`
- `.\.venv\Scripts\python.exe -m unittest discover tests`

Result: All verification commands passed. The initial collision command without
`PYTHONPATH=tests` failed because several collision tests import
`collision_test_support` as a top-level helper; rerunning with the helper path
set passed.

## Next Phase

Phase 6 should remove obsolete duplicated training drawing helpers and simplify
legacy wrappers now that play and training both use the shared controller and
display-owned stars.
