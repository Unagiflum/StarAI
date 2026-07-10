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

## Next Phase

Phase 4 should migrate training display-on mode to call the shared battle
drawing controller with the training arena and HUD rectangles, while preserving
training menu screen ownership and display-off log behavior.
