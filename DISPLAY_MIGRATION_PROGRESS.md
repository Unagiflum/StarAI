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

## Next Phase

Phase 3 should adjust the Training UI layout so display-on mode can fit
full-size shared HUD rectangles without overlapping bottom controls, while
preserving display-off log behavior.
