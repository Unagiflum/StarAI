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

Phase 2 should make HUD rendering rectangle-based for arbitrary caller-supplied
HUD rectangles while preserving existing play-mode output through the play
layout.
