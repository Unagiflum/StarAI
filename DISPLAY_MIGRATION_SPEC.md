# Battle Display Migration Specification

## Prompt for a new implementation chat

Implement the battle display migration described in this document. Work one
phase at a time, beginning with the earliest incomplete phase. Before changing
code, inspect the current repository because the battle renderer, Training UI,
display settings, and tests may have changed. Preserve unrelated work and
existing game behavior.

For each phase:

1. Restate the phase scope and assumptions.
2. Add or update tests alongside the implementation.
3. Run focused tests, then the broader relevant test suite.
4. Stop after the phase is complete and report changed files, verification,
   remaining limitations, and the next phase.

Do not silently change battle simulation timing, training semantics, model
metadata, observation schemas, action ordering, reward definitions, or display
settings persistence. If a rendering detail makes a specified feature
impossible or ambiguous, report it before selecting a materially different
behavior.

## 1. Goal

Migrate normal play mode and AI training display mode to use one shared battle
drawing controller.

Both modes must show the same battle presentation features:

- The same display-owned star field.
- Player color indicators from display settings.
- Battle arena/world drawing.
- Ship crosshairs when enabled by display settings.
- Planet gravity markers when enabled by display settings.
- Same player HUD styling.
- Same status bars.
- Same ship viewports.
- Same boarded marine icons.
- Same limpet counters.
- Same special indicators.
- Same relevant battle effects and ability layers.

The only intended differences between normal play mode and training mode are:

- The surrounding UI controls are different.
- The battle arena rectangle is located in a different place.
- The player HUD rectangles are located in different places.
- Normal play may draw battle-only instructions and pause overlays.
- Training may draw training-only controls, tabs, logs, modals, and notices
  outside the shared battle rectangles.

## 2. Non-Negotiable Invariants

### 2.1 Screen ownership

Shared battle drawing code must not own the window.

The shared drawing controller must not:

- Call `pygame.display.flip()`.
- Call `pygame.display.update()`.
- Tick a `pygame.time.Clock`.
- Sleep.
- Poll or mutate pygame events.
- Step simulation.
- Clear the entire screen unless the caller explicitly passes the entire screen
  as a draw rectangle.

Normal play mode owns its loop, timing, event handling, full-screen clear, and
display flip.

Training mode owns its menu loop, training controls, timing around display
presentation, full-screen/menu clear, and display flip.

### 2.2 Training semantics

Training display changes must remain presentation-only.

Display on/off must not change:

- Training world setup.
- Simulation frame count.
- RNG consumption used by battle simulation or training.
- Replay contents.
- Reward returns.
- Model updates.
- Stop behavior.

Stars must be migrated out of battle simulation ownership and into display
ownership. The shared star field must be presentation state only: it may use
world-space coordinates and camera/midpoint information for parallax, but it
must not be stored in or updated by the gameplay `World`.

Do not add real `Star` objects to the training `BattleSimulation` world merely
to make training display stars. Do not keep normal battle dependent on
simulation-owned stars after the star-field migration phase is complete.

### 2.3 Compatibility during migration

Keep existing public entry points working until their callers are migrated.
It is acceptable for old functions such as `draw_battle()` and
`draw_battle_arena()` to become wrappers around the new controller during the
migration.

### 2.4 Visual feature parity

When training display is on, training must use the same battle rendering
features as play mode unless explicitly disabled by the same display setting
that disables them in play mode.

Training should not keep a separate simplified HUD once the shared HUD renderer
supports training layout rectangles.

### 2.5 Star field ownership

The star field is a display resource, not a battle object.

The migrated star field must:

- Be created at startup, renderer initialization, or another display/asset
  initialization boundary.
- Use an isolated deterministic seed or isolated RNG that is unrelated to
  simulation RNG and training decision RNG.
- Be present equally in normal battle and training display.
- Receive camera/world-view information from the drawing controller so it can
  keep the same parallax behavior as the current renderer.
- Stay out of `World`, collision snapshots, observations, rewards, event
  ledgers, and training replay data.

### 2.6 Collision system boundary

The star-field migration should not add new collision behavior.

Stars are visual background presentation and must not become collision
candidates. If any collision code, spatial index code, world snapshot code, or
typed-world cache currently references stars because they are in `World`, that
dependency must be audited during the star migration. Rendering dependencies
should move to the display-owned star field; collision and simulation code
should either ignore stars or no longer see them at all.

## 3. Target Architecture

### 3.1 Shared layout contract

Introduce a shared layout value object, for example:

```python
@dataclass(frozen=True)
class BattleDrawLayout:
    arena_rect: pygame.Rect
    player1_hud_rect: pygame.Rect | None
    player2_hud_rect: pygame.Rect | None
```

The exact field names may differ if the codebase already has a better local
pattern. The required idea is that battle drawing receives explicit rectangles
for the arena and HUD panels.

Normal play mode should provide the existing center arena rect and the existing
left/right HUD panel regions.

Training mode should provide the training arena rect and full-size HUD panel
rects. The training controls may be adjusted to make room for those full-size
HUDs.

### 3.2 Shared draw options

Introduce a shared options value object, for example:

```python
@dataclass(frozen=True)
class BattleDrawOptions:
    draw_arena: bool = True
    draw_huds: bool = True
    draw_instructions: bool = False
    is_paused: bool = False
    show_entry_trails: bool = True
    interp_t: float = 0.0
```

The exact fields may differ, but options must keep mode-specific concerns out
of the renderer where possible.

### 3.3 Shared battle view contract

The controller may accept the existing world/simulation objects directly or a
snapshot dictionary during early phases. By the end of the migration, both play
and training should feed equivalent battle-view data into the same controller:

- Game objects or render snapshot.
- Border color.
- Camera targets.
- Entry state, if any.
- Frame ID.
- Original ships.

Stars should not be supplied as battle-view game objects after the star-field
migration phase. The drawing controller should obtain the shared display-owned
star field from renderer/display resources and draw it using the current
world-camera parameters.

### 3.4 Shared display star field

Introduce or adapt a display-owned star-field object, for example:

```python
class DisplayStarField:
    def draw(self, surface, scale_factor, translation, midpoint):
        ...
```

The exact class shape may differ. The required behavior is:

- It is created outside battle simulation setup.
- It owns the star positions, depths, and images needed for rendering.
- It uses world-space positions so camera panning, wrapping, and parallax work
  like the current battle star rendering.
- It does not insert stars into the `World`.
- It is shared by both normal play and training display through the same battle
  drawing controller.
- It may reuse existing star assets and `StarFieldRenderer` logic if that keeps
  the implementation smaller and behaviorally equivalent.

### 3.5 Shared drawing controller

Introduce a controller or cohesive helper API, for example:

```python
class BattleDrawController:
    def draw(self, surface, battle_view, layout, options):
        ...
```

The exact class/function shape may differ. The required behavior is:

- Draw arena/world content into `layout.arena_rect`.
- Draw the display-owned star field before gameplay objects in both play and
  training display modes.
- Draw player HUDs into supplied HUD rectangles.
- Use the same status bars and HUD feature drawing for both modes.
- Respect display settings through the same constants/config used by play mode.
- Preserve the no-flip/no-timing/no-simulation invariant.

### 3.6 Top-level wrappers

Normal play mode should retain a top-level function that does play-mode
composition:

1. Clear the full screen.
2. Call the shared controller with the play layout.
3. Draw play-only instructions/pause overlay if still owned outside the
   controller.
4. Flip the display once.

Training mode should retain its menu loop composition:

1. Draw menu background and training controls.
2. Call the shared controller with the training layout when display is on.
3. Draw display-off logs when display is off.
4. Draw training-only modals/notices.
5. Flip the display once.

## 4. Phase Plan

## Phase 1: Extract a Side-Effect-Light Battle Controller

### Scope

Extract the existing arena/world rendering and current play-mode HUD rendering
behind a shared controller API without changing normal play visuals.

Keep existing training display behavior unchanged in this phase.

### Required implementation details

- Add a layout contract for the current play arena and HUD regions.
- Add a controller or shared function that can draw:
  - The existing star field behavior.
  - Arena/world.
  - Current play HUDs.
  - Optional pause/instruction overlays if that is the least disruptive first
    extraction.
- Ensure the shared controller does not call `pygame.display.flip()`.
- Ensure `draw_battle()` remains callable and keeps the current play-mode
  behavior by wrapping the new controller and then flipping once.
- Preserve existing viewport rendering behavior, including clipped player
  viewports and no star drawing inside HUD viewports.

### Tests

Add or update tests proving:

- The shared controller does not call `pygame.display.flip()`.
- `draw_battle()` still calls `pygame.display.flip()` exactly as before.
- The shared controller can draw the play arena and HUDs into supplied rects.
- Existing match UI and battle draw tests still pass.

### Acceptance criteria

- Normal battle rendering is visually and behaviorally unchanged.
- There is a shared controller API ready for training to call later.
- No training UI behavior changes yet.

## Phase 2: Make HUD Rendering Rectangle-Based

### Scope

Remove hard-coded assumptions that player HUDs must live only in the normal
left/right battle panels. The shared HUD renderer must draw the full play-mode
HUD feature set into caller-supplied rectangles.

### Required implementation details

- Refactor HUD panel drawing so each player HUD receives an explicit
  destination rect.
- Preserve existing play-mode HUD dimensions and alignment when using the play
  layout.
- The HUD renderer must support full-size training HUDs at training-provided
  locations.
- Keep the same:
  - Player color overlay and border.
  - Status bars.
  - Ship viewport.
  - Boarded marine icons.
  - Limpet count.
  - Special indicator.
  - Dead/no-ship placeholders.
- Avoid separate training-only versions of these elements once the shared HUD
  renderer is capable of drawing them.

### Tests

Add or update tests proving:

- HUD panels render into arbitrary caller-supplied rects.
- Rendering one HUD does not paint outside its rect except where explicitly
  expected by legacy play-mode behavior.
- Existing play-mode HUD output still appears in the expected screen regions.
- Dead/no-ship HUD placeholders still render.

### Acceptance criteria

- Play mode continues to use the shared controller through the play layout.
- HUD drawing is no longer tied to fixed module-level `P1_X`/`P2_X` placement
  for all uses.
- Training can request full-size HUDs in a later phase without custom HUD code.

## Phase 3: Adjust Training Layout for Full-Size HUDs

### Scope

Adjust the Training UI layout so display-on mode can fit full-size shared HUDs.
Do not switch training to the shared HUD renderer yet unless Phase 2 made that
trivial and tests remain focused.

### Required implementation details

- Move training HUD rectangles flush with the screen bottom.
- Adjust the Play/Start, Stop, Back, Display, or related bottom controls only
  as much as needed to make the larger HUDs fit cleanly.
- Preserve ergonomic access to training controls.
- Preserve display-off batch log behavior.
- Ensure text remains inside buttons and panels at the current window size.

### Tests

Add or update tests proving:

- Training layout HUD rectangles can contain the full shared HUD size.
- Training bottom controls do not overlap the HUD rectangles.
- Display-off log still uses the arena/log region correctly.
- Existing training UI layout tests still pass or are updated to the new
  expected geometry.

### Acceptance criteria

- Training mode has enough space for full shared HUDs.
- No shared renderer migration is required to complete this phase.
- Training display remains stable in display-on and display-off modes.

## Phase 4: Migrate Training Display to the Shared Controller

### Scope

Replace training's custom display-on arena and HUD drawing with the shared
battle drawing controller.

### Required implementation details

- Training display-on mode must call the shared controller with:
  - Training arena rect.
  - Training player 1 HUD rect.
  - Training player 2 HUD rect.
  - Current battle view snapshot.
- Remove or stop using simplified training HUD drawing functions once shared
  HUD rendering is active.
- Preserve training menu screen ownership:
  - Training menu still draws background/controls.
  - Training menu still calls `pygame.display.flip()` once.
  - Shared controller must not flip.
- Preserve display-off behavior:
  - Display-off still shows the batch log and diagnostics instead of the
    battle view.
- Preserve training worker/session ownership of simulation.

### Tests

Add or update tests proving:

- Training display-on calls the shared controller.
- Training display-on does not call `pygame.display.flip()` from the shared
  controller.
- Training display-on draws both HUDs with shared HUD features.
- Training display-off does not require a battle render.
- Display-on/off training semantic equivalence tests still pass.

### Acceptance criteria

- Training and play both use the same battle drawing controller.
- Training no longer has a separate simplified HUD implementation in the active
  display-on path.
- Flicker-prone full-screen clearing/flipping is not introduced.

## Phase 5: Migrate Stars to Display Ownership

### Scope

Move stars out of battle `World` ownership and into a shared display-owned star
field used equally by normal play and training display.

### Required implementation details

- Create the shared star field at startup, renderer initialization, or another
  display/asset initialization boundary.
- Use an isolated deterministic seed or isolated RNG that cannot consume or
  perturb battle simulation RNG or training RNG.
- Use the same display-owned star field in both normal play and training
  display mode.
- The star field must receive world camera information from the drawing
  controller, such as scale factor, translation, and midpoint, so it can
  preserve current parallax behavior.
- Remove or bypass per-battle star insertion into the gameplay `World` once the
  display-owned star field is active.
- Move any code that depends on `World.stars` for rendering to the renderer or
  display-owned star field path.
- Audit collision snapshots, spatial indexes, typed world caches, and broad
  world-object loops for star assumptions. The expected result is removal of
  render-only star coupling, not new star collision handling.
- Keep stars out of collision, observations, rewards, event ledgers, and replay
  data.
- Prefer reusing existing star assets and `StarFieldRenderer` behavior.
- Make sure display settings that affect star/player presentation remain
  respected consistently with play mode.

### Tests

Add or update tests proving:

- Training battle simulation world omits real stars.
- Normal battle simulation world also no longer needs real stars after this
  phase.
- Display-on/off training results remain identical.
- The shared renderer draws the same display-owned star field in normal play
  and training display mode.
- Star-field creation does not consume the RNG object passed to battle
  initialization or training orchestration.
- Any code previously relying on `World.stars` for presentation is migrated to
  renderer/display ownership.
- Collision snapshots and spatial indexes do not include stars.
- Removing stars from `World` does not change focused collision behavior tests.

### Acceptance criteria

- Normal play and training display visually include the same display-owned star
  field.
- Stars are no longer simulation-owned battle objects.
- Collision code has no render-only dependency on `World.stars`.
- Training semantics remain unchanged.
- Star rendering uses the same feature path in both modes.

## Phase 6: Clean Up Legacy Drawing Paths

### Scope

Remove obsolete duplicated training drawing code and simplify compatibility
wrappers after both modes use the shared controller.

### Required implementation details

- Delete or deprecate unused training-specific HUD draw helpers.
- Keep any public wrapper functions that are still useful, but make them thin
  and documented.
- Remove duplicated player color, status bar, viewport, and placeholder logic
  from training code.
- Confirm naming and module boundaries are clear:
  - Battle rendering code lives under `src/Battle`.
  - Training UI code owns training controls and layout only.

### Tests

Run:

- Focused battle draw tests.
- Focused training UI tests.
- Focused training session/orchestration tests related to display equivalence.
- Full relevant test suite.

### Acceptance criteria

- There is one active battle drawing controller for play and training.
- Training UI contains no active duplicate HUD implementation.
- Normal play and training display show the same battle features.
- All relevant tests pass.

## 5. Suggested Focused Test Commands

Use the commands that match the current test layout. As of this spec, likely
focused commands include:

```powershell
python -m unittest tests.test_match_ui tests.test_train_ai_ui
python -m unittest tests.test_training_orchestration tests.test_training_session tests.test_train_ai_ui
python -m unittest discover tests
```

If the active environment requires `.venv`, use:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_match_ui tests.test_train_ai_ui
.\.venv\Scripts\python.exe -m unittest tests.test_training_orchestration tests.test_training_session tests.test_train_ai_ui
.\.venv\Scripts\python.exe -m unittest discover tests
```

## 6. Risks and Mitigations

### Risk: Flicker from full-screen clears or nested flips

Mitigation: Shared controller must not flip and must draw only inside caller
rectangles. Add tests that patch `pygame.display.flip()`.

### Risk: Training speed changes

Mitigation: Keep training simulation and throttling in the session/orchestration
layer. The renderer must not sleep, tick, or step simulation.

### Risk: Results change because stars alter RNG/world state

Mitigation: Stars are display-owned, created with isolated deterministic display
RNG, and omitted from gameplay `World` in both normal play and training. Add
display-on/off equivalence tests and RNG-isolation tests.

### Risk: Collision code retains hidden star assumptions

Mitigation: Audit collision snapshots, spatial indexes, typed world caches, and
broad world-object loops during Phase 5. Add or keep focused collision tests
showing stars are absent from collision candidates and collision behavior is
unchanged after stars leave `World`.

### Risk: HUD viewport clipping assumptions break in arbitrary rects

Mitigation: Refactor viewport drawing with explicit source/destination surfaces
and clips. Add tests for arbitrary rect placement.

### Risk: Button/control overlap in training

Mitigation: Complete the training layout phase before switching HUDs. Add layout
assertions for non-overlap.

## 7. Completion Definition

The migration is complete when:

- Normal play and training display both use the shared battle drawing
  controller.
- Both modes show the same battle presentation features, including stars and
  full HUD details.
- Only layout and surrounding controls differ between modes.
- Training display remains presentation-only.
- Shared draw code does not own timing, event handling, simulation, or display
  flipping.
- Focused and broad relevant tests pass.
