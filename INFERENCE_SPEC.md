# Battle AI Inference Specification

## Goal

Add optional trained-AI inference to battle mode while preserving the existing
lightweight game build and the existing GPU training build.

The feature must let the user select fleets for both sides, mark either side as
AI-controlled, automatically select AI ships, run trained models when available,
fall back to a simple deterministic/probabilistic controller when no trained
model is available, and show the selected AI status in the battle HUD.

This document is the implementation source of truth for the feature.

## Build And Dependency Requirements

### Existing Builds Must Remain Valid

- `build.cmd` must continue to build the lightweight `StarAI` distribution.
- `build.cmd` must not bundle PyTorch.
- `buildtrain.cmd` must continue to build the training distribution.
- The training distribution may continue to use GPU PyTorch.
- The lightweight distribution must still launch when PyTorch is absent.

### Add A CPU Inference Build

Add a third build target:

- `buildcpuai.cmd`
- `StarAI_CPUAI.spec`

`buildcpuai.cmd` should call `build.ps1` with `-BuildName "StarAI_CPUAI"`.

`StarAI_CPUAI.spec` should be based on `StarAI.spec` with these differences:

- Do not exclude `torch`.
- Exclude `torchvision` and `torchaudio`.
- Include the same ship, ability, model, and asset data as the normal build.
- Keep the executable name and collect name as `StarAI_CPUAI`.
- Smoke test through the same `build.ps1` path.

PyInstaller must bundle the installed PyTorch wheel from the active build
environment. It must not be expected to convert GPU PyTorch into CPU PyTorch.

### CPU PyTorch Installation

Add a documented dependency path for CPU inference builds. The implementation
should add a requirements file such as:

- `requirements-cpuai.txt`

The CPU AI requirements must install:

- `pygame-ce==2.5.2`
- `numpy==2.5.1`, unless a later implementation proves it can be omitted safely
- CPU-only `torch` matching the model checkpoint compatibility target

Use the official PyTorch CPU wheel index. The intended command is:

```powershell
python -m pip install -r requirements-cpuai.txt
```

The requirements file should not install CUDA PyTorch, `torchvision`, or
`torchaudio`.

Recommended local workflow:

1. Keep the existing `.venv` available for the current default workflow.
2. Use a separate CPU build environment, for example `.venv-cpuai`.
3. Update `build.ps1` only if needed to allow a specific Python executable or
   virtual environment to be selected without breaking existing commands.

If `build.ps1` is updated for an explicit Python path, existing `build.cmd` and
`buildtrain.cmd` behavior must remain unchanged.

## Runtime Requirements

### Fleet AI Selection

The existing per-player AI toggle in fleet selection remains the source of
truth for whether that side is AI-controlled.

Requirements:

- The user still selects fleets for both sides in fleet selection.
- AI toggles are saved and loaded with the fleet data.
- When a side is marked AI, that side is AI-controlled for every ship selected
  from that fleet during the match.
- A side marked AI remains AI-controlled even when no trained model is found.
  In that case it uses fallback controls and displays `AI: None found`.

### Ship Selection Automation

When a side is AI-controlled:

- If that side needs to select a ship, wait 0.5 seconds, then randomly select
  one alive ship from that side's available fleet.
- AI random selection must use the same alive-ship rules as manual and existing
  random selection.
- The AI random choice should be hidden using the existing random-lock behavior.
- If a side already has a surviving preselected ship, do not reselect it.
- If forced selection order is active after a battle, only start the AI side's
  0.5 second timer when that side is currently allowed to select.
- If both sides are AI-controlled and both selections are ready, wait 0.5
  seconds, then automatically activate Continue.
- If only one side is AI-controlled, do not auto-Continue; the human still
  controls match progression after both selections are ready.
- Escape/cancel/end-match behavior must remain available during automation.

Timing requirements:

- Use wall-clock menu time, not simulation frames.
- Timers reset when the relevant selection state changes.
- Do not block the event loop with `sleep`.

### Battle Input Ownership

When a side is AI-controlled:

- Human movement/action inputs must not affect that side's ship.
- F1 pause must still work.
- Escape/end-match must still work.
- The battle can still be paused, resumed, and exited by the user.
- AI-controlled ship controls must be reset when AI ownership starts for a
  selected round.
- Stale key-down state from human input must not leak into an AI ship.

Human-controlled sides must keep current behavior.

### Model Selection

Use the existing training model repository:

- Bundled models directory: `const.DEFAULT_MODELS_PATH`
- User models directory: `const.MODELS_PATH`
- Repository type: `TrainingModelRepository`

For each selected AI-controlled ship, resolve one model at round start.

Eligibility:

- The slot must exist.
- The `.pth` file must exist and be non-empty.
- PyTorch must be available.
- The checkpoint must load into the configured value-network architecture.
- The loaded model must support the current observation/action schema.

Selection priority:

1. Prefer a loadable default AI.
2. If no default AI is loadable, use the first loadable AI for that ship.
3. If no loadable AI exists, use fallback controls.

Default AI definition:

- A bundled model is a default AI.
- A user model whose description normalizes to `default` is also a default AI.
- If more than one default exists, pick the lowest slot number.
- If default candidates fail to load, continue scanning remaining eligible
  candidates before falling back.

First AI definition:

- Lowest slot number from `1` through `MODEL_SLOT_COUNT`.
- `TrainingModelRepository.slot_for()` already resolves bundled/user precedence
  for a slot and should be reused.

Status label:

- Loaded model label should use `model_basename(ship, slot)`, for example
  `Earthling-01`.
- If no model is loaded, label is `None found`.

### Model Loading Time

Models must not be loaded per frame.

Load or resolve the model:

- After both active ships for a round are known.
- Before the first actionable simulation step for that round.
- Again whenever `BattleSimulation.select_next_round()` changes one or both
  active ships.

The implementation should keep model loading outside the ship-selection menu.
Ship selection should only decide which ships enter the next fight. Battle mode
should own battle AI controller creation because it has the active ship pair,
simulation RNG, and world state.

Recommended behavior:

- Create battle AI controllers immediately after `BattleSimulation` is created.
- Recreate or refresh controllers immediately after `select_next_round()`.
- Cache successfully loaded models by `(ship_name, slot, pth_path)` within a
  match so the same model is not deserialized repeatedly.
- If model loading fails, record the failure for debugging and use fallback
  controls for that selected ship.

### Trained Model Inference

The inference path should reuse training contracts:

- `encode_observation()`
- `select_action_epsilon_greedy()` with `epsilon=0.0`, or an equivalent greedy
  inference helper
- `controls_for_action_index()`
- `ValueNetworkConfig`
- `build_value_network()`
- `load_training_checkpoint()`

Requirements:

- All PyTorch access must remain behind `src.training.torch_backend`.
- Runtime must tolerate PyTorch being absent.
- Inference must use CPU tensors.
- Inference must use `model.eval()`.
- Per-frame inference must run under `torch.inference_mode()` or `torch.no_grad()`.
- No training optimizer or replay buffer should be created for battle inference.
- No gradients should be tracked.
- Do not import `torch` directly outside the guarded backend/inference module.

The trained model controller should produce a normal battle action dictionary:

```python
{
    "forward": bool,
    "left": bool,
    "right": bool,
    "action1": bool,
    "action2": bool,
}
```

### Fallback Controls

If an AI-controlled side has no loadable trained model, use a fallback controller.

Each frame:

- Point the ship at its enemy.
- Move forward.
- If A1 is not pressed, press A1 with probability `1 / FPS`.
- If A1 is pressed, release A1 with probability `1 / FPS`.
- If A2 is not pressed, press A2 with probability `1 / (2 * FPS)`.
- If A2 is pressed, release A2 with probability `1 / (2 * FPS)`.

Fallback controller requirements:

- It is stateful for A1/A2 held state.
- It uses the battle simulation RNG so behavior is deterministic when the
  simulation RNG is seeded.
- It uses toroidal wrapped direction to face the enemy.
- It must not call gameplay mutating methods while deciding.
- It must return the same action dictionary shape as trained inference.

Turning behavior:

- If the target is within half a turn increment, hold neither left nor right.
- Otherwise hold exactly one of left/right.
- The sign convention must match the existing training helper or battle
  controls.

### HUD AI Status

If a player is AI-controlled, display the AI status immediately below that
player's HUD:

- Loaded model: `AI: Earthling-01`
- No model loaded: `AI: None found`

Requirements:

- Human-controlled sides show no AI label.
- Labels must be clipped or positioned so they do not overlap the status bars,
  ship viewport, pause/exit instructions, or screen edge.
- Labels must be drawn through the battle HUD rendering path so normal battle
  and any reused draw controller behavior stay consistent.
- Tests should assert label text is requested in the expected state; pixel tests
  are optional unless layout risk becomes high.

## Proposed Implementation Shape

### New Module: Battle AI Runtime

Add a battle-focused runtime module, for example:

- `src/Battle/battle_ai.py`

Responsibilities:

- Resolve model slots for selected ships.
- Load and cache CPU inference models.
- Create per-player AI controllers.
- Generate per-frame action dictionaries.
- Expose AI HUD labels.
- Keep PyTorch optional.

Suggested public API:

```python
class BattleAIManager:
    def __init__(self, ai_enabled, repository=None, rng=None):
        ...

    def bind_round(self, simulation) -> None:
        ...

    def actions_for_frame(self, simulation) -> dict[int, dict[str, bool]]:
        ...

    def is_ai_player(self, player: int) -> bool:
        ...

    def label_for_player(self, player: int) -> str | None:
        ...
```

`ai_enabled` should be a mapping such as `{1: bool, 2: bool}`.

### Battle Loop Integration

Update `battle.run()` to accept AI ownership flags:

```python
def run(..., player1_ai=False, player2_ai=False):
```

During each physics step:

1. Filter accumulated key changes so AI-controlled player action keys are
   ignored.
2. Ask the AI manager for per-frame actions.
3. Pass both human key changes and AI actions into `simulation.step()`.
4. Continue to allow F1 and Escape before filtering action keys.

When `state["needs_selection"]`:

1. Call `pick_ship.run()` with AI ownership flags.
2. Call `simulation.select_next_round(selected)`.
3. Re-bind AI controllers for the new active round before resuming battle.

### Ship Selection Integration

Update `pick_ship.run()` to accept:

```python
player1_ai=False
player2_ai=False
```

Behavior:

- Use these flags to drive auto-random selection.
- Use these flags to drive both-AI auto-Continue.
- Pass the flags through when starting `battle.run()`.

Update `pick_fleet.confirm_callback()` so saved AI toggle values are passed into
`pick_ship.run()`.

### Draw Integration

Update battle drawing to accept AI labels without requiring the draw layer to
know how models are loaded.

Possible API:

```python
draw_battle(..., ai_labels=None)
BattleDrawOptions(..., ai_labels: Mapping[int, str] | None = None)
```

`ai_labels` should map player number to final text value without the `AI: `
prefix, or to the full display string. Pick one representation and keep it
consistent.

## Testing Requirements

### Unit Tests

Add focused tests for:

- CPU AI build files exist and use the expected build name.
- Normal `StarAI.spec` continues to exclude `torch`.
- CPU AI spec does not exclude `torch`.
- AI model slot selection prefers bundled/default over first user slot.
- AI model slot selection falls back to first loadable model.
- AI model load failure results in fallback and `None found`.
- PyTorch unavailable results in fallback and does not crash.
- Fallback controller faces enemy and holds forward.
- Fallback controller A1/A2 press/release probabilities are driven by injected
  RNG.
- Battle input filtering ignores AI player's action keys but preserves human
  action keys.
- AI manager rebinds after `select_next_round()`.
- Ship selection auto-random selects only after 0.5 seconds.
- Both-AI auto-Continue triggers only after both selections are ready and 0.5
  seconds have elapsed.
- One-human/one-AI does not auto-Continue.
- HUD label selection returns expected strings.

### Integration/Smoke Tests

Existing smoke tests must continue to pass for:

- `build.cmd`
- `buildtrain.cmd`

Add smoke coverage for:

- `buildcpuai.cmd`, if PyTorch CPU is installed in the build environment.

If CI or local test environments do not have CPU PyTorch installed, CPU build
smoke tests may be opt-in, but normal unit tests must still pass without
PyTorch.

## Phases

### Phase 1: Dependency And Build Split

Deliverables:

- Add `requirements-cpuai.txt`.
- Add `StarAI_CPUAI.spec`.
- Add `buildcpuai.cmd`.
- Preserve `build.cmd` and `buildtrain.cmd`.
- Optionally update `build.ps1` to accept an explicit Python executable or venv
  without changing existing command behavior.
- Add tests or static assertions for build/spec configuration.

Acceptance criteria:

- `StarAI.spec` still excludes PyTorch.
- `StarAI_CPUAI.spec` includes PyTorch by not excluding it.
- `buildcpuai.cmd` targets `StarAI_CPUAI`.
- No runtime code depends on PyTorch being installed.

### Phase 2: Battle AI Runtime Skeleton

Deliverables:

- Add `src/Battle/battle_ai.py`.
- Implement AI manager, fallback controller, model selection policy, and label
  generation.
- Keep model loading optional and guarded.
- Add unit tests for model resolution and fallback behavior.

Acceptance criteria:

- With PyTorch unavailable, AI manager returns fallback controllers.
- Missing/empty/load-failing models produce `None found`.
- Default model priority and first-model fallback are tested.
- No per-frame model loading occurs.

### Phase 3: Battle Loop Integration

Deliverables:

- Extend `battle.run()` with AI flags.
- Bind AI controllers at initial round start.
- Rebind AI controllers after next-round selection.
- Filter AI-owned player input.
- Feed AI action dictionaries into `BattleSimulation.step()`.
- Add tests for input filtering and rebinding.

Acceptance criteria:

- AI-controlled ships ignore user movement/action keys.
- Human-controlled ships behave as before.
- Pause and exit still work for AI and human matches.
- AI actions are applied every simulation frame.

### Phase 4: Ship Selection Automation

Deliverables:

- Extend `pick_ship.run()` with AI flags.
- Pass AI flags from `pick_fleet.run()` to `pick_ship.run()`.
- Pass AI flags from `pick_ship.run()` to `battle.run()`.
- Implement non-blocking 0.5 second AI random-selection timers.
- Implement non-blocking 0.5 second both-AI auto-Continue timer.
- Add tests for selection timing and forced-order cases.

Acceptance criteria:

- AI sides randomly select alive ships after 0.5 seconds.
- AI random selections remain hidden.
- Both-AI fights progress without user pressing Continue.
- Human-involved fights still require Continue.
- End-match/cancel remains responsive.

### Phase 5: HUD Status Labels

Deliverables:

- Add AI label support to battle drawing.
- Show label only for AI-controlled players.
- Add tests for label routing and formatting.

Acceptance criteria:

- Loaded model displays `AI: <model basename>`.
- Fallback displays `AI: None found`.
- Human sides display no AI label.
- Labels do not overlap existing HUD elements in normal battle layout.

### Phase 6: End-To-End Verification

Deliverables:

- Run unit tests.
- Run lightweight smoke test.
- Run CPU AI smoke test in a CPU PyTorch environment.
- Manually verify:
  - human vs AI
  - AI vs human
  - AI vs AI
  - no-model fallback
  - pause/resume
  - exit/end-match
  - next-round ship selection after a death

Acceptance criteria:

- Normal build remains lightweight.
- CPU AI build can load trained models when CPU PyTorch is installed.
- GPU training build remains usable.
- Battle AI does not introduce visible frame-rate degradation.

## Performance Requirements

- Model deserialization must happen at round binding time, not per frame.
- Inference must be greedy and no-grad.
- The battle loop should make at most one model inference per AI-controlled
  ship per simulation frame.
- If profiling shows frame pressure, it is acceptable to add an AI decision
  interval and hold controls between decisions, but the initial implementation
  should attempt every-frame inference because the model is small.

## Non-Goals

- Do not add AI training changes.
- Do not require PyTorch in the normal lightweight build.
- Do not implement ONNX, NumPy-only, or custom exported inference in this phase.
- Do not change fleet file format unless existing AI toggle persistence is
  insufficient.
- Do not change battle physics or ship ability behavior.

## Open Implementation Notes

- The current training code has private helpers for opponent model loading.
  Battle inference should extract or duplicate only the minimal public loading
  path needed for inference rather than importing private orchestration helpers.
- Model compatibility should reuse existing metadata checks where practical.
- If metadata is missing but the checkpoint can be loaded into the current
  value-network architecture, the first implementation may treat it as loadable
  and rely on the checkpoint load to fail when incompatible.
- Log or retain model-load failure reasons for developer inspection, but do not
  show raw exception text in the battle UI.
