# AI Training Implementation Specification

## Prompt for a new implementation chat

Implement the AI training system described in this document. Work one phase at
a time, beginning with the earliest incomplete phase. Before changing code,
inspect the current repository because the battle engine, Training UI, model
repository, and tests may have changed. Preserve unrelated work and existing
game behavior.

For each phase:

1. Restate the phase scope and any assumptions.
2. Add or update tests alongside the implementation.
3. Run the focused tests, then the broader relevant test suite.
4. Stop after the phase is complete and report changed files, verification,
   remaining limitations, and the next phase.

Do not silently change the observation schema, action ordering, reward
definitions, or model metadata contract. If an engine detail makes a specified
feature impossible or ambiguous, report it before selecting a materially
different behavior.

## 1. Goal

Add trainable, per-ship battle AIs using a finite-horizon DQN-style value
network. Each model observes the complete battle state every simulation frame,
predicts an undiscounted rolling reward for each valid key combination, and
chooses one combination for that frame.

This is deliberately perfect-information AI. It may observe the position,
velocity, timers, crew, and battery of a cloaked ship. Cloaking only prevents
game-controlled tracking abilities from locking on.

Training always uses the current game settings for:

- Ship-direction quantization.
- Asteroid configuration.
- Input repeat-key delay.

Those settings are recorded with the model. Loading a model under different
settings must produce a clear warning rather than silently treating the model
as compatible.

Every AI-controlled Shofixti starts a training round with A2 fully armed, so a
single A2 press activates self-destruct.

## 2. Learning contract

### 2.1 Network

- PyTorch implementation using the repository's guarded optional-training
  import boundary.
- Fully connected network.
- ReLU after every hidden layer.
- Exactly 24 unrestricted linear outputs.
- Output `i` predicts the finite-horizon reward for action `i`.
- No softmax.
- Hidden-layer width and count come from the Training UI.
- The initially supported optimizer should be Adam using the UI learning rate.
- Use a regression loss against the selected action's realized target. Huber
  loss is preferred for robustness unless existing training infrastructure
  establishes a different tested convention.

This is DQN-style action-value selection, but the initial target is a direct
finite-horizon Monte Carlo return. There is no discount, bootstrap term, or
target network in the initial implementation.

### 2.2 Actions

The five output controls are:

1. Thrust
2. Turn left
3. Turn right
4. A1
5. A2

There are 32 binary combinations. Exclude the eight combinations in which left
and right are both pressed, leaving 24 actions.

- Define one stable, explicit action-index table.
- Persist an action-schema version with every model.
- An action specifies which keys are held for exactly one simulation frame.
- Select a new action every simulation frame.
- Apply changes through the normal ship control-state API so press, release,
  held-key, repeat-delay, edge-triggered, and A1+A2/A3 semantics remain intact.
- Select the highest predicted value except during epsilon-greedy exploration.
- Epsilon comes from the Training UI.
- Do not mask actions merely because energy or a cooldown currently prevents an
  ability; holding a temporarily unavailable control can be intentional.

### 2.3 Training sample and target

For a decision at simulation frame `t`, store:

```text
(observation_t, action_t, return_t)
```

where:

```text
return_t = sum of applicable rewards from frame t through
           min(t + prediction_window - 1, terminal frame)
```

Rules:

- The return is undiscounted.
- Death, round timeout, or any other terminal condition truncates every pending
  window that crosses that terminal frame.
- No reward after the terminal frame is included.
- Pointing and range shaping are special normalized components described in
  section 5; other event rewards accumulate normally.
- Pending samples only become replayable once their window matures or the round
  terminates.
- Replay-buffer capacity comes from the Training UI.
- Sample minibatches from replay at the end of each epoch initially, matching
  the existing product description.
- Keep update scheduling modular so training frequency can be changed after
  observing performance without changing observation or reward contracts.

## 3. Observation schema

### 3.1 Schema requirements

- Total input size: **533**.
- Use one canonical encoder and one canonical ordered field definition.
- Persist an observation-schema version.
- Encoder output must always be finite numeric values of exactly this length.
- Use the engine's toroidal nearest-path geometry for distance, bearing, and
  relative velocity.
- Sort nearest-object groups by toroidal distance with a deterministic
  tie-breaker.
- If an object slot is absent, all 11 values in that slot are zero.
- Dynamic movement values such as `turn_wait`, `thrust_wait`, `max_thrust`, and
  `thrust_increment` must be read from current runtime state. Limpet penalties
  and form changes must therefore be reflected immediately.

The layout is:

```text
25 enemy ship-type values
45 self values
45 enemy values
38 object slots * 11 values
= 533 inputs
```

### 3.2 Enemy ship type: 25 values

One-hot encode the enemy's ship type using the stable 25-ship catalog order.
The matching type is `1.0`; every other value is `0.0`.

Persist or version the exact catalog order rather than relying on incidental
dictionary ordering.

### 3.3 Per-ship block: 45 values

Encode this block once for self and once for the enemy, in the exact same
order:

1. Maximum crew / 50
2. Maximum battery / 50
3. Current crew / 50
4. Current battery / 50
5. Current `thrust_wait` / FPS
6. Current thrust timer / FPS
7. Current `turn_wait` / FPS
8. Current turn timer / FPS
9. Current thrust increment / 10
10. Current A1 wait / FPS
11. Current A1 timer / FPS
12. Current A2 wait / FPS
13. Current A2 timer / FPS
14. Current A3 wait / FPS
15. Current A3 timer / FPS
16. Current energy-wait timer / FPS
17. Absolute angle / 360
18. Absolute speed / 100
19. Absolute x velocity / 100
20. Absolute y velocity / 100
21. Countdown to thrust repeat readiness
22. Countdown to left repeat readiness
23. Countdown to right repeat readiness
24. Countdown to A1 repeat readiness
25. Countdown to A2 repeat readiness
26. Trackable, as 0 or 1
27. Thrust currently held, as 0 or 1
28. Left currently held, as 0 or 1
29. Right currently held, as 0 or 1
30. A1 currently held, as 0 or 1
31. A2 currently held, as 0 or 1
32. In Androsynth blazer form, as 0 or 1
33. In the alternate Mmrnmrhm form, as 0 or 1
34. Current limpet count / 64
35. Currently protected by an active damage shield, as 0 or 1
36. Currently in an Ilwrath cloak transition, as 0 or 1
37. Cloak transition direction: `-1` uncloaking, `0` stable, `1` cloaking
38. Sine of Orz turret angle relative to hull
39. Cosine of Orz turret angle relative to hull
40. Own Orz marines out and floating / 8
41. Own Orz marines currently boarded on the opposing ship / 8
42. Own Ur-Quan fighters out / 25
43. Own Chmmr satellites alive / 3
44. Own Chenjesu Dogis alive / 4
45. Own Kohr-Ah saws alive / 8

For non-applicable ship-specific fields, encode zero. For the turret cosine,
use zero rather than one when the ship has no turret so the two-field pair is
fully masked for non-Orz ships.

The enemy block uses the same ownership-relative meanings. Thus its “own
marines” are enemy-launched marines, and its “boarded” count is enemy marines
currently boarded on self.

The five repeat countdowns are frame counts:

- `0` if the control is not held.
- `0` if it is held and repeat-ready.
- Otherwise, the positive number of frames until it becomes repeat-ready.

The corresponding held flag resolves the intentional zero-value ambiguity.

### 3.4 Object slots: 38 groups

Track these groups in this exact order:

1. Enemy ship: 1 slot
2. Planet: 1 slot
3. Closest enemy A1 objects: 8 slots
4. Closest enemy non-A1 objects: 8 slots
5. Closest friendly A1 objects: 5 slots
6. Closest friendly non-A1 objects: 5 slots
7. Closest asteroids: 5 slots
8. Closest Syreen crew: 5 slots

This is 38 slots total.

Object classification rules:

- “A1” means explicitly created by or representing A1.
- Every ship ability object not explicitly A1 is non-A1, including objects
  internally associated with A2 or A3.
- Orz marines are non-A1.
- Chmmr satellites are excluded from all A1/non-A1 positional groups because
  their live count is already included in each ship block.
- Include relevant projectiles, lasers, areas, persistent special objects,
  fighters, marines, Dogis, saws, and similar spatial ability instances when
  they fit the ownership/action group.
- Natural engine objects such as the planet, asteroids, and Syreen crew remain 
  in their dedicated groups.

Each slot contains:

1. Object present, 0 or 1
2. Object expires, 0 or 1
3. Remaining timer / FPS; use 5 for a present non-expiring object
4. `sin(angle_to_object)` relative to observer heading and nearest-path position
5. `cos(angle_to_object)` relative to observer heading and nearest-path position
6. `min(5, 100 / distance_to_object)`
7. `sin(object_relative_velocity_angle)`
8. `cos(object_relative_velocity_angle)`
9. Object-relative speed / 100
10. Expected crew effect on contact / 10
11. Expected battery effect on contact / 10

For hostile objects, the last two values describe expected effects on self.
For friendly objects, they describe expected effects on the enemy. Crew gain
is negative expected crew damage, such as collectible Syreen crew. Values are
expectations used for observation, not reward events.

Ship/ability-specific adapters may be required to expose expiration,
classification, and expected contact effects. Keep defaults centralized and
make exceptions explicit and tested.

## 4. Ability pointing and range adapters

Reward settings differ by trained ship, so the user may set irrelevant rewards
to zero. Nevertheless, each ability needs stable predicates where applicable:

```text
is_pointing_at_enemy(ship, enemy, world)
is_enemy_in_effective_range(ship, enemy, world)
```

Requirements:

- Evaluate using current runtime form, turret direction, tracking state, and
  toroidal geometry.
- “Pointing” should use the best alignment permitted by current ship-direction
  quantization.
- Omnidirectional or automatically aimed abilities may define pointing as
  always true while their target is valid.
- When a target is cloaked/untrackable to an automatic ability, use that
  ability's actual fallback/default direction rather than granting automatic
  alignment.
- For lasers and area effects, range is true when activating the ability now
  would hit.
- For ordinary projectiles, account for projectile lifetime, projectile
  velocity, enemy velocity, and nearest-path separation.
- Exceptional weapons may override the generic calculation.
- An unsupported/nonexistent A1 or A2 predicate should return false, not raise.

Do not change battle mechanics to make these predicates easier to calculate.
They are read-only training adapters.

## 5. Rewards

Each reward has a user-selected signed weight from the existing Rewards UI.
Zero disables that reward. Keep reward-component totals separately observable
for diagnostics even though the network target uses their weighted sum.

Use a typed event ledger for events that cannot be recovered from endpoint
state. A loss followed by an equal gain must record both.

### 5.1 Normalized shaping rewards

These four components are normalized over the sample's actual window length,
including terminal truncation:

- Point A1 at enemy
- Get in A1 weapon range
- Point A2 at enemy
- Get in A2 weapon range

For each component:

```text
component value = qualifying frames / actual frames in this sample's window
```

Multiplying by its configured weight gives a maximum contribution equal to one
weight regardless of configured or truncated window length.

### 5.2 Window and event rewards

Implement the remaining rewards as follows:

- **Spawn A1 object:** Award once if one or more A1 objects are spawned during
  the window.
- **Spawn A2 object:** Award once if one or more non-A1 ability objects are
  spawned during the window for the A2 reward contract. A1+A2/A3-created
  objects are non-A1.
- **Get to high speed:** Award if self started the window at or below current
  `max_thrust` speed and ends the window above `max_thrust`. Planet-assisted
  speed is valid.
- **Enemy loses crew:** One unit per enemy crew lost from any cause. Count
  deaths of crew represented by launched Orz marines and Ur-Quan fighters.
  Launching those units is not itself crew loss. Later healing, regeneration,
  or Pkunk reincarnation does not cancel prior loss.
- **Debuff enemy:** One unit per application or reapplication. Includes each
  limpet increment, each boarding marine, each Melnorme confusion application,
  and Chenjesu/Dogi battery-drain debuffs.
- **Kill enemy object:** One unit when an enemy projectile or special ability
  object is destroyed before normal expiration. Direct trainee attribution is
  not required; leading it into a planet, asteroid, or another object counts.
- **Kill enemy:** One unit whenever enemy crew reaches zero. Count it even if
  Pkunk subsequently reincarnates.
- **Gain crew:** One unit per crew gained through Mycon regeneration or
  collecting Syreen crew. Pkunk reincarnation does not count.
- **Gain battery:** Positive endpoint battery delta, one unit per battery point.
- **Lose crew:** One unit per self crew lost from any cause. Use the same
  launched-unit rules as enemy crew loss. Later healing or reincarnation does
  not cancel it.
- **Lose battery:** Positive magnitude of a negative endpoint battery delta,
  one unit per battery point.
- **Battery at zero:** Award if self battery is zero at the end of the window.
- **Get debuffed:** One unit per debuff application or reapplication to self,
  using the same event meanings as “Debuff enemy.”
- **Die:** One unit whenever self crew reaches zero, even if Pkunk subsequently
  reincarnates.

Endpoint battery rewards deliberately use start/end state; crew and debuff
rewards deliberately use ledger events.

### 5.3 Event-ledger requirements

Record enough typed data to calculate rewards without reverse-engineering
collision side effects:

- Frame identifier
- Event type
- Acting/owning ship when meaningful
- Affected ship or object
- Magnitude
- Ability/action classification when meaningful
- Whether object removal was destruction versus natural expiration/return

Instrumentation must not change collision ordering, RNG consumption, damage,
spawning, or other battle behavior when training is disabled.

## 6. Training rounds and epochs

### 6.1 Simple-behavior opponents

Use the currently selected simple movement/turning/A1/A2 behavior settings.
For each repetition, play that configuration against all 25 ship types:

```text
rounds per epoch = UI rounds_per_epoch * 25
```

Use a deterministic documented ship order before any optional shuffling.

### 6.2 Existing-AI opponents

- Discover up to four stored AI slots for each of 25 ships.
- Skip empty slots.
- Include bundled and user models that can be loaded.
- Older snapshots may be opponents.
- Maximum is 100 opponent AIs per repetition.

```text
rounds per epoch = UI rounds_per_epoch * available_AI_count
```

Freeze the opponent list and loaded model snapshots at a safe boundary such as
epoch start so saving the trainee cannot mutate an opponent during a round.
Report incompatible or unloadable opponents and skip them safely.

### 6.3 Round behavior

- Use the current ship-direction, asteroid, repeat-delay, and other normal
  battle settings.
- Respect the UI match-time limit.
- Reset battle state fully between rounds.
- Fully arm every AI-controlled Shofixti before the first actionable frame.
- Display-off mode must avoid rendering work while preserving identical
  simulation semantics.
- Display-on mode may visualize the current training battle at normal UI
  presentation speed without changing physics results.
- Stopping training should finish or cancel at a documented safe boundary,
  preserve a valid model/replay state, and leave the UI responsive.

## 7. Persistence and compatibility

Continue using the repository's bundled-versus-user model-slot rules:

- Bundled models are read-only.
- User models may be created, updated, or deleted.
- Save model weights in `.pth`.
- Save descriptive and compatibility metadata in the JSON sidecar.
- Use atomic metadata writes and a safe weight-save strategy.

Metadata must include at least:

- Metadata schema version
- Observation schema version and input size 533
- Action schema version and exact action ordering
- Ship name and slot
- Description
- Hidden-layer count and width
- Output count 24
- Optimizer/loss identifiers
- Training settings, including learning rate, epsilon, replay capacity,
  prediction window, rounds per epoch, match-time limit, and opponent mode
- Reward names and weights
- Ship-direction setting
- Asteroid setting
- Repeat-key delay
- Simulation FPS
- Training progress counters

When loading:

- Validate architecture and schema before using weights.
- Reject structurally incompatible files with a clear user-facing error.
- Warn for game-setting differences such as directions, asteroids, repeat
  delay, or FPS.
- Do not overwrite a bundled slot.
- Keep lightweight/non-training builds functional when PyTorch is absent.

## 8. UI and diagnostics

Connect the existing Training UI without redesigning it unless required:

- Start/stop training.
- Selected trainee ship and writable slot.
- Slot description.
- Opponent mode and simple behaviors.
- Reward weights.
- Replay-buffer size.
- Rounds per epoch.
- Prediction window.
- Match-time limit.
- Learning rate.
- Epsilon.
- Hidden-layer width/count.
- Display toggle.

Show useful live status:

- Epoch and round progress
- Current opponent
- Current round frame/time
- Replay-buffer occupancy
- Most recent loss
- Exploration versus greedy action
- Weighted total return
- Per-component reward totals
- Model-save or compatibility errors

Do not implement a separate automated evaluation system in the initial phases.
Initial assessment is by user observation. Keep training metrics structured so
later evaluation can reuse them.

## 9. Implementation phases

### Phase 1: Contracts, action table, and model

Scope:

- Add versioned constants/types for the 533-input observation and 24 actions.
- Implement and test the stable action-index table.
- Implement the configurable ReLU/linear-output network.
- Add guarded model construction, prediction, and selected-action regression.
- Extend metadata structures without running battles yet.

Acceptance:

- Exactly 24 unique valid actions; none holds left and right together.
- Model accepts `[batch, 533]` and returns `[batch, 24]`.
- Outputs are not softmax-normalized and can be negative.
- Lightweight startup works without PyTorch.
- Metadata round-trips schema and architecture fields.

### Phase 2: Base observation encoder

Scope:

- Encode ship type and the general portions of both 45-value ship blocks.
- Add held-key and repeat-countdown encoding.
- Add dynamic timers, movement values, velocity, and tracking state.
- Use a pure/read-only encoder interface.

Acceptance:

- Encoder produces exactly 533 finite values with object sections temporarily
  zero-filled behind an explicit boundary.
- Held versus unheld repeat-ready states are distinguishable.
- Limpet-altered waits/thrust values update immediately.
- Tests cover representative forms, cooldowns, held keys, and zero defaults.

### Phase 3: Ship-specific state and object observations

Scope:

- Complete all ship-specific fields.
- Add friendly/hostile A1 and non-A1 classification.
- Add the 38 object slots and toroidal geometry.
- Add expiration and expected-contact-effect adapters.
- Exclude satellites from positional ability groups.

Acceptance:

- Exact slot ordering and zero masking are tested.
- Toroidal nearest-object selection is correct at arena boundaries.
- Stable tie ordering is reproducible.
- Friendly damage features target the enemy; hostile features target self.
- Marine, fighter, satellite, Dogi, and saw counts match live engine state.

### Phase 4: Pointing/range adapters and event ledger

Scope:

- Add read-only A1/A2 pointing and range predicates.
- Introduce typed battle events needed by rewards.
- Instrument damage, healing, battery change, debuffs, spawn, destruction,
  death, and reincarnation without changing mechanics.

Acceptance:

- Representative conventional and exceptional abilities are characterized.
- Loss followed by healing produces both ledger events.
- Debuff refreshes count.
- Pkunk reincarnation is excluded from crew gain but does not erase death.
- Natural object expiration is distinguishable from destruction.
- Training-disabled battle characterization tests remain unchanged.

### Phase 5: Reward and rolling-return pipeline

Scope:

- Implement all reward components.
- Implement pending rolling windows and terminal truncation.
- Normalize pointing/range by actual truncated length.
- Produce mature `(observation, action, return)` samples.

Acceptance:

- Exact-window and terminal-truncated examples are tested frame by frame.
- Pointing/range maxima do not change with window length.
- Crew loss/healing and death/rebirth examples match this specification.
- Endpoint battery rewards and event-ledger crew rewards remain distinct.
- No sample includes post-terminal rewards.

### Phase 6: Replay and optimization

Scope:

- Add bounded replay storage.
- Add epsilon-greedy inference.
- Add minibatch selected-action regression and optimizer state.
- Save/load weights and training state safely.

Acceptance:

- Replay eviction is deterministic and capacity-bound.
- Epsilon 0 is greedy; controlled RNG tests cover exploration.
- Only selected action outputs receive direct regression loss.
- A synthetic learnable problem reduces loss.
- Saved/reloaded predictions match.

### Phase 7: Training orchestration

Scope:

- Run full training rounds and epochs.
- Implement both opponent modes.
- Add Shofixti arming, timeout handling, terminal flushing, and display-off
  simulation.
- Perform the initial end-of-epoch replay updates.

Acceptance:

- Simple mode schedules exactly `rounds_per_epoch * 25`.
- Existing-AI mode skips empty slots and uses the correct available count.
- Every pending sample is flushed correctly at terminal state.
- Shofixti requires one A2 press.
- A short deterministic training regimen completes without rendering.

### Phase 8: UI integration and hardening

Scope:

- Connect Start/Stop and all UI settings.
- Show progress and reward diagnostics.
- Add compatibility warnings and error handling.
- Verify display-on versus display-off simulation equivalence.
- Profile obvious bottlenecks without changing learning semantics.

Acceptance:

- A user can create, train, stop, save, reload, and continue a writable model.
- Bundled slots remain protected.
- Setting mismatches warn clearly.
- UI remains responsive and reports failures without corrupting the model.
- Focused training tests and the full relevant game suite pass.

## 10. Deferred work

Do not include these unless separately requested after the initial system works:

- Automated benchmark/evaluation phase.
- Discounted or bootstrapped targets.
- Target networks or Double DQN.
- Prioritized replay.
- Multi-process simulation.
- Recurrent networks or frame stacking.
- Automatic network-size restrictions.
- Curriculum design.
- Dynamic update-frequency tuning.

Design module boundaries so these can be added later without changing stored
observation/action schemas unnecessarily.

## 11. Known specification decisions to preserve

- Perfect information includes cloaked enemy state.
- Cloaking still affects normal game-controlled tracking.
- New action every simulation frame.
- Linear outputs, not softmax.
- Undiscounted finite rolling return.
- Terminal conditions truncate pending windows.
- Pointing/range are fractions of the actual window length.
- Friendly object positions are observed in 5 A1 and 5 non-A1 slots.
- Enemy object positions use 8 A1 and 8 non-A1 slots.
- Satellites are represented by count, not positional slots.
- All non-A1 ability objects are grouped as non-A1 for observation purposes.
- Destruction reward does not require direct trainee attribution.
- Debuff reapplication counts.
- Crew loss and crew gain are independently counted.
- Pkunk reincarnation counts as death/kill but not crew gain.
- Current runtime movement values reflect limpets and form changes.
- Model-setting mismatches warn the user.
