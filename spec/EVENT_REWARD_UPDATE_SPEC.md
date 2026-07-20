# Event Reward Update Specification

## Status And Authority

This document is the implementation source of truth for causal placement of
delayed discrete training rewards.

The feature changes training reward attribution only. It must not change battle
damage, collisions, ability behavior, ship replacement, audio, rendering, or
non-training play.

Implementation work should follow the phases in this document. A later phase
must not be enabled until the acceptance criteria for all preceding phases pass.

## Goal

Attribute delayed positive combat effects to the trainee actions that caused
them instead of to the frames on which the effects happen.

Examples:

- A projectile launched on frame `-5` and hitting on frame `0` places its
  crew-loss or kill reward on frame `-5`.
- Frames `-4` through `0` receive no credit from that hit.
- Frame `-6` receives `gamma * reward`, frame `-7` receives
  `gamma**2 * reward`, and propagation continues backward to the configured
  discount cutoff.
- A Chenjesu Dogi that applies several debuffs places every debuff reward on
  the Dogi's original launch frame.
- An Orz marine that boards, damages crew repeatedly, and eventually kills a
  ship places those positive rewards on the marine's original launch frame.
- A trainee weapon that permanently destroys an enemy crew-bearing fighter may
  produce both `Kill enemy object` and `Enemy loses crew`; both components use
  the destroying weapon's origin.
- A laser fired automatically by a trainee Chmmr satellite creates a new origin
  on that firing frame, owned by the parent Chmmr ship's open trajectory. Kills,
  crew loss, and enemy-object kills caused by that laser use that firing-frame
  origin rather than the satellite's spawn frame.
- A trainee projectile that survives an enemy death may damage or kill the
  replacement enemy and still credit its original launch action.

The trainee's life is the reward trajectory. Enemy death remains a combat and
statistics boundary, but it is not a reward-propagation terminal.

## Non-Goals

This feature does not:

- Predict or award an effect when an ability is merely fired. The effect must
  actually occur before reward is placed.
- Give impact-frame credit to the frames between an ability origin and its
  effect.
- Change the existing gamma value or discount cutoff.
- Introduce infinite-horizon returns beyond the existing cutoff.
- Reallocate incoming penalties to opponent action frames.
- Add proportional kill credit across every historical damage source in the
  first implementation.
- Change gameplay cleanup or force abilities to disappear when an enemy dies.
- Update replay entries that have already participated in optimization.
- Change reward slider names, stored reward keys, or reward weights.
- Redefine the existing `Use A1` and `Use A2` rewards as release rewards.

## Terminology

### Effect Frame

The simulation frame on which a debuff, crew change, or death actually occurs.

### Origin Frame

The decision frame containing either:

- the successful trainee action that created or materially transformed the
  ability responsible for an effect; or
- an explicitly specified autonomous child action, currently a trainee Chmmr
  satellite firing its laser, attributed to the parent ship's staged sample on
  that frame.

### Combat Episode

A period ending when either ship dies. Combat episodes continue to define
kills, deaths, wins, losses, draws, opponent replacement, and associated UI
statistics.

### Reward Trajectory

The sequence of trainee decisions beginning when a trainee ship life starts and
ending when that trainee dies or its simulation is discarded. One reward
trajectory may span several enemy lives and combat episodes.

### Causal Credit

Immutable training metadata carried by an ability. It identifies the open
trainee reward trajectory and one or more weighted origin frames.

### Staged Sample

A current-trajectory observation, action, and mutable immediate reward-component
vector that has not yet been committed to the ordinary replay buffer.

## Current-System Constraints

The implementation must account for these existing properties:

- `RollingReturnPipeline` currently places events in the matching effect-frame
  outcome and matures samples after a fixed discount horizon or terminal.
- `TrainingReplayBuffer` stores packed immutable observation/action/return
  values and has no stable update API.
- Batch optimization starts only after simulation work for the batch finishes.
- Independent, coordinated in-process, coordinated worker-process, and worker
  helper paths all produce reward samples.
- Worker processes currently transfer mature samples incrementally.
- Training immediately replaces dead ships in the existing simulation.
- Surviving abilities may have their opponent restored to the replacement
  ship.
- `perform_action1_release()` is called on a release edge, but release is not
  currently represented by `EVENT_ACTION_USED`.
- Chenjesu shards are created from a source crystal but use the player ship as
  their gameplay parent, so gameplay parent traversal alone cannot recover the
  crystal's causal origin.
- Chmmr satellites spawn lasers automatically during their update. Each laser
  needs a new parent-owned firing-frame origin rather than inherited satellite
  spawn credit.
- Permanent loss of a launched crew unit may currently be recorded with the
  unit itself as source even when a trainee weapon destroyed it, while the
  accompanying object-removal event retains the destroying source.
- Some debuff paths currently record incomplete source information and require
  an attribution audit.

Relevant implementation surfaces include:

- `src/training/rewards.py`
- `src/training/event_ledger.py`
- `src/training/replay.py`
- `src/training/orchestration.py`
- `src/training/coordinated.py`
- `src/training/coordinated_simulation.py`
- `src/training/process_worker.py`
- `src/training/worker_protocol.py`
- `src/Battle/collision_responses.py`
- `src/Objects/Ships/space_ship.py`
- `src/Objects/Ships/ability.py`
- `src/Objects/Ships/Chmmr/A3/ChmmrSatellite.py`
- `src/Objects/Ships/Chmmr/A3/ChmmrSatelliteLaser.py`
- `src/Objects/Ships/KzerZa/A2/KzerZaA2.py`
- `src/Objects/Ships/Orz/A3/OrzA3.py`
- ship-specific press/release and derived-object implementations

## Normative Reward Semantics

### Basic Placement

For an effect reward amount `R`, origin frame `s`, decision frame `i`, gamma
`g`, and discount horizon `H`, the effect contributes:

```text
g ** (s - i) * R
```

when:

```text
i <= s < i + H
```

The effect frame is irrelevant to this discount once the effect has been
causally routed. Impact latency must not reduce the reward on the origin frame.

An effect that occurs after the origin has fallen outside the old rolling
window must still credit the origin, provided its reward trajectory remains
open. The cutoff applies backward from the origin, not backward from the effect
frame.

### Weighted Origins

An ability may have more than one origin. Origin weights must be finite,
non-negative, and sum to `1.0` within a small floating-point tolerance.

For origins `(s_j, w_j)`, an effect amount `R` places `w_j * R` at each `s_j`.
Normal discounted propagation then begins independently at every origin.

### Components Relocated To Causal Origins

The first complete implementation relocates these positive trainee-caused
components:

- `Enemy loses crew`
- `Debuff enemy`
- `Kill enemy object`
- `Kill enemy`

Relocation occurs only when the event source resolves to valid causal credit
owned by the open trainee reward trajectory.

### Components That Remain At Their Existing Frames

These components retain their current timing unless explicitly excepted below:

- `Lose crew`, except permanent loss of trainee Kzer-Za A2 fighters and Orz A3
  marines
- `Get debuffed`
- `Die`
- `Gain crew`
- `Gain battery`
- `Lose battery`
- `Use A1`
- `Use A2`
- `Destroy own object`
- all ongoing point/range/speed/zero-battery rewards
- `Lose crew` and `Die` components caused by damage to or destruction of the
  trainee's own Chmmr satellites

Other incoming negative events remain effect-timed so earlier trainee movement
and avoidance decisions receive ordinary backward credit or penalty.

### Permanent Loss Of Trainee Launched Crew

In live `causal` mode, permanent loss of a trainee Kzer-Za A2 fighter or Orz A3
marine is an exception to ordinary effect-timed incoming penalties:

- Natural expiration, environmental loss, opponent damage, boarded RNG, host
  destruction, and other non-trainee causes place `Lose crew` at the lost
  unit's launch origin.
- When a trainee-owned projectile, special object, laser, or area effect causes
  friendly fire, compare the damaging object's spawn stamp with the launched
  crew unit's spawn stamp. The later spawn owns the loss.
- A spawn stamp contains both frame and ledger sequence. Equal stamps split the
  component 50/50 between the two causal credits.
- The launched unit credit, damaging-source credit, and both spawn stamps are
  snapshotted on the crew-loss event. Reward origins are not used as a proxy
  for spawn ordering because an ability may have weighted press/release
  origins.
- Missing provenance retains the existing effect-frame component and records a
  diagnostic. If the selected provenance is closed, omit the component rather
  than assigning it to the effect frame or another trajectory.
- `legacy` and `shadow` retain the existing effect-frame `Lose crew` behavior.

Safe return remains a crew transfer and produces no `Lose crew` component.

### Ownership Requirement

Target identity alone is insufficient for a relocated positive reward. The
router must confirm that causal credit belongs to the current open trainee
reward trajectory.

An opponent projectile, environmental object, opponent self-damage source, or
closed trainee-life source must not receive trainee causal placement.

For object destruction, ownership of the destroyed object is not causal
ownership. The destroying source recorded on the removal event must resolve to
the current trainee trajectory. Destroying an enemy object does not by itself
prove trainee causation.

Existing source reward factors and enemy-death reward factors remain in force.
The router changes where the calculated amount is placed, not how the amount is
calculated.

The enemy object present at effect time does not need to be the enemy object
present at launch time. If the trainee trajectory is still open, hitting a
replacement enemy is a valid routed effect by design.

### Non-Routeable Events

Legitimate events without a trainee ability origin continue to use their
existing behavior.

An event from an ability type that is expected to carry causal credit but does
not must:

- increment a missing-provenance diagnostic counter;
- use impact-frame fallback during staged rollout rather than silently dropping
  or duplicating reward; and
- fail focused tests for any covered built-in ability.

The final implementation should normally report zero missing-provenance events
for built-in trainee abilities.

### No Double Counting

A relocated component is added to its origins and omitted from the effect
frame. It must never be present in both locations.

Two different configured components caused by one event are not double
counting. In particular, permanent destruction of an enemy crew-bearing object
may retain both `Kill enemy object` and `Enemy loses crew`, each exactly once.

Raw event counts may be retained for diagnostics, but they must not contribute
twice to a training return.

## Reward-Trajectory Lifecycle

### Start

A new reward trajectory begins when a trainee ship life becomes active. It
receives a unique trajectory identifier that cannot collide with another
simulation, worker, or replacement ship.

The identifier must not be only a player number or bare frame number.

### Enemy Death

Enemy-only death:

- ends a combat episode;
- records the win/kill outcome;
- may reset exploration span as it does now;
- does not close or flush the reward trajectory;
- does not invalidate causal credit owned by the living trainee; and
- does not reset the causal reward staging buffer.

The replacement enemy begins a new combat episode inside the same trainee
reward trajectory.

### Trainee Death

Trainee death, including simultaneous death:

- processes all events from the death frame first;
- places any outgoing causal rewards generated on that frame;
- places incoming crew-loss and death penalties on the death frame, except for
  the permanent trainee launched-crew rule above;
- closes and finalizes the reward trajectory;
- invalidates every causal origin owned by that trajectory; and
- starts a fresh trajectory for the replacement trainee on the next decision.

Any gameplay object that survives the old trainee's death must not update the
closed trajectory or credit the replacement trainee.

### Window, Round, Stop, And Batch Boundaries

When a simulation window or round ends, the active trajectory is finalized
using effects observed so far, even if the trainee is still alive. Its objects
are being discarded, so no later reward update is possible.

A stop request that discards a simulation must either finalize the observed
partial trajectory through its last completed frame or explicitly discard the
entire partial trajectory. The implementation must choose one behavior
consistently across all execution paths. The recommended behavior is to
finalize completed frames, matching current timeout flush behavior.

Batch optimization must not begin until every active trajectory belonging to
that batch has been finalized and committed.

## Causal-Credit Data Contract

The names below are recommended. Equivalent names are acceptable if the same
invariants are enforced.

```text
RewardOrigin
    trajectory_id
    frame_index or stable staged_sample_id
    weight
    kind: "press" | "release" | "launch" | "autonomous_fire" | other future kind

AbilityRewardCredit
    trajectory_id
    origins: immutable tuple[RewardOrigin, ...]
```

Requirements:

- Credit is training-only metadata.
- Credit must not contain the full observation.
- An origin must resolve to exactly one staged decision.
- An origin from a closed trajectory is invalid for future updates.
- Derived abilities inherit credit explicitly or through an audited causal
  source chain.
- Replacing an ability's future credit must not rewrite rewards from effects
  that already occurred.
- Weight validation happens when credit is created or replaced.
- Credit lookup must tolerate an object being detached from its gameplay
  parent.

The preferred ownership location is `BattleEventLedger` or a dedicated
training reward context referenced by the ledger. Gameplay objects may carry a
small opaque token or causal-source reference when necessary, but they should
not own replay samples or reward calculations.

## Provenance Creation And Propagation

### Successful Action Commit

Except for an explicitly specified autonomous origin, only a successfully
committed trainee action may create an origin. The only autonomous origin in
this specification is a laser fired by a trainee Chmmr satellite as defined
below.

Invalid presses caused by cooldown, insufficient energy, ship-specific limits,
or other validation failures must not create credit.

`SpaceShip.commit_action()` already knows:

- the committed action number;
- the spawned objects;
- the current ledger frame; and
- whether the action was valid.

The training hook at or immediately after commit should bind default launch
credit `[(current_decision, 1.0)]` to each causally created ability object.

### Derived Objects

Every derived damaging or debuffing object must inherit causal credit from the
ability that created it, not merely from the root player ship, unless this
specification explicitly requires the child to create a new origin. A Chmmr
satellite laser is that explicit exception.

The implementation must audit at least:

- Chenjesu A1 shards;
- lasers or projectiles spawned by fighters;
- Chmmr satellite lasers, using their special firing-frame rule;
- area effects created by an ability;
- children spawned during an ability update;
- long-lived special objects;
- abilities whose gameplay parent is detached during cleanup; and
- abilities that reacquire a replacement opponent.

Provide a common helper for credit inheritance. Avoid unrelated one-off fields
in every ship implementation where a generic helper suffices.

### Event Source Completeness

Crew-loss, debuff, object-removal, and ship-death events must retain enough
source information to resolve causal credit.

Audit all calls to:

- `record_crew_changed()`
- `record_launched_crew_lost()`
- `record_debuff_applied()`
- `record_object_removed()` / `record_removed()`
- `take_damage()`
- launched-unit `on_destroyed()` paths
- ship-specific debuff methods such as limpet and confusion application

When a trainee source permanently destroys an enemy launched-crew unit, the
launched-crew-loss event must retain that destroying source. It must not replace
the source with the destroyed unit merely because crew accounting happens from
the unit's `on_destroyed()` callback. Correlation with the matching
object-removal event is acceptable only if it is deterministic, same-destruction
identity is preserved, and ordering cannot associate the wrong source.

The source audit must not change damage or debuff mechanics.

## Press/Release Attribution

### Release Decision Identity

The release origin is the current trainee decision frame on which the input
changes from A1 pressed to A1 not pressed.

Because the action space represents complete control combinations, the release
origin uses the complete selected action index for that frame, including any
simultaneous movement or A2 input.

Release must be represented by a distinct training event or hook. It is not a
second `Use A1` reward.

No release origin is created if the release changes no live ability.

### Chenjesu A1

- A crystal begins with `[(press, 1.0)]`.
- A damaging effect before a valid release uses the full press origin.
- A valid release affecting the live crystal changes future credit to
  `[(press, 0.5), (release, 0.5)]`.
- Every shard inherits the two-origin credit.
- Effects already recorded before release are not adjusted.
- Releasing after the crystal is gone creates no release origin.
- A release accepted before collision processing on a frame is considered to
  affect later effects from that frame.

### Kohr-Ah A1

- Each saw begins with its own `[(press, 1.0)]` credit.
- A release applies to every live saw actually commanded by that release.
- Future credit for each affected saw becomes
  `[(that_saw_press, 0.5), (latest_effective_release, 0.5)]`.
- A saw hit before any release remains fully press-attributed.
- A later effective release replaces only the release half for future effects.
- Rewards from prior effects remain assigned to the release active when those
  effects occurred.

### Melnorme A1

Melnorme A1 also changes materially from held/charging to launched on release.
For consistency it is in scope:

- Effects while held use the press origin.
- A valid release changes future credit to a 50/50 press/release split.
- A release with no live held projectile creates no release origin.

If implementation intentionally excludes Melnorme, that deviation must be
recorded in the progress document and covered by a test demonstrating the
chosen behavior.

### Discount Interaction For Split Origins

For a press at `-5`, release at `-2`, and effect at `0`, immediate causal
placement is:

```text
frame -5: 0.5 * R
frame -2: 0.5 * R
```

Normal return calculation means the press-frame sample receives:

```text
0.5 * R + gamma**3 * 0.5 * R
```

Frames `-4` and `-3` receive only discounted propagation from the release
half. Frames `-1` and `0` receive none.

This is intentional. The 50/50 split describes causal reward deposits, not
exclusive final sample returns. Action-exclusive credit is out of scope.

## Long-Lived Ability Rules

### Chenjesu Dogi

- Launch creates one full-weight origin.
- Every successful enemy debuff event places a full debuff reward on that
  launch origin.
- Repeated debuffs accumulate without a per-launch cap unless a future reward
  balancing change explicitly adds one.
- Enemy replacement does not change the origin.
- Trainee death closes the origin.

### Orz Marine

- Launch creates one full-weight origin.
- Boarding debuff, enemy crew loss, and a lethal kill caused by that marine use
  the same origin.
- Repeated crew-loss events accumulate at launch.
- Marine return, destruction, and the existing launched-crew accounting keep
  their gameplay behavior.
- Negative trainee crew-loss accounting remains effect-timed unless a separate
  specification changes it.

### Other Persistent Fighters And Special Objects

Kzer-Za fighters, Syreen crew, Kohr-Ah saws, and other persistent trainee
objects follow the same origin lifecycle. Their effects may cross enemy deaths
but may not cross trainee death or window disposal for reward purposes.

### Chmmr Satellite Autonomous Fire

Chmmr satellites exist automatically with the ship and do not have a trainee
launch action suitable for causal attribution. Their outgoing lasers therefore
use this explicit autonomous-origin rule:

- When a satellite belonging to the trainee creates a
  `ChmmrSatelliteLaser`, create full-weight credit on the current staged
  decision frame with kind `autonomous_fire`.
- The credit is owned by the open reward trajectory of the root parent Chmmr
  ship. The reward is deposited in that parent ship's sample for the firing
  frame, including its complete selected action index, even though firing is
  automatic.
- Create one independent origin for every laser firing. Do not inherit the
  satellite's creation time, the ship-life start, or an earlier laser's origin.
- Bind the origin when the laser is spawned, not when it hits. The staged sample
  for the frame must already exist before satellite updates run so a same-frame
  hit can resolve it.
- If one selected decision advances more than one internal simulation tick, use
  the staged parent decision whose simulation interval contains the firing
  tick. Do not create an observation-less replay entry for an internal tick.
- `Enemy loses crew`, `Kill enemy`, and `Kill enemy object` caused by that laser
  use its firing-frame credit. This includes the existing fractional positive
  components produced when the laser damages or destroys an enemy Chmmr
  satellite and the two separate components produced when it destroys an enemy
  crew-bearing fighter.
- The laser and all events caused by it must preserve the root parent as owner
  and the laser as causal source. A ship-death or launched-crew-loss callback
  must not collapse the source to the satellite or destroyed target.
- An opponent Chmmr satellite cannot create credit in the trainee trajectory.
  A laser from a closed trainee trajectory is invalid, while a laser hitting a
  replacement enemy during the same open trainee trajectory remains valid.

For this rule, an effect "caused by a Chmmr satellite" means an effect whose
causal source is its `ChmmrSatelliteLaser`. Destruction caused by contact with
the satellite body has no corresponding laser-fire frame and must use the
ordinary non-routeable fallback. It must not reuse the most recent laser origin.

## Enemy-Object And Launched-Crew Destruction Attribution

`Kill enemy object` uses final-source attribution, parallel to ship kills:

1. Preserve the existing eligibility rules and reward amount for deciding that
   a destroyed projectile or special object earns `Kill enemy object`.
2. Resolve causal credit from the destroying source stored on the matching
   `EVENT_OBJECT_REMOVED` event.
3. If the source belongs to the open trainee trajectory, place the component at
   its weighted origin or origins and omit it from the destruction frame.
4. If no valid trainee origin exists, use the specified non-routeable fallback
   and diagnostics. Natural expiration, cleanup, environmental destruction,
   and opponent-caused destruction must not be converted into trainee credit.

This feature changes timing and provenance, not which object types qualify or
the configured component amount.

A crew-bearing fighter or marine can be both an enemy object and launched crew.
When a trainee source permanently destroys one, the existing semantics may
produce both:

- one `Kill enemy object` component for destroying the special object; and
- one `Enemy loses crew` component for permanent crew loss by its parent ship.

These are distinct configured rewards and are intentionally both retained. Do
not coalesce, suppress, or normalize one because the other occurred. Both must
resolve from the actual destroying source and normally use the same weighted
origins. If the unit returns safely, no permanent crew-loss component occurs.
If it expires or is lost without trainee causation, no trainee causal origin may
be fabricated.

Enemy Chmmr satellite HP and destruction events retain their existing
fractional `Enemy loses crew` and `Kill enemy` amounts. Those positive
components follow the causal source like other relocated components, including
the Chmmr satellite laser firing-frame rule above. Damage to or destruction of
the trainee's own satellites continues to produce effect-timed incoming
`Lose crew` or `Die` components.

## Kill Attribution

The first implementation uses final-source kill attribution.

When `EVENT_SHIP_DIED` targets the enemy:

1. Calculate the kill reward amount using the existing enemy-death reward
   factor logic.
2. Resolve causal credit from the lethal source recorded on the death event.
3. If the source belongs to the open trainee trajectory, distribute that amount
   across the source's weighted origins.
4. Omit the relocated amount from the death/effect frame.
5. If the death has no valid trainee causal source, preserve existing
   non-routeable behavior and diagnostics.

Do not apportion the kill reward among all earlier damaging projectiles in this
feature. That would require a separate per-origin damage-contribution design.

## Staged Reward Pipeline

### Required State

Each open reward trajectory stores, in packed or array-backed form:

- one observation per decision frame;
- one selected action index per decision frame;
- one mutable immediate component vector per decision frame;
- frame/sample lookup for causal origins;
- combat-episode boundary metadata;
- trajectory identifier and open/closed state; and
- diagnostics such as missing provenance and routed-event counts.

Do not retain live ship references or Python float tuples for every staged
observation. The observation payload is large and must use float32-compatible
packed storage.

### Per-Frame Processing

For every simulation decision frame:

1. Create the decision and append a staged sample before the simulation outcome
   is routed, so same-frame action effects can resolve their origin.
2. Calculate ongoing rewards and non-relocated immediate components normally.
3. For every routeable positive event, add its component amount to its origin
   frame or weighted origin frames.
4. Do not add the relocated component to the effect frame.
5. Record combat boundaries without finalizing the reward trajectory on
   enemy-only death.

### Finalization

On trainee death or simulation disposal:

1. Close the trajectory to future causal updates.
2. Compute discounted component sums for every staged decision using the
   existing gamma and exact cutoff semantics.
3. Convert component sums to weighted scalar returns using the existing reward
   weights.
4. Produce immutable mature/replay-transfer samples.
5. Construct delayed per-combat-episode metrics from finalized samples.
6. Release staged observations, component arrays, origin lookup state, and
   invalidated credit.

The finalization algorithm should be linear in trajectory length times reward
component count. Do not perform an `O(discount_horizon)` update for every Dogi
or marine effect if a single backward/rolling finalization pass can produce the
same result.

The existing cutoff must be preserved exactly. A simple unbounded recurrence
that includes rewards beyond the cutoff is not equivalent.

### Replay Commit Timing

Finalized samples may be appended to the main replay buffer immediately after
their trajectory closes because batch optimization has not started. Open
trajectory samples must remain outside immutable replay.

Prior-batch replay entries must never be changed.

### Worker Transfer

The recommended worker-process behavior is:

- stage the open trajectory in the simulation worker;
- return no causal samples for that trajectory on ordinary frame responses;
- finalize on trainee death or window completion;
- transfer finalized samples in bounded chunks; and
- append them to the main replay buffer before batch optimization.

If a different protocol uses stable sample IDs and adjustment messages, it must
prove that:

- adjusted samples cannot be evicted before their last update;
- no adjusted sample can be optimized early;
- updates are idempotent and ordered; and
- worker/main failure cannot partially apply one reward event.

Worker-local staging is preferred because it has fewer ordering and eviction
failure modes.

## Metrics And Episode Reporting

Combat-episode outcomes are known immediately, but sample returns are not final
until the containing reward trajectory closes.

The implementation must not report fabricated zero returns for an enemy-death
episode merely because its samples remain staged.

Recommended design:

- record an internal pending combat-episode boundary containing start frame,
  end frame, terminal reason, win/loss/draw, kills, and deaths;
- reset exploration immediately when the combat boundary occurs;
- delay the completed `TrainingEpisodeResult` until trajectory finalization;
- group finalized samples by the combat episode containing their decision
  frame; and
- calculate mature count, average return, and component totals from that group.

A sample assigned to an earlier combat episode may intentionally contain
discounted reward placed in a later enemy life. This follows the rule that
enemy death is not a reward terminal.

Batch totals and UI status must include finalized samples exactly once.
Progress displays may show finalized totals only. If replay occupancy includes
staged samples for presentation, it must label or calculate that count
consistently without claiming they are sampleable replay entries.

## Memory And Performance Requirements

`OBSERVATION_INPUT_SIZE` is currently 621. Approximate packed observation
storage is:

- about 2.8 MiB for 1,200 frames;
- about 28.4 MiB for 12,000 frames.

Actions, component vectors, indexes, Python/container overhead, IPC copies, and
multiple workers add to those figures.

Requirements:

- Stage observations as float32-compatible packed arrays or buffers.
- Bound staging by one active trajectory per simulation window.
- Release finalized staging promptly.
- Avoid holding an entire multi-round batch in every worker when closed
  trajectories can be transferred earlier.
- Chunk large worker transfers to avoid oversized pipe messages and long
  scheduler stalls.
- Add diagnostics for peak staged frames and bytes.
- Benchmark default and maximum frame-limit configurations with coordinated
  workers.

## Compatibility And Migration

- No reward configuration schema change is required.
- Existing model checkpoints remain loadable.
- Replay is intentionally not restored from checkpoints, so a process restart
  begins with no stale replay samples using the former timing semantics.
- Existing models may continue training, but their learned target scale and
  temporal semantics differ from the new targets.
- Fresh training or a new model slot is recommended for comparative evaluation.
- Reward weights may require retuning because repeated long-lived effects are
  concentrated on fewer origin samples.
- Huber loss reduces but does not eliminate the risk from larger, higher-
  variance launch-frame targets.

## Diagnostics And Shadow Mode

Before changing training targets, compute proposed causal placement alongside
the existing pipeline without inserting the proposed samples into replay.

Shadow diagnostics should include:

- routed events by component and ability;
- missing-provenance events by ability;
- effects crossing one or more enemy deaths;
- effects rejected because the trainee trajectory closed;
- mean, percentile, and maximum return by component and action;
- old versus proposed return deltas;
- peak staged frames and bytes;
- trajectory length distribution; and
- finalization and IPC timing.

Shadow calculation must not alter gameplay RNG, action selection, replay,
optimization, model weights, or user-visible reward totals unless explicitly
shown as diagnostic output.

## Feature Gating

Implementation must keep the reward behavior selectable internally while the
phases are being validated. A user-facing setting is not required.

Recommended modes:

- `legacy`: current effect-frame placement and current terminal semantics;
- `shadow`: train with legacy samples while calculating causal samples and
  diagnostics without replay insertion; and
- `causal`: train with the completed semantics in this specification.

`legacy` remains the production default through Phases 1 through 6. New
semantics may be enabled explicitly in focused tests and controlled evaluation
runs during those phases. Phase 7 is the only phase that may make `causal` the
default.

Do not persist a temporary development-mode flag into model metadata unless a
later compatibility decision requires it. Tests should inject the mode through
the reward-pipeline or orchestration configuration boundary.

## Implementation Phases

### Phase 1: Provenance Contracts And Shadow Event Audit

Goal: establish causal identity without changing any trained target.

Work:

- Add trajectory IDs and causal-credit contracts.
- Bind full-weight origins to successfully committed spawned trainee abilities.
- Add generic credit inheritance for derived objects.
- Add the parent-owned `autonomous_fire` origin hook for trainee Chmmr
  satellite lasers in shadow mode.
- Audit crew-loss, debuff, object-removal, and death sources.
- Preserve the destroying source through permanent launched-crew-loss
  accounting.
- Add a distinct release event/hook with affected-object identity.
- Track missing provenance and cross-enemy-death effects.
- Compute causal event destinations in shadow mode only.

Acceptance:

- Existing replay samples and returns remain byte/value compatible within
  current floating-point tolerance.
- Covered built-in abilities report zero unexpected missing provenance.
- Chenjesu shards inherit their crystal's press credit in tests.
- Dogis and marines retain one origin across enemy replacement in tests.
- A shadow Chmmr satellite-laser origin identifies the parent ship's current
  decision frame, not the satellite's creation frame.
- Destruction of an enemy launched-crew unit exposes the same trainee destroying
  source to both object-removal and crew-loss attribution.
- Closed trainee origins are rejected in tests.
- No gameplay test changes except additions validating inert metadata.

### Phase 2: Staged Trajectory Engine With Legacy Semantics

Goal: replace incremental immutable sample commitment with a trajectory staging
engine while reproducing current reward placement and terminal behavior.

Work:

- Add packed staged trajectory storage.
- Implement exact cutoff-aware finalization.
- Integrate independent and coordinated in-process paths.
- Integrate worker-local staging and bounded final sample transfer.
- Preserve current effect-frame placement and current terminal rules during
  this phase.
- Compare finalized samples against the old `RollingReturnPipeline`.

Acceptance:

- Deterministic fixtures produce the same observations, actions, component
  values, returns, frame IDs, and terminal-truncation flags as the old pipeline.
- Independent, coordinated, and worker-process paths produce equivalent
  samples.
- No optimization starts with an open staged trajectory.
- Default and maximum-window memory measurements are recorded.
- Full training and coordinated-training test suites pass.

### Phase 3: Separate Combat And Reward Boundaries

Goal: make trainee life the reward trajectory while preserving combat episode
statistics.

Work:

- Treat enemy-only death as nonterminal for reward staging.
- Close staging on trainee death, simultaneous death, timeout/window end, or
  simulation disposal.
- Add pending combat-episode boundary records.
- Delay sample-derived episode metrics until trajectory finalization.
- Keep exploration reset and enemy replacement behavior at combat boundaries.
- Implement these semantics under causal/shadow evaluation while production
  training remains in legacy mode.

Acceptance:

- A trajectory spans multiple enemy replacements when the trainee survives.
- Trainee death finalizes exactly once and starts a new trajectory.
- Simultaneous death processes all same-frame rewards before closure.
- Wins, losses, draws, kills, and deaths remain correct.
- Per-episode sample counts and component totals equal regrouped finalized
  samples.
- No sample or component is counted twice across combat episodes.

### Phase 4: Enable Long-Lived Ability Routing

Goal: enable causal placement first for the clearest delayed abilities.

Initial covered abilities:

- Chenjesu A2 Dogi
- Orz A3 marine
- Kzer-Za A2 fighter where it produces covered positive effects
- Syreen persistent crew objects where they produce covered positive effects

Work:

- Route enemy debuff, enemy crew-loss, and enemy-object-kill rewards to launch
  origins.
- Route lethal kill reward to the final causal source origin.
- Keep other incoming penalties effect-timed.
- Remove effect-frame copies of relocated components.
- Enable diagnostics in production mode.
- Make the routing available to causal and shadow evaluation; do not make it
  the production default before Phase 7.

Acceptance:

- Multiple Dogi debuffs accumulate at the initial launch.
- Marine boarding, repeated crew loss, and lethal kill use the same launch.
- Permanent trainee-caused destruction of an enemy crew-bearing fighter retains
  both `Kill enemy object` and `Enemy loses crew` at the destroying source's
  origin.
- Natural loss of a crew-bearing fighter does not fabricate trainee credit.
- Natural or externally caused permanent loss of trainee launched crew uses
  the launched unit's origin in live causal mode.
- Parent friendly fire assigns trainee launched-crew loss to the later spawn,
  with equal stamps split evenly.
- Closed selected launched-crew provenance omits the loss component.
- Frames between launch and effect receive no relocated component.
- Frames before launch receive correct gamma propagation.
- Effects across enemy replacement retain the original launch.
- Effects after trainee death cannot update the old or replacement trajectory.
- Raw component totals are conserved except where existing source factors apply.

### Phase 5: Enable General Projectile And Derived-Effect Routing

Goal: apply causal placement to ordinary trainee projectiles and their children.

Work:

- Enable routing for covered positive components from all built-in projectile,
  laser, special-object, and ability-area sources.
- Complete the derived-object provenance audit.
- Enable the special parent-owned firing-frame origin for each trainee Chmmr
  satellite laser.
- Preserve legitimate non-routeable/environmental fallback behavior.
- Validate final-source ship-kill and enemy-object-kill attribution for direct
  and derived sources.
- Keep legacy training as the default while causal mode is evaluated.

Acceptance:

- Every built-in ship has focused or parameterized provenance coverage.
- A normal projectile launched at `-5` and hitting at `0` matches the normative
  timeline exactly.
- Chenjesu shard damage resolves to the crystal launch.
- Fighter child weapons resolve to the fighter's original launch action.
- Each trainee Chmmr satellite laser resolves to a fresh origin on its firing
  frame in the parent Chmmr trajectory; consecutive shots do not share origins.
- Ship kills, crew loss, ordinary object kills, and existing fractional enemy
  Chmmr-satellite rewards caused by a trainee satellite laser all use that
  laser's firing-frame origin.
- Satellite-body contact does not reuse a laser origin.
- No covered ability reports missing provenance in deterministic integration
  runs.
- No relocated reward remains on an effect frame.

### Phase 6: Enable Press/Release Split Attribution

Goal: reward both creation and release decisions for abilities materially
controlled by release.

Work:

- Enable Chenjesu A1 50/50 press/release future credit.
- Enable Kohr-Ah A1 per-saw press/latest-effective-release credit.
- Enable Melnorme A1 held/released credit, unless an explicit documented
  deviation is approved.
- Ensure release uses the complete current action index.
- Keep `Use A1` press-only.
- Keep the completed split behavior gated behind causal/shadow mode until the
  Phase 7 enablement decision.

Acceptance:

- Press-only hits receive full press placement.
- Post-release effects use two origins whose weights sum to one.
- The split-origin discount example in this spec is reproduced exactly.
- Releasing with no affected object creates no reward origin.
- Repeated Kohr-Ah releases change only future attribution.
- Previously recorded effects are never rewritten by a later release.

### Phase 7: Shadow Comparison, Tuning, And Default Enablement

Goal: validate learning-target quality before making causal routing the only
training behavior.

Work:

- Run old and proposed return calculations side by side on representative
  batches.
- Compare target distributions per ship, action, and reward component.
- Measure Dogi/marine concentration and extreme return values.
- Benchmark memory, finalization time, IPC, and batch throughput.
- Retune reward defaults only through a separate explicit decision; do not hide
  tuning inside this feature.
- Remove or demote legacy code only after parity and rollout evidence is saved.

Acceptance:

- No unexplained reward loss or duplication remains.
- Missing-provenance counters are zero for covered built-in abilities.
- Peak memory and throughput remain acceptable at configured maximum windows
  and supported worker counts.
- Target percentiles and maxima are reviewed for long-lived abilities.
- Independent and coordinated training remain deterministic under seeded tests.
- Full test suite passes.
- A progress document records measurements, deviations, and the final enablement
  decision.

## Required Test Matrix

### Reward Mathematics

- Single origin, delayed effect, exact gamma powers.
- No credit after origin and before effect.
- Existing cutoff applied relative to origin.
- Effect occurring later than the cutoff still fully credits its origin.
- Multiple effects sharing one origin.
- Multiple weighted origins.
- No double counting.

### Lifecycle

- Enemy death without trainee death.
- Trainee death without enemy death.
- Simultaneous death.
- Multiple enemy replacements in one trainee life.
- Window timeout with live objects.
- Stop during an open trajectory.
- Stale object effect after trainee replacement.

### Ability Attribution

- Dogi repeated debuffs.
- Orz marine boarding, damage, death, and kill.
- Ordinary projectile crew loss and kill.
- Ordinary projectile enemy-object destruction.
- Enemy crew-bearing fighter destruction producing both object-kill and
  permanent crew-loss components at the destroying source origin.
- Crew-bearing fighter safe return, natural expiration/loss, and destruction by
  a non-trainee source.
- Chenjesu primary crystal and shards.
- Kohr-Ah saw before and after release.
- Multiple Kohr-Ah saws affected by one release.
- Repeated Kohr-Ah releases.
- Melnorme held and released projectile.
- Fighter or persistent-object child weapons.
- Chmmr satellite laser ship kill, crew loss, enemy-object kill, and consecutive
  firing-frame origins owned by the parent ship.
- Chmmr satellite laser damage/destruction of an enemy Chmmr satellite using
  existing fractional amounts at the laser firing origin.
- Opponent Chmmr satellite laser, stale post-trainee-death laser, replacement-
  enemy hit, same-frame fire-and-hit, and satellite-body-contact fallback.
- Limpet and confusion source propagation.
- Environmental and opponent self-damage fallback.

### Replay And Execution Paths

- Independent training.
- Coordinated in-process training helper.
- Coordinated simulation module.
- Process worker frame and window completion protocol.
- Replay capacity and deterministic eviction after staged commit.
- Batch optimization ordering.
- Packed observation fidelity.
- Large-trajectory chunk transfer.

### Reporting

- Immediate combat outcome recording.
- Delayed finalized episode return.
- Samples grouped by their decision-frame combat episode.
- Cross-enemy-boundary return included intentionally.
- Batch component totals counted once.
- UI/progress replay size semantics while samples are staged.

## Risk Assessment

Overall implementation risk is medium-high, approximately `7/10` before phased
validation.

### High Risk

- Silent missing provenance can train plausible but incorrect targets.
- Accidental impact-plus-origin double counting can inflate rewards.
- Losing the destroying source between object removal and launched-crew
  accounting can route two consequences of one fighter kill inconsistently.
- Concentrating many Dogi or marine effects on one launch sample increases
  target variance and may change learned policy substantially.
- Combat episode metrics can become incorrect if they are finalized before the
  reward trajectory.

### Medium-High Risk

- Worker-process streaming must change without allowing early optimization or
  excessive IPC stalls.
- Press/release attribution affects complete action indices and multiple live
  Kohr-Ah objects.
- Chmmr satellite lasers create parent-owned origins during autonomous object
  updates, so decision/update ordering and same-frame hits must be exact.
- Several training entry points contain parallel reward lifecycle logic and can
  drift if updated inconsistently.

### Medium Risk

- Packed staging can consume tens of megabytes per maximum-length active
  worker.
- Existing trained models remain loadable but were calibrated to different
  temporal targets.
- Progress metrics become delayed relative to enemy deaths.

### Low Risk

- Gameplay behavior, if training hooks remain observational and metadata-only.
- Checkpoint file compatibility, because reward configuration keys and model
  architecture do not change.

### Primary Mitigations

- Shadow mode before target changes.
- Staged-pipeline parity before causal routing.
- Per-ability provenance tests.
- Paired object-removal/launched-crew-loss source tests and Chmmr autonomous-fire
  ordering tests.
- Missing-provenance and double-count diagnostics.
- Worker-local staging with bounded chunk transfer.
- Exact cross-path deterministic tests.
- Measured rollout on new model slots before continuing important existing
  models.

With the phased rollout and acceptance gates above, expected residual deployment
risk is approximately `4/10`. The dominant residual risk is learning-quality
change rather than gameplay instability or data-format corruption.

## Completion Definition

The feature is complete only when:

- all seven phases meet their acceptance criteria;
- trainee life, not enemy life, is the reward trajectory;
- covered positive ability effects are placed exclusively at causal origins;
- press/release abilities follow the specified split rules;
- replay entries are finalized before insertion and never updated after
  optimization;
- all execution paths use equivalent semantics;
- diagnostics show no unexplained missing provenance or double counting;
- memory and throughput measurements are documented; and
- the full automated test suite passes.
