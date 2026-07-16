# Event Reward Update Progress

This record tracks implementation and gate evidence for
`EVENT_REWARD_UPDATE_SPEC.md`. Phases are enabled only after every preceding
phase gate passes.

## Baseline

Status: Verified.

- Worktree was clean before implementation.
- Existing architecture uses `RollingReturnPipeline` with incremental replay
  insertion and combat-terminal pipeline reset in independent, coordinated
  in-process, coordinated simulation, and process-worker paths.
- Baseline full suite: `runalltests.cmd` — 1,003 tests passed in 12.363 seconds.

## Phase 1: Provenance Contracts And Shadow Event Audit

Status: Implemented and verified.

- Added immutable, validated `RewardOrigin` and `AbilityRewardCredit` contracts,
  globally unique trajectory IDs, reward modes, origin kinds, and causal
  diagnostics in `src/training/causal_credit.py`.
- Added ledger-owned open trajectory and current-decision identity, committed
  trainee-action binding, autonomous Chmmr-fire binding, event-time credit
  snapshots, missing/closed/cross-death diagnostics, and a distinct release
  event containing affected-object and complete action-index identity.
- Added generic constructor-time credit inheritance plus explicit Chenjesu
  crystal-to-shard inheritance, which remains valid after parent detachment.
- Added fresh parent-trajectory `autonomous_fire` provenance to each trainee
  Chmmr satellite laser.
- Audited limpet and confusion debuffs so their events retain the ability source.
- Preserved the actual destroying source through Kzer-Za fighter and Orz marine
  permanent launched-crew-loss callbacks.
- Added decision-context integration to independent, coordinated in-process,
  coordinated simulation, and worker-backed coordinated paths.
- Focused Phase 1 gate: 8 provenance tests passed.
- Full Phase 1 regression gate: `runalltests.cmd` — 1,011 tests passed in
  12.321 seconds. Existing reward/replay calculations remain unchanged.

## Phase 2: Staged Trajectory Engine With Legacy Semantics

Status: Implemented and verified.

- Added `StagedTrajectoryPipeline` with array-backed float32 observations,
  int32 actions, int64 frame IDs, mutable float64 component vectors, origin
  frame lookup, and no live gameplay references in staged samples.
- Implemented linear cutoff-aware backward finalization preserving legacy
  gamma, exact horizon length, mature sample frame IDs/counts, and terminal
  truncation flags.
- Retained `RollingReturnPipeline` as the parity oracle while all production
  independent/coordinated paths now use packed staging.
- Staged every selected decision before simulation updates so same-frame future
  origins can resolve an existing sample.
- Integrated independent, coordinated in-process, coordinated simulation, and
  process-worker paths. Open worker trajectories return no replay samples;
  finalized transfers are split into chunks of at most 256 samples.
- Stop behavior finalizes completed frames before aborting; replay optimization
  remains downstream of completed/committed windows.
- Deterministic rolling-versus-staged fixtures match observations, actions,
  components, returns (within 1e-12), frame IDs, counts, and truncation flags.
- Packed staging measurements (571 observation floats plus actions, IDs, and 19
  component values per frame): 1,200 frames = 2.802 MiB in 0.043713 seconds;
  12,000 frames = 28.015 MiB in 0.459070 seconds.
- Focused training/orchestration/replay gate: 135 tests passed.
- Full Phase 2 regression gate: `runalltests.cmd` — 1,013 tests passed in
  12.287 seconds.

## Phase 3: Separate Combat And Reward Boundaries

Status: Implemented and verified.

- Added validated internal `legacy`, `shadow`, and `causal` reward modes to the
  orchestration boundary; production remains `legacy` for this phase.
- In shadow/causal lifecycle mode, enemy-only death records a combat boundary,
  resets exploration, and retains the same open trainee-life staged pipeline.
- Trainee death, simultaneous death, timeout/window disposal, and compatibility
  simulations that cannot replace ships close/finalize exactly once.
- Added `PendingCombatEpisode` and delayed final reporting that groups immutable
  samples by decision-frame combat interval after trajectory finalization.
- Integrated lifecycle/reporting semantics in independent, coordinated fixed
  window, in-process coordinated runtime, coordinated simulation module, and
  process-worker paths.
- Process workers emit no fabricated zero-return terminal episode at enemy-only
  death; finalized results and samples arrive together when the trajectory or
  window closes.
- Tests cover enemy replacement spanning one trajectory, unique post-trainee-
  death trajectory IDs, simultaneous-death frame processing, cross-boundary
  discounted return, correct wins, and exact-once sample grouping.
- Full Phase 3 regression gate: `runalltests.cmd` — 1,018 tests passed in
  15.459 seconds.

## Phase 4: Enable Long-Lived Ability Routing

Status: Implemented and verified.

- Added causal relocation for the existing calculated `Enemy loses crew`,
  `Debuff enemy`, `Kill enemy object`, and final-source `Kill enemy` amounts
  produced by Dogis, Orz marines, Kzer-Za fighters/lasers, and Syreen crew.
- Relocation validates immutable event-time credit against the current open
  trainee trajectory, adds weighted origin deposits, and removes the effect
  copy only after every origin resolves. Invalid/missing sources keep one
  impact fallback and update diagnostics.
- Shadow mode keeps legacy training vectors while maintaining a separate causal
  component vector; legacy mode remains the production default.
- Preserved existing source factors, death-credit factors, and fractional Chmmr
  amounts by relocating the already-calculated component amount.
- Tests cover repeated Dogi effects, marine boarding/crew/kill, intentional
  fighter object-kill plus permanent crew-loss components, natural fighter
  loss, gamma propagation before launch, no intermediate-frame deposits,
  cross-enemy replacement routing, raw-total conservation, and post-trainee-
  death rejection.
- Full Phase 4 regression gate: `runalltests.cmd` — 1,025 tests passed in
  14.945 seconds.

## Phase 5: Enable General Projectile And Derived-Effect Routing

Status: Implemented and verified.

- Expanded causal relocation from the Phase 4 long-lived set to every built-in
  projectile, laser, special-object, and ability-area source whose event-time
  credit belongs to the open trainee trajectory.
- General routing distinguishes expected trainee ability provenance from
  legitimate ship/contact/environment/opponent fallback using source ownership
  and source type; Chmmr satellite-body contact is explicitly non-routeable.
- Completed explicit inheritance for Syreen crew created by the song effect in
  addition to Chenjesu shards; generic constructor inheritance covers fighter
  and persistent-object child weapons whose gameplay parent is the source.
- Enabled fresh parent-owned `autonomous_fire` origins for every trainee Chmmr
  satellite laser in causal/shadow routing, including crew loss, ship kill,
  object kill, and existing fractional enemy-satellite components.
- Tests cover the normative ordinary-projectile timeline, effect latency beyond
  the cutoff, catalog-wide controlled-ability binding, derived sources,
  consecutive/firing-frame Chmmr identity, fractional Chmmr effects, stale and
  opponent/body fallbacks, object final-source attribution, and zero relocated
  effect copies.
- Full Phase 5 regression gate: `runalltests.cmd` — 1,038 tests passed in
  12.257 seconds.

## Phase 6: Enable Press/Release Split Attribution

Status: Implemented and verified.

- Added mode-gated future 50/50 press/latest-effective-release credit while
  retaining immutable event-time snapshots, so later releases never rewrite
  effects already recorded.
- Release identity uses the current complete action index and a distinct
  `action_released` event; it does not create a second `Use A1` reward.
- Chenjesu reports only the live latched crystal affected by the release, and
  shards inherit its current split credit. Repeated ineffective release edges
  no longer reuse a pending fragmentation request.
- Kohr-Ah reports every live saw commanded by a release. Each saw retains its
  own press half while subsequent effective releases replace only its future
  release half.
- Melnorme reports only a live held projectile that materially changes to its
  launched state; release without one produces no release origin.
- The normative split example is reproduced exactly: a press-frame sample
  receives `0.5 * R + gamma**3 * 0.5 * R`, intervening pre-release samples
  receive only the release half's propagation, and post-release frames receive
  no relocated copy.
- Focused release and ship-mechanics gate: 47 tests passed.
- Full Phase 6 regression gate: `runalltests.cmd` — 1,052 tests passed in
  15.428 seconds.

## Phase 7: Shadow Comparison, Tuning, And Default Enablement

Status: Implemented and verified.

- Shadow mode now finalizes aligned legacy-training and proposed-causal sample
  sets without inserting the causal set into replay. It exposes exact compact
  mean, p50, p95, p99, minimum, maximum, and maximum-absolute summaries for
  total return, component, complete action index, and component/action, plus
  old-versus-proposed deltas.
- Finalized independent and coordinated window results expose causal reward
  diagnostics. Diagnostics include routed/missing/cross-death/closed counters,
  packed staging peaks, trajectory lengths, finalization time, and shadow
  comparisons. Coordinated process metrics now also measure command/result IPC
  serialization and transfer time.
- A seeded 300-frame shadow catalog run completed 7,500 frames across all 25
  ships in 5.618 seconds (1,335.1 frames/second), observed 35 routed covered
  effects, and reported zero missing provenance. The asset-less headless
  benchmark's active Umgah A1 cycle hit its existing empty-anchor limitation,
  so that one measurement used no-action fallback; Umgah and every other
  controlled ability remain covered by catalog-wide commit/provenance tests.
- Seeded 1,200-frame representative shadow runs reported zero missing and zero
  closed-trajectory provenance events across Earthling (5 routed effects),
  Chenjesu (8), and Orz (58). Raw immediate component conservation and absence
  of effect-frame duplicates remain verified by focused tests.
- Target review, using raw component returns:
  - Earthling `Enemy loses crew`: old/new p99 5.491/5.219 and maximum
    6.195/5.888; causal A1 action index 6 maximum 5.888.
  - Chenjesu Dogi `Debuff enemy`: old/new mean 0.599/0.085, p99
    3.606/0.895, and maximum 4.014/7.049; the causal maximum is concentrated on
    launch action index 12. The lower mean is expected because frames between
    launch and effect no longer receive impact-timed propagation.
  - Orz marine `Enemy loses crew`: old/new p99 10.525/8.975 and maximum
    11.374/10.023; combined action index 18 maximum 10.023. Marine `Debuff
    enemy` old/new p99 is 5.324/3.878 and maximum is 6.006/4.331.
- Packed pipeline microbenchmarks with 571-float observations:
  - 1,200 frames: causal 2.802 MiB peak staging and 14,051 frames/second with
    0.037-second finalization; shadow 2.975 MiB and 8,194 frames/second with
    0.097-second
    dual-target finalization.
  - 12,000 frames: causal 28.015 MiB peak staging and 13,607 frames/second with
    0.383-second finalization; shadow 29.755 MiB and 7,749 frames/second with
    1.023-second
    dual-target finalization.
  - Bounded replay-transfer IPC used 5 chunks/2.670 MiB/0.0049 seconds for
    1,200 samples and 47 chunks/26.699 MiB/0.0426 seconds for 12,000 samples.
- End-to-end coordinated worker measurements, including simulation commands,
  observation replies, finalization, and replay transfer:
  - One worker: 1,200 frames in 0.869 seconds (1,381 frames/second, 0.050-second
    IPC); 12,000 frames in 8.104 seconds (1,481 frames/second, 0.499-second IPC).
  - Four workers: 4,800 default-limit frames in 0.920 seconds (5,218 aggregate
    frames/second, 11.90 MiB combined peak staging); 48,000 maximum-limit frames
    in 11.266 seconds (4,261 aggregate frames/second, 106.79 MiB observed
    combined peak staging, 1.653-second aggregate IPC). This end-to-end run
    preceded lazy shadow allocation; the current theoretical all-workers-live
    causal packed maximum is 112.06 MiB rather than 119.02 MiB. Closed
    trajectories streamed
    7,068 samples before window finish and the remaining 40,932 in 161 bounded
    finish chunks.
- Added exact seeded target-equality tests for independent and coordinated
  causal execution. Existing process-worker, replay ordering/capacity, chunking,
  and staged observation fidelity tests continue to pass.
- No reward defaults were retuned. Causal reward semantics are now the
  `TrainingOrchestrationConfig` default; `legacy` and `shadow` remain internal
  parity/evaluation modes, and the temporary mode is not persisted in model
  metadata.
- Final Phase 7 regression gate: `runalltests.cmd` — 1,057 tests passed in
  15.272 seconds.

## Deviations And Final Enablement Decision

- No normative reward-semantic deviations were taken; Melnorme press/release
  attribution is included.
- No reward-weight tuning is bundled with this feature.
- Final decision: enable `causal` as the production orchestration default based
  on zero covered missing-provenance diagnostics, conserved raw components,
  reviewed long-lived target extremes, deterministic execution, acceptable
  maximum-window memory/throughput, and the passing full regression gate.
- Keep `legacy` as the staged parity oracle and `shadow` as the non-training
  comparison path; neither mode is written to model metadata.

## Post-Enablement: Lazy Shadow Execution

Status: Implemented and verified.

- Shadow component storage is now allocated only for explicit `shadow` mode.
  Causal and legacy modes do not allocate, initialize, copy, relocate, inspect,
  finalize, or summarize a shadow component vector.
- Causal routing continues to update only the training component vector;
  legacy mode skips causal/shadow routing entirely.
- Calling shadow inspection outside shadow mode fails explicitly instead of
  silently activating shadow work.
- At 12,000 frames this removes 1.740 MiB from every open causal worker
  trajectory (29.755 MiB to 28.015 MiB). Four simultaneously live maximum
  windows therefore use 112.06 MiB of packed staging rather than 119.02 MiB.
- Focused reward/provenance gate: 22 tests passed.
- Full post-enablement regression gate: `runalltests.cmd` — 1,057 tests passed
  in 12.352 seconds.
