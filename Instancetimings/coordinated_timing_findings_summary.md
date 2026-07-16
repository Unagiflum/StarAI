# Coordinated Training Timing Findings

Generated from the 25 `*.coordinated.csv` files produced by the coordinated
training measurement run on July 16, 2026.

## Scope and Data Interpretation

The 25 files are the per-ship views of one coordinated training session. Their
session-level timing fields are duplicated, while ship-specific reward and
outcome fields vary. Each file contains the same three measured coordinated
batches: 89, 90, and 91.

Consequently, this dataset contains three performance observations, not 75
independent observations. Batch 89 includes worker startup; batches 90 and 91
represent steady-state operation.

Each batch contains:

- 25 training instances.
- 30,000 frames per instance.
- 750,000 total instance-frames.
- 750,000 action requests.
- 751,250 messages in each direction between the parent and workers.

## Batch Summary

| Batch | Wall time | Frames/second | Coordinator work | Waiting for workers | Parent IPC | Unattributed | Startup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 89 | 273.884s | 2,738 | 102.666s (37.5%) | 119.648s (43.7%) | 26.673s (9.7%) | 19.425s (7.1%) | 5.471s (2.0%) |
| 90 | 246.381s | 3,044 | 102.802s (41.7%) | 97.515s (39.6%) | 26.788s (10.9%) | 19.276s (7.8%) | 0 |
| 91 | 289.585s | 2,590 | 101.098s (34.9%) | 142.249s (49.1%) | 25.953s (9.0%) | 20.284s (7.0%) | 0 |

Across all three batches, the run processed 2.25 million instance-frames in
809.85 seconds, or approximately 10.0 million instance-frames per hour. This is
equivalent to about 333 complete 30,000-frame instance-batches per hour.

## Measurement Integrity

The timing and transport measurements are internally consistent:

- Parent-received byte counts exactly equal aggregate worker-sent byte counts.
- Parent-sent byte counts exactly equal aggregate worker-received byte counts.
- Message counts match at both endpoints.
- The parent IPC endpoint components sum exactly to the reported parent IPC
  cost.
- The parent and worker endpoint components sum to the reported aggregate IPC
  endpoint time.
- Coordinator wall-time attribution closes to within one microsecond after the
  explicit unattributed bucket is included.

This indicates that the endpoint timing and byte accounting are functioning
correctly.

The aggregate `IPC Endpoint Seconds` value, 187-213 seconds, must not be added
directly to batch wall time. It sums time across the parent and all 25 workers,
including overlapping and blocked time. The directly additive parent endpoint
cost is approximately 26 seconds per batch.

## Main Findings

### 1. Batch-to-batch slowdown is worker-side

Batch 91 took 43.204 seconds longer than batch 90. Over the same comparison:

- Waiting for workers increased by 44.734 seconds.
- Coordinator work decreased by 1.703 seconds.
- Parent IPC decreased by 0.835 seconds.
- Coordinator unattributed time increased by 1.008 seconds.

The entire slowdown is therefore explained by worker-side variation rather
than additional coordinator inference or parent IPC work.

Aggregate worker active time increased from 1,359.0 CPU-seconds in batch 90 to
1,517.2 CPU-seconds in batch 91. The largest increases were:

- Simulation: +65.8 seconds.
- Observation encoding: +45.6 seconds.
- Post-simulation reward processing: +33.0 seconds.
- Reward decision processing: +10.9 seconds.

Within simulation, object updates increased by 30.4 seconds and collision work
increased by 24.6 seconds. Replay insertion and reward flushing together
increased by 28.8 seconds.

### 2. Coordinator inference is a large, stable cost

The main coordinator phases per batch were:

- Trainee inference: 66.3-68.4 seconds.
- Opponent inference: 28.2-28.8 seconds.
- Observation preparation: approximately 2.7 seconds.
- Optimization: 3.0-3.3 seconds.
- Saving: zero except for one 0.53-second save.

Inference therefore creates a stable floor of approximately 95-97 seconds per
batch. The timed inference calls include CPU preparation, device transfer, GPU
execution, and the synchronized return of action indices. The measurements do
not separately identify host-to-device transfer, kernel execution, and
device-to-host transfer.

Approximately 47-48% of individual trainee actions were exploratory. The
current selection path avoids inference for those individual actions, but a GPU
inference call is still required on every coordinated frame because at least
one of the 25 records requires a greedy action.

### 3. Parent IPC is significant but not dominant

Each batch transferred approximately:

- 222-224 MB from the parent to workers.
- 4.64-4.73 GB from workers to the parent.
- 751,250 messages in each direction.
- 295-298 bytes per parent command on average.
- 6.18-6.30 KB per worker result on average.

The parent spends approximately 26 seconds per batch serializing, sending,
receiving, and deserializing this traffic. This is 9-11% of wall time. It is a
meaningful optimization target, but it is not the dominant cost and remained
stable when batch 91 became slower.

Worker send-transfer time increased from 101.0 aggregate seconds in batch 90 to
121.8 seconds in batch 91. This value overlaps across workers and includes pipe
backpressure while the parent is occupied. It is more appropriately treated as
a symptom of synchronization and worker variability than as 20.8 seconds of
additional critical-path transfer overhead. Parent endpoint time actually fell
during the same batch.

### 4. Worker simulation and observation encoding dominate worker CPU work

In the steady-state batches, aggregate worker active time was divided
approximately as follows:

- Simulation: 49.5-50.4%.
- Observation encoding: 26.9-27.1%.
- Reward decision processing: 11.2-11.7%.
- Subsequent reward and replay processing: 8.7-9.9%.
- Unattributed worker activity: approximately 2.2%.

Collision handling was the largest measured simulation sub-phase. Reward and
replay work is measurable, particularly in batch 91, but it is not the primary
steady-state bottleneck.

### 5. Mature replay samples are still attached to frame responses

Workers currently call `collector.drain_pending()` when constructing each
`FrameSteppedResult`. Mature replay samples can therefore be included in
individual frame responses.

The placement of these samples is avoidable, but the underlying packed replay
observations must eventually reach the parent. Merely delaying them until the
end of a window would reduce ordinary frame-result size without eliminating
most of the transferred bytes, and could create large end-of-window bursts.

Potential alternatives include:

- Accumulating multiple replay samples into a contiguous transfer block.
- Sending replay chunks at a controlled interval rather than every frame.
- Using a shared-memory replay-transfer buffer.

These add progressively more complexity. Because the entire directly additive
parent IPC cost is only about 26 seconds and includes required action and
observation traffic, replay transfer should be changed only after an A/B
prototype demonstrates a material gain.

### 6. Seven to eight percent of coordinator wall time remains unattributed

Coordinator unattributed time is stable at 19-20 seconds per batch. It includes
coordinator loop bookkeeping, status checks, result handling outside existing
buckets, scheduling, and the configured periodic yield.

The coordinator requests a 0.25 ms sleep every four coordinated frames. There
are 7,500 such yields in a 30,000-frame batch. An idle local measurement
projected approximately 3.75 seconds for those sleeps, but duration under load
can differ. The current data cannot divide the remaining residual precisely
between yielding and Python orchestration overhead.

## Comparison With the Older Timing Report

The older report in this folder estimated approximately 47 completed
instance-batches per hour for seven independently coordinated training
instances. This coordinated run achieved approximately 333 instance-batches
per hour.

That is roughly a sevenfold aggregate-throughput improvement. It is not a
controlled comparison because the architecture, instance count, and code have
changed, but it strongly indicates that centralized batched inference and the
current coordinated worker design corrected the former multi-process GPU
contention problem.

## Recommendations

### Priority 1: Investigate worker stragglers and variable CPU work

Worker wait is the largest and most variable critical-path component. Before
changing IPC architecture, collect several more batches and determine whether
the batch 91 increases in observation, object-update, collision, and replay
work recur predictably with battle composition or episode termination.

Useful follow-up analysis would compare per-worker rather than only aggregate
and maximum timings. This would show whether one consistently expensive ship or
opponent controls the frame barrier, or whether the slowest worker changes from
frame to frame.

### Priority 2: Treat inference as the stable optimization target

Trainee and opponent inference together cost approximately 95-97 seconds per
batch. Any inference optimization should preserve the current cross-instance
batching, since exploratory workers and variable battle timelines can otherwise
reduce GPU batch size.

Do not restore inference for exploratory actions merely for simpler control
flow; those individual inferences are currently avoided. Conversely, do not
move exploratory spans entirely into workers without an A/B test, because doing
so can shrink coordinated GPU batches and increase synchronization imbalance.

### Priority 3: Split the coordinator residual only if pursuing another gain

The stable 19-20 second unattributed bucket is large enough to investigate, but
not large enough to justify broad instrumentation changes by itself. A targeted
timing bucket around the periodic yield and another around frame-result
bookkeeping would be sufficient to distinguish intentional yielding from
orchestration overhead.

An A/B run with the configured yield disabled or reduced would establish its
actual throughput cost, but UI responsiveness and process fairness should be
checked during that test.

### Priority 4: Prototype replay batching before redesigning IPC

If IPC optimization is pursued, begin with the smallest reversible experiment:
encode a group of replay observations, actions, and returns as contiguous arrays
and transfer that group at a bounded interval. Measure parent endpoint time,
worker send blocking, total bytes, and batch wall time against this baseline.

Do not move directly to shared-memory replay buffers unless compact chunked
transfer produces a clear gain and IPC remains limiting. The current data puts
the absolute ceiling from eliminating all parent endpoint work at only 9-11%,
and required observation traffic means the realistic gain is lower.

### Priority 5: Gather more steady-state samples before implementation

Only two batches in this dataset exclude startup. The 43-second steady-state
range is large enough that a small optimization could be hidden by battle
variation. At least 10 steady-state batches would provide a more reliable
baseline for detecting changes below roughly 10%.

For normal throughput runs, disable `TRAINING_TIMING_ENABLED` after measurement
so that detailed worker timers and timing payloads do not affect the result.
The regular training CSV will continue to be written independently of this
flag.

## Bottom Line

The timing system is correctly accounting for parent/worker IPC and major
coordinator and worker phases. The present bottleneck is a combination of
stable GPU inference and variable worker completion time. Parent IPC is worth
approximately 10% of wall time, but the measured slow batch was not caused by
IPC. The best next step is to gather more batches and localize worker
stragglers; replay-transfer redesign should proceed only through a measured,
bounded prototype.
