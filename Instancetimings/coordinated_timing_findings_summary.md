# Coordinated Training Timing Findings

Updated from the current coordinated timing data on July 17, 2026.

## Scope and Data Interpretation

The current data comes from two coordinated training sessions:

- The 25 per-ship `*.coordinated.csv` files each contain batches 320 and 321.
  Their session-level timing fields are duplicated; ship-specific training
  results differ. These files therefore contribute two timing observations,
  not 50.
- `coordinated-timing.csv` contains six session-level observations: batches
  322 through 327.

Together the files contain eight independent performance observations.
Batches 320 and 322 include worker startup and are treated as cold-start
batches. Batches 321 and 323 through 327 are the six steady-state observations.

Each batch contains:

- 25 training instances.
- 625 completed rounds across those instances.
- 30,000 frames per instance.
- 750,000 total instance-frames and action requests.
- 751,250 messages in each direction between the parent and workers.

## Batch Summary

`Coordinator work` is observation preparation, trainee and opponent inference,
optimization, and saving. `Parent IPC` is the directly additive parent-side
serialization, transfer, receive, and deserialization cost.

| Batch | Wall time | Frames/second | Coordinator work | Inference | Waiting for workers | Parent IPC | Unattributed | Startup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 265.260s | 2,827 | 113.047s | 106.642s | 105.189s | 25.785s | 15.749s | 5.490s |
| 321 | 241.432s | 3,106 | 112.216s | 106.257s | 87.762s | 25.552s | 15.902s | 0 |
| 322 | 268.541s | 2,793 | 115.503s | 109.577s | 107.250s | 24.812s | 15.385s | 5.590s |
| 323 | 245.301s | 3,057 | 114.295s | 108.662s | 90.136s | 25.296s | 15.575s | 0 |
| 324 | 241.234s | 3,109 | 113.672s | 107.758s | 86.014s | 25.148s | 16.400s | 0 |
| 325 | 231.608s | 3,238 | 103.899s | 98.335s | 86.886s | 25.686s | 15.137s | 0 |
| 326 | 236.644s | 3,169 | 107.247s | 100.571s | 87.770s | 26.492s | 15.135s | 0 |
| 327 | 221.820s | 3,381 | 92.531s | 86.291s | 87.734s | 26.403s | 15.152s | 0 |

The coordinator attribution closes to within one microsecond for every batch.

## Steady-State Cost Breakdown

Across batches 321 and 323 through 327:

| Additive bucket | Mean seconds | Share of wall time |
| --- | ---: | ---: |
| Trainee and opponent inference | 101.312s | 42.9% |
| Waiting for workers | 87.717s | 37.1% |
| Parent IPC | 25.763s | 10.9% |
| Coordinator unattributed | 15.550s | 6.6% |
| Optimization | 3.314s | 1.4% |
| Coordinator observation preparation | 2.683s | 1.1% |
| Saving | 0 | 0% |
| **Batch wall time** | **236.340s** | **100%** |

Steady-state wall time ranged from 221.820 to 245.301 seconds, with a standard
deviation of 8.526 seconds. Aggregate steady-state throughput was approximately
11.42 million instance-frames per hour, equivalent to 381 completed 30,000-frame
instance-batches per hour. Including both cold-start batches lowers throughput
to approximately 369 instance-batches per hour.

## Measurement Integrity

The timing and transport measurements are internally consistent:

- Parent-received byte counts exactly equal aggregate worker-sent byte counts.
- Parent-sent byte counts exactly equal aggregate worker-received byte counts.
- Message counts match at both endpoints.
- The parent IPC endpoint components sum to the directly additive parent IPC
  cost.
- Coordinator work, worker wait, parent IPC, coordinator unattributed time, and
  startup sum to batch wall time within one microsecond.

`IPC Endpoint Seconds` must not be added directly to wall time. It aggregates
time across the parent and all 25 workers, including simultaneous and blocked
time. Only the parent endpoint components are directly additive in the batch
wall-time attribution.

## Main Findings

### 1. Steady-state wall-time variation is inference-side

The clearest comparison is batch 323 versus batch 327:

- Wall time fell by 23.481 seconds.
- Trainee and opponent inference fell by 22.371 seconds.
- Worker wait fell by only 2.402 seconds.
- Parent IPC increased by 1.107 seconds.
- Coordinator unattributed time fell by 0.423 seconds.
- Aggregate worker active time fell by only 1.903 CPU-seconds across all 25
  workers.

Inference therefore explains approximately 95% of the wall-time improvement
between those batches. Across all six steady-state observations, inference had
an 8.3% coefficient of variation while worker wait varied by only 1.6% and
aggregate worker active time by only 0.8%.

Every batch made the same 30,000 batched trainee inference calls. Exploration
also remained close to 19%, so the number of greedy trainee records changed by
less than one percent. The approximately 21% inference-time reduction from
batch 323 to batch 327 is too large to attribute to that change alone.

The inference timer currently includes host preparation, device transfer, GPU
execution, synchronization, and returning action indices. The data cannot yet
identify which part improved. Possible explanations include device/runtime
warm-up, GPU clock or load changes, transfer/synchronization variation, or a
change in actual opponent inference batch composition.

### 2. Cold-start cost is larger than the explicit startup bucket

The two process starts produced nearly identical penalties:

| Comparison | Extra wall time | Explicit startup | Extra worker wait | Extra inference |
| --- | ---: | ---: | ---: | ---: |
| Batch 320 versus 321 | 23.828s | 5.490s | 17.427s | 0.385s |
| Batch 322 versus 323 | 23.240s | 5.590s | 17.114s | 0.916s |

The explicit timer captures process creation, but most of the first-batch
penalty appears as additional worker wait. Cold-start batches should be
excluded from steady-state comparisons in their entirety rather than corrected
only by subtracting `Worker Startup Seconds`.

### 3. Worker CPU work is stable and concentrated in simulation and encoding

Aggregate worker active time averaged 1,202.8 CPU-seconds and varied by less
than one percent across steady-state batches. Its largest measured components
were:

| Worker component | Mean aggregate seconds | Share of worker active time |
| --- | ---: | ---: |
| Simulation | 702.013s | 58.4% |
| Collision processing | 348.843s | 29.0% |
| Observation encoding | 314.763s | 26.2% |
| Object updates | 167.597s | 13.9% |
| Ship inputs | 125.416s | 10.4% |
| Reward decisions | 76.685s | 6.4% |
| Worker unattributed | 50.145s | 4.2% |
| Reward pipeline | 14.126s | 1.2% |

Collision processing is included within simulation, so these rows are not all
additive. If worker CPU optimization is pursued, collision processing,
observation encoding, and object updates are the largest candidates.

However, aggregate worker CPU savings improve wall time only when they shorten
the slowest response at a frame barrier. The present aggregate and maximum
fields do not identify which worker controls that barrier or whether the
straggler identity changes from frame to frame.

### 4. Parent IPC is meaningful but does not explain batch variation

The parent IPC endpoint averaged 25.763 seconds, or 10.9% of steady-state wall
time. Its average components were:

- Send serialization: 5.657 seconds.
- Send transfer: 5.472 seconds.
- Receive transfer: 9.216 seconds.
- Receive deserialization: 5.418 seconds.

Each batch sent approximately 222 MB from the parent and returned 4.62 GB from
workers through 751,250 messages in each direction. These values and the parent
endpoint time were stable. Parent IPC increased slightly while batch wall time
fell, so it did not cause the observed performance improvement.

Worker send-transfer time was more variable, ranging from 46.1 to 63.9
aggregate seconds in steady state. Because this overlaps across workers and
includes pipe backpressure, it is better treated as a synchronization symptom
than as directly additive transfer overhead.

### 5. Coordinator unattributed time is a stable secondary cost

Coordinator unattributed time averaged 15.550 seconds, or 6.6% of wall time,
and varied by only 3.3%. It includes coordinator loop bookkeeping, status
checks, result handling outside existing buckets, scheduling, and the periodic
yield.

This is large enough for a targeted measurement or A/B test, but it is not
responsible for the current batch-to-batch variation.

## Recommendations

### Priority 1: Split and stabilize inference timing

Add separate timing and counts for:

- Constructing the host batch.
- Host-to-device transfer.
- Model forward execution.
- Action selection, device-to-host transfer, and synchronization.
- The number of trainee and opponent records processed per inference call.

During the same run, record GPU utilization, clocks, and competing device load
if practical. Batch 327 demonstrates that the current architecture can complete
inference approximately 22 seconds faster than batch 323; the first goal should
be determining why and making the faster behavior repeatable.

Collect at least ten additional steady-state batches before judging changes
smaller than roughly five percent. The current downward inference trend may be
warm-up or environmental variation rather than a permanent improvement.

### Priority 2: Localize frame-barrier stragglers

Record the slowest worker identity and response duration per frame, then report
per-worker counts and wait percentiles. This will establish whether a specific
ship, battle composition, or worker repeatedly controls the barrier.

If worker optimization is then justified, start with collision processing and
observation encoding, followed by object updates. Do not infer critical-path
impact solely from aggregate worker CPU-seconds.

### Priority 3: Treat first batches as warm-up

Exclude the entire first batch after worker creation from performance baselines.
If startup throughput matters operationally, investigate the additional 17
seconds of worker wait or introduce an explicit warm-up phase; optimizing only
the 5.5-second process-start timer will miss most of the penalty.

### Priority 4: Keep IPC changes bounded and measured

The absolute ceiling from eliminating all parent IPC is about 11%, and required
observation and replay traffic makes the realistic gain smaller. If IPC work is
pursued, begin with a reversible chunked-message or replay-transfer prototype
and compare wall time, parent endpoint time, worker send blocking, bytes, and
message counts before considering shared-memory redesign.

### Priority 5: Split the coordinator residual only after inference

Add a bucket around the periodic yield and another around frame-result
bookkeeping, or run a controlled yield-reduction A/B test. Check UI
responsiveness and process fairness during that experiment.

## Bottom Line

The current measurements account cleanly for wall time and show two repeatable
costs: an approximately 23.5-second whole-batch cold-start penalty and a
steady-state floor dominated by inference and worker synchronization. The new
information shifts the immediate focus toward inference: nearly all observed
steady-state wall-time improvement came from faster inference while worker CPU
work, worker wait, IPC, and coordinator overhead stayed nearly constant.

After inference is understood, per-frame worker-straggler data should determine
whether collision or observation work can shorten the barrier. IPC remains a
measurable but lower-priority optimization target.
