# StarAI Training Timing Findings

Generated from the timing CSVs in this folder after:

- A 7-instance concurrent training run.
- A later single-instance Androsynth run whose files start with `Single instance`.

## Files Compared

7-instance timing rows:

- `Androsynth-01.instance-01.timings.csv`
- `Arilou-01.instance-02.timings.csv`
- `Chenjesu-01.instance-03.timings.csv`
- `Chmmr-01.instance-04.timings.csv`
- `Druuge-01.instance-05.timings.csv`
- `Earthling-01.instance-06.timings.csv`
- `Ilwrath-01.instance-07.timings.csv`

Single-instance comparison:

- `Single instance Androsynth-01.instance-01.timings.csv`

Event logs:

- `training-multi-instance-events.csv`
- `Single instance training-multi-instance-events.csv`

## Main Finding

The bottleneck is PyTorch model work, not the Python simulation/reward loop.

In the 7-instance run, average completed-batch time was about `439.4s`.
Average bucket share across the 7 completed batch rows:

| Bucket | Avg seconds | Share |
| --- | ---: | ---: |
| trainee inference | 246.2s | 56.0% |
| opponent inference | 89.7s | 20.4% |
| optimization | 47.1s | 10.7% |
| simulation | 22.8s | 5.2% |
| reward | 18.9s | 4.3% |
| observation | 4.9s | 1.1% |

Together, trainee inference + opponent inference + optimization account for about `87%` of 7-instance batch wall time.

## Androsynth Single vs 7-Instance Comparison

Androsynth single-instance completed batch:

- Total: `58.9s`
- Frames: `26698`
- Running instances: `1 -> 1`

Androsynth during 7-instance run:

- Total: `441.5s`
- Frames: `27598`
- Running instances: `3 -> 7`

Overall ratio:

- `441.5 / 58.9 = 7.5x slower`

Bucket comparison:

| Bucket | Single | 7-instance | Extra |
| --- | ---: | ---: | ---: |
| trainee inference | 11.6s | 245.1s | +233.5s |
| opponent inference | 3.8s | 103.6s | +99.8s |
| optimization | 2.9s | 35.9s | +33.0s |
| simulation | 12.9s | 18.3s | +5.4s |
| reward | 16.8s | 21.6s | +4.8s |
| observation | 5.0s | 4.9s | no meaningful change |

Normalized per 1000 frames for Androsynth:

| Bucket | Single per 1k frames | 7-instance per 1k frames | Ratio |
| --- | ---: | ---: | ---: |
| total | 2.206s | 15.998s | 7.25x |
| trainee inference | 0.435s | 8.882s | 20.4x |
| opponent inference | 0.144s | 3.755s | 26.1x |
| optimization | 0.108s | 1.301s | 12.0x |
| simulation | 0.483s | 0.661s | 1.37x |
| reward | 0.628s | 0.782s | 1.24x |
| observation | 0.189s | 0.179s | 0.95x |

This strongly points to GPU/PyTorch contention from many small per-frame inference calls and optimizer steps.

## Aggregate Throughput

The 7-instance event log shows 7 completed batches over roughly `537s`.

Approximate aggregate throughput:

- `7 batches / 537s * 3600 = ~47 batches/hour`

Single Androsynth throughput:

- `1 batch / 58.9s * 3600 = ~61 batches/hour`

So in this run, 7 instances did not improve aggregate throughput. They were slightly worse than one instance.

## What Is Not the Main Bottleneck

Simulation and reward code did slow down under concurrency, but not enough to explain the throughput collapse.

Androsynth:

- simulation: `12.9s -> 18.3s`
- reward: `16.8s -> 21.6s`

Those increases are modest compared with:

- trainee inference: `11.6s -> 245.1s`
- opponent inference: `3.8s -> 103.6s`
- optimization: `2.9s -> 35.9s`

Observation encoding did not slow down meaningfully.

## Save I/O Caveat

The per-batch timing rows show `save_seconds = 0` because the completed batches did not hit a grouped save during those rows.

However, after the user stopped the 7-instance run, all 7 instances saved final progress concurrently. The multi-instance event log shows those saves took about `24-25s` each.

That means concurrent final save I/O is expensive, but it was not the steady-state completed-batch bottleneck for the measured rows.

## Likely Cause

The model calls are tiny and frequent:

- One trainee inference per decision frame.
- Existing-AI opponent inference on frames where the selected opponent uses a trained model.
- End-of-batch replay optimization uses many small optimizer updates.

With multiple instances sharing the same CUDA device, these small operations appear to serialize or contend heavily. The measured inference buckets grow far more than the Python-heavy buckets.

## Recommended Next Steps

1. Test CPU inference/training mode for small models.
   The data suggests CUDA overhead/contention may dominate these tiny per-frame calls.

2. Add a device selection option per training session.
   This would allow comparing:
   - all CUDA
   - all CPU
   - mixed CPU inference / CUDA optimization
   - one CUDA instance plus CPU background instances

3. Reduce optimizer contention.
   Consider fewer replay updates per batch for multi-instance runs, or stagger optimization/saves so all instances do not hit the GPU/disk at the same time.

4. Consider batching inference only if the architecture changes.
   Current instances run independent battle timelines, so batching policy calls across instances would require a central scheduler or worker queue. That is a larger design change.

5. Do not prioritize reward-pipeline optimization first.
   Reward optimization could still help, but these measurements show it is not the main throughput limiter for multi-instance runs.

## Bottom Line

The timing diagnostics show that 7 concurrent instances are bottlenecked primarily by PyTorch inference and optimizer contention. Multi-instance training currently gives near-flat or worse aggregate throughput because each instance slows down by about the same factor as the number of active instances.
