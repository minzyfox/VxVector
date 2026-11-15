# Step 5: every training option on the GPU (November 14, 2026)

This record completes Step 5 of the [active roadmap](pg_vxvector.md) for the standalone worker on Apple Metal.
Every option that the CPU kernel accepts now trains on the GPU: L2, inner product and cosine, restarts, spherical centers, and two-level trees.
In `auto`, the worker now uses the CPU only when the GPU library is missing, no Metal device is present, the sample exceeds the kernels' `i32` limits, or the memory budget cannot hold the CPU and GPU buffers together.

The results match the CPU bit for bit, except at float32 near-ties. Then they stay within the [declared tolerance](#the-declared-tolerance).
PostgreSQL does not use the GPU yet. Its extension still sends version 1 requests (Step 6).

Vx is `423a10db`, the current pin. The CPU kernel builds at `-O3` ([record](cpu_o3_2026-11-14.md)). The device is an Apple M2 Pro running macOS 27.2.1. The [metadata file](evidence/step5_metadata_2026-11-14.json) lists every hash.

## How each option runs

| Option | Where it runs | Same as the CPU |
| --- | --- | --- |
| Two-level trees | The GPU trains the leaves into their final place. The data path then trains the roots on the CPU from those leaves, with `vxvector_kmeans_v2`'s own sample count, seed, and options. | Yes. Roots train on `leaf_count` rows, too few for the GPU to help |
| Inner product and cosine | A second assignment kernel adds each dot product and squared norm in float32, eight centers at a time. It keeps the CPU's rules: `-dot`, and a cosine of 0 for two zero vectors and 1 when one is zero | float32 distances. Cosine takes `sqrt(left) * sqrt(right)`, because `left * right` can overflow float32 |
| Restarts | Each restart runs on the GPU from the CPU's shifted seeds, `(c * n / k + r * 104729) % n`. The data path keeps the first restart, then any with a strictly lower objective. When an earlier restart wins, the GPU assigns the sample to it again | Yes, wherever the labels are the float64 nearest centers. The objective adds the CPU's float64 distance from each sample to its GPU label: O(n × dims), not the CPU's O(n × k × dims) |
| Spherical centers | The CPU normalizes the centers in float64, between GPU launches, after seeding and after each update: `vxvector_gpu_normalize_rows` in [src/gpu_l2.c](../src/gpu_l2.c) | Yes. It is the CPU's arithmetic, built with `-ffp-contract=off` |

Spherical normalization runs on the CPU because Apple GPUs have no float64. The memory is unified, so the CPU reads and writes the GPU's buffer in place. Normalizing k × dims values costs little.

Normalizing in C is a recorded exception to "logic in Vx": a Vx function cannot read GPU memory. Vx gets it as a function pointer, because `vxc` resolves `extern` symbols when it emits the object, and this one is not built yet then. Before the call, Vx calls the runtime's `vx_metal_wait()`, so the GPU has finished with the buffer. That wait is required. vxc waits by itself before a call to an `extern` C function that gets GPU memory, but not before a call through a function pointer. A standalone test copied 1,048,576 stale values of 1,048,576 without it ([Vx issue draft 8](vx_issue_drafts_2026-11-14.md)).

The compiler may fuse `squared += wide * wide` into one fused multiply-add, which rounds once instead of twice. That changes the squared norm by about 1e-16, which no test data shows. So [scripts/build_gpu.sh](../scripts/build_gpu.sh) checks the instructions instead: it fails when `vxvector_gpu_normalize_rows` contains `fmadd` or `fmsub`. Built with `-ffp-contract=fast`, it stops with an error.

## The declared tolerance

Step 3 found that the GPU's float32 distances can choose another center than the CPU's float64 ones at a near-tie. It left the tolerance to this step.

**What happens.** Nearly parallel vectors give competing cosine distances about 1e-8 apart, below float32's resolution of about 6e-8 near 1. One sample then takes the other center, and the runs diverge after it.
Diagnosis on one case found a sample at 0.0042121950 against 0.0042122224, a gap of 2.7e-8. Spherical centers against raw SIFT vectors gave the same under L2.

| Inputs ([log](evidence/step5_worker_sift_2026-11-14.txt) and diagnosis) | Runs that diverged | Largest objective difference |
| --- | --- | --- |
| L2, blobs | 0 of 6 | none |
| Inner product, blobs | 0 of 6 | none |
| Cosine, blobs | 4 of 6 | 1.6e-4 |
| SIFT-128 with 256 leaves: L2, 3 restarts, roots | 0 of 3 | none |
| SIFT-128: cosine, spherical L2, and all options together | 3 of 3 | 1.1e-5 |

**The rule.** A GPU result passes if it is bit for bit the CPU's. Otherwise it must meet both of these:
1. Every GPU label is the float64 nearest center of the GPU's own centers, except where the float64 gap is below 1e-5 of the distance, or 1e-6 for cosine.
2. The float64 objective of the GPU's centers, scored independently, is within 1e-3 of the CPU's.

**Where it applies.** Only to runs that met a near-tie: the case `cosine_near_ties`, the run `sift_8192/spherical`, and cosine in the worker test. Every other run must stay exact, because exact runs are what catch the mutations below.
The gate prints each divergence, so a divergence is reported, not hidden.

## Results

| Check | Result |
| --- | --- |
| GPU kernel check ([log](evidence/step5_gpu_check_2026-11-14.jsonl)) | 97 runs passed: 14 cases under L2, 11 under inner product and cosine, and 8 also spherical, at the default and a split launch budget. Exact, apart from the runs listed above |
| Native suite ([log](evidence/step5_suite_2026-11-14.txt)) | 54 tests passed. These include 14 selection tests with the fake GPU, and 3 with the real library |
| Codegen and audit ([audit](evidence/step5_audit_2026-11-14.txt)) | Passed. The audit lists 56 trusted clauses, 10 more than Step 4. All 10 are at the C boundary `vxvector_run_reason_v2`: the prover cannot see that the GPU initialized the centers and labels it wrote |
| Contracts | 5 proofs, 6 mutations |

The worker tests with the real Metal library run 18 configurations, one comparison each: L2, inner product, and cosine, with and without spherical centers, roots, and 3 restarts.
They compare nodes and parents with the CPU worker, byte for byte. Only cosine may diverge, within the tolerance; one configuration did, by 1.2e-4.
A test through `vxvector_run_v2` checks that the sample's assignments are the CPU's, and the float64 nearest leaves, when an earlier restart wins.

The Linux worker, extension, and image were rebuilt with these sources. The image, extension, snapshot, and synthetic recall checks passed ([log](evidence/step5_linux_2026-11-14.txt)). They use version 1 and the CPU.

Other checks:
- Non-finite rows are rejected before GPU training, as on the CPU.
- The `i32` limit is exact. Samples of 2^21 − 1 rows of 1,024 values select the GPU, and 2^21 select the CPU, or fail in `gpu` mode.
- The kill test from Step 4 still passes.

### Mutations

Each must fail a check ([metrics](evidence/step5_metric_mutations_2026-11-14.jsonl), [restarts](evidence/step5_restart_mutations_2026-11-14.jsonl), [spherical](evidence/step5_spherical_mutations_2026-11-14.jsonl)):

| Mutation | Caught by |
| --- | --- |
| Cosine of two zero vectors is 1 | `group_ties/cos` |
| Cosine with one zero vector is 0 | `empty_clusters/cos`, `group_ties/cos` |
| Inner product keeps its sign | 16 inner-product runs |
| A tie in a group of eight goes to the higher index (inner product and cosine) | `group_ties` |
| Cosine takes `sqrt(left * right)` | `large_vectors/cos`, whose norm product overflows float32 |
| A worse restart replaces the best | Worker tests, real and fake |
| Restart seeds shift by 104723 instead of 104729 | Worker test against the CPU |
| No new assignment when an earlier restart wins | The `vxvector_run_v2` test |
| The best restart is never copied | Worker tests |
| Normalization in float32 | 11 spherical runs |
| Seeds not normalized | 7 spherical runs |
| Centers not normalized after an update | 23 spherical runs |
| A fused squared norm | `scripts/build_gpu.sh` |

Three were missed at first and are caught now:
- **`sqrt(left * right)`.** No input overflowed float32, so `large_vectors` was added.
- **No new assignment.** The worker does not send assignments, so the `vxvector_run_v2` test was added.
- **The fused squared norm.** The data cannot show it, so the build check was added. That check itself did not work at first: macOS `grep -E` has no `\b`, and with `pipefail`, `grep -q` stopping early failed `objdump`. Both are fixed.

## Timings and versions

The worker's version 2 reply now adds the GPU's share of `clustering_seconds`: allocation (with release), copy-in, training, and copy-out. It also adds the GPU library's version, such as `Metal, Vx 423a10db`.
The GPU library sums them per thread, and the worker resets them before the run.

100,000 SIFT-128 rows, 256 leaves (32,768 samples), 10 iterations, `auto` against the `-O3` CPU worker ([log](evidence/step5_worker_sift_2026-11-14.txt)):

| Options | GPU clustering | of which GPU training | CPU clustering | Faster | Result |
| --- | --- | --- | --- | --- | --- |
| Defaults: L2, 1 restart | 0.284 s | 0.283 s | 6.92 s | 24.3x | same bytes |
| Cosine | 0.336 s | 0.334 s | 9.42 s | 28.1x | near-tie, objective −1.1e-5 |
| Spherical L2 | 0.295 s | 0.293 s | 6.99 s | 23.7x | near-tie, objective +4.9e-8 |
| 3 restarts | 0.829 s | 0.818 s | 19.73 s | 23.8x | same bytes |
| 16 roots | 0.279 s | 0.274 s | 6.92 s | 24.8x | same bytes |
| Cosine, spherical, 3 restarts, 16 roots | 0.996 s | 0.967 s | 26.62 s | 26.7x | near-tie, objective −8.1e-6 |

Allocation and copies take a few milliseconds, so training is almost all of the time. The first total for 3 restarts, 2.25 s, did not repeat in three reruns, which took 1.19 to 1.21 s.
The CPU is one thread of ten.

## Not covered

- **PostgreSQL.** Step 6.
- **Removing near-ties.** The CPU could settle each label whose float32 gap is within float32 error, in float64. Then cosine would match the CPU bit for bit too. Rows near a tie are few, but each costs O(k × dims) on the CPU between launches.
- **GPU allocation failure.** It returns status 2, which the worker reports as a GPU failure, but it is not injected in a test.
- **Larger GPU sizes.** The `i32` limit, 2^31 values, holds. Sizes near it were tested through selection, not trained.
- **Kernel speed.** Centers are still not in threadgroup memory, and the update reads every label once per center and dimension. That is Step 7.
