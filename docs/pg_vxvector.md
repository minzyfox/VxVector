# VxVector active roadmap

Updated November 11, 2026.
This is the single source of truth for next steps, priorities, and milestone completion.
The [CPU correctness document](milestone_4_cpu_correctness.md) contains reference procedures and historical evidence.
The [earlier roadmap and research assessments](roadmap_reference.md) preserve design ideas without defining additional execution gates.
Dated compiler reports retain their original revision and validation limits.

## Goal

Run VxVector training on an available supported GPU, with automatic CPU selection when GPU execution is unavailable.
Apple Metal is the first GPU target. NVIDIA support and cuVS comparison follow later.
The existing PostgreSQL function and VectorChord centroid format remain the integration boundary.
Query-time GPU search and replacing VectorChord remain outside this work.

CPU SIMD optimization is no longer a prerequisite for GPU execution.
Full kernel proofs and hosted CI remain useful work, but do not block a GPU experiment independently of relevant correctness failures.
Compiler and runtime improvements should follow concrete application failures with focused regression tests.

Application logic belongs in Vx. C remains only for PostgreSQL internals, process and pipe handling, the specific operations Vx cannot yet express, and deliberate exceptions such as allocation that tests must intercept.
Each Vx gap needs a recorded issue, and each exception a recorded reason, rather than a silent C fallback.

## Current state and evidence limits

The project pins Vx `df51035d` (main, November 11) in [vx-revision.txt](../vx-revision.txt).
The [Step 2 record](step2_data_path_vx_2026-11-11.md) validates the current kernel and data path with that compiler on native macOS arm64.

The CPU core, bounded sampling adapter, separate worker, and PostgreSQL/VectorChord integration are implemented.
The adapter's logic is in Vx since Step 2. Its kernel ABI is `vxvector_kmeans_v2`.
The installed Linux worker and saved Linux build manifests still identify Vx `c9612a1f` and an earlier kernel source hash.
The extension build now links the Vx object and has not been run since that change.
PostgreSQL, recall, and objective results remain evidence for their recorded source and compiler revisions.
No GPU execution has been performed.

The NVIDIA-oriented [GPU prototype](../src/gpu_lloyd.vx) is not a validated Metal implementation.
Its ordinary loops and compilation result do not establish useful GPU parallelism.
CPU distance calculations use f64 intermediates, while the prototype uses f32.
The prototype retains old centroids for empty clusters, while CPU training selects sampled replacements.
These differences require explicit numerical and algorithm policies before result comparisons.

## Backend selection contract

The following modes are proposed behavior, not an implemented public API.
Finalize the SQL and worker-protocol spelling during integration.

| Mode | Required behavior |
| --- | --- |
| `auto` (default) | Select an available GPU that supports the requested operation and numerical policy. Otherwise select CPU. |
| `cpu` | Run CPU training even when a GPU is available. |
| `gpu` | Require supported GPU execution. Return a clear error if selection cannot succeed. |

Make selection before training starts and before GPU-dependent allocation.
Check capabilities as well as device presence.
Report the actual backend, device identity where applicable, and selection reason.
Preserve CPU-only installation and loading without mandatory GPU dependencies.
Report GPU execution failures after work starts. Do not silently replay failed GPU training on CPU.
Keep CPU forcing and GPU requiring modes available for tests and diagnostics.

## Step 1: refresh the CPU correctness baseline

Complete November 9, 2026, for Vx `7e3cf6a3` on macOS arm64. See the [Step 1 record](step1_cpu_baseline_2026-11-09.md).

- [x] Select the Vx revision and rebuild its compiler/runtime and the current VxVector kernel.
- [x] Record matching compiler, source, object, and worker hashes.
- [x] Run native unit tests, independent Lloyd/Faiss checks, edge cases, sanitizers, and contract/code-generation mutations.
- [x] Confirm the intended flat code generator and audit the actual current kernel.
- [x] Record relevant remaining proof gaps without treating old milestone prose as current compiler evidence.

The new compiler crashed on a float branch condition in `normalize`; the kernel works around it, and the reproduction is filed as [vx-lang/Vx#3145](https://github.com/vx-lang/Vx/issues/3145).

Exit: the current source has a reproducible CPU reference and fallback.
Revalidate PostgreSQL installation and recall when packaging the refreshed worker.
Do not replace an installed worker merely to begin standalone Metal development.

## Step 2: convert the C data path to Vx

[src/data_path.c](../src/data_path.c) holds the data-path logic in C: option checks, buffer sizing, the memory budget, the sampler, row validation, and batch copying.
The kernel has a second sampler, the stride permutation in `train`, so two code paths choose samples.
Convert this layer to Vx before GPU work so that one sampling policy feeds both backends and every GPU comparison.
This is items D1 and D2 of the [November 8 usage review](vx_usage_review_2026-11-08.md), extended to the whole file.

Native work complete November 11, 2026, for Vx `df51035d` on macOS arm64. See the [Step 2 record](step2_data_path_vx_2026-11-11.md).

- [x] Convert `vxvector_plan_cpu_v1` to Vx: option validation, overflow-checked buffer sizing, and the memory-budget check.
- [x] Convert the sampler to Vx: SplitMix64, Floyd's selection with its hash table, and the position sort.
- [x] Remove the stride permutation from `train`, so that the converted sampler is the only one.
- [x] Convert row validation (null rows, type, dimensions, non-finite values), batch copying, and the final centroid finiteness check.
- [x] Prove that sampled row indices are less than `rows`. Add a mutation that fails each proof.
- [ ] Prove that buffer sizes do not overflow. Tested, not proved: the prover ignores `/` and cannot use the `u64` constants.
- [x] Remove `batch_size` from the kernel ABI. Keep it in the row reader. Bump the ABI version if the kernel signature changes.
- [x] Keep the `vxvector_plan_cpu_v1`, `vxvector_run_cpu_v1`, and `vxvector_free_cpu_result_v1` entry points, error codes, and messages, so that `worker.c` and `bridge.c` stay unchanged.
- [x] Keep in C only what Vx cannot express, or record why not. Allocation, the clock, and the message text stay in C by choice, for the reasons in the record. The row-reader callback works from Vx.
- [x] Rerun the Step 1 checks and the data-path tests, including allocation failure and sanitizers, and record new hashes.
  Sample rows and plans match the C implementation exactly. Centroids change, as the record explains.
- [ ] Build the Linux worker and the extension, which now links the Vx object, and rerun the PostgreSQL suite and recall.
- [x] Check the Vx finding that an assigned call result loses its `ensures`. It is [vx-lang/Vx#3040](https://github.com/vx-lang/Vx/issues/3040), already open.

Out of scope: [pg_vxvector/src/bridge.c](../pg_vxvector/src/bridge.c) and the Rust wrapper, which depend on PostgreSQL internals, and the process and pipe code in [src/worker.c](../src/worker.c).

Exit: `src/data_path.c` contains only Vx gaps that have recorded issues, or exceptions with recorded reasons. The array API and the streaming adapter choose the same sample for the same seed and rows.
PostgreSQL and recall revalidation still waits for packaging of the refreshed worker, as in Step 1.
The Linux build needs a check before then, because the extension build changed.

## Step 3: execute standalone L2 clustering on Apple Metal

Run this directly on macOS, outside PostgreSQL and Linux Docker.

- [ ] Define the GPU precision policy, tolerances, finite-value handling, tie policy, and empty-cluster replacement behavior.
- [ ] Preserve the CPU f64-distance baseline. Validate any f32 GPU policy explicitly rather than silently changing CPU semantics.
- [ ] Implement bounded point assignment and centroid updates using supported Apple backend operations.
- [ ] Validate actual launch parallelism and race-free reductions. Choose partial buffers and merging if atomics are unavailable.
- [ ] Keep samples, centroids, assignments, and reusable scratch resident across Lloyd iterations.
- [ ] Record device completion before host reads, buffer reuse, and release.
- [ ] Compare one fixed L2 input with the CPU reference and independent objective calculations.
- [ ] Require Metal dispatch evidence. A host copy or CPU fallback cannot satisfy this step.

Exit: one real L2 training operation executes on Apple GPU with correct bounded results and cleanup.
Do not add transfer overlap, multiple devices, or general tuning infrastructure to this first result.

## Step 4: automatic selection and CPU fallback

- [ ] Add a backend capability/availability probe and implement the three selection modes above.
- [ ] Test `auto` with GPU present, unavailable, and incompatible with a requested capability.
- [ ] Test forced CPU and required GPU behavior without depending on the developer machine's device availability.
- [ ] Report the selected backend and reason from the standalone worker interface.
- [ ] Verify completion, cancellation, and resource cleanup on success and failure.

Exit: the same training request runs on a supported GPU or selects CPU before training when GPU execution is unavailable.

## Step 5: complete standalone GPU semantics and validation

- [ ] Extend the supported path to cosine, inner product, normalization, sampling, restarts, and flat/two-level output.
- [ ] Keep unsupported combinations on the CPU in `auto`; reject them in required GPU mode.
- [ ] Check zero vectors, empty clusters, duplicate points, non-finite inputs, awkward dimensions, and repeated execution.
- [ ] Compare centroids, assignments, independently scored objectives, and hierarchy structure under declared tolerances.
- [ ] Test buffer limits, allocation failure behavior, and cancellation with outstanding device work.
- [ ] Record transfer, allocation, training, and total time separately, including dispatch evidence and backend versions.

Exit: supported GPU options preserve the application's defined behavior, with explicit CPU selection for unsupported requests.
No performance advantage is assumed from successful device execution.

## Step 6: PostgreSQL integration and final-index quality

The current database runs in a Linux Docker container.
It cannot directly execute a native macOS Metal worker.
Select one deployment path after standalone Metal works:

| Path | Work required |
| --- | --- |
| Native macOS PostgreSQL and worker | Validate the PostgreSQL extension and its dependencies natively. Invoke the Metal worker as a native process. |
| Existing Docker PostgreSQL and macOS worker service | Add a host-service transport with bounded requests, lifecycle control, cancellation, and error handling. |

Neither path is established by existing Linux worker tests.

- [ ] Choose and validate one deployment path. Keep the current Linux CPU setup usable.
- [ ] Extend SQL/backend reporting and version worker communication if its contract changes.
- [ ] Exercise SQL-to-sampling-to-GPU-training-to-centroid-table-to-VectorChord-index execution.
- [ ] Recheck exact-scan ground truth, held-out recall, and objective quality using the existing declared gates.
- [ ] Test `auto`, forced CPU, required GPU, worker failure, timeout, cancellation, and transaction rollback.
- [ ] Verify unsupported or missing GPU capability selects CPU without requiring application SQL changes in default mode.

Exit: the PostgreSQL workflow uses the GPU when supported and CPU otherwise, with observable selection and validated index quality.

## Step 7: performance and later backends

- [ ] Establish an optimized CPU baseline with `-O3` and inspect its generated instructions.
- [ ] Profile GPU assignment, centroid updates, transfers, allocation, and launch overhead on native hardware.
- [ ] Improve measured bottlenecks through tiling, fusion, reductions, and workspace reuse.
- [ ] Recheck numerical behavior and final-index recall after arithmetic or reduction-order changes.
- [ ] Add completion-safe overlap only with device timelines and simultaneous-memory accounting.
- [ ] Develop CPU SIMD as fallback optimization independently of GPU availability.
- [ ] Add NVIDIA execution and CUDA integration after the Apple path is validated.
- [ ] Compare against cuVS only on supported NVIDIA hardware with matched semantics and timing boundaries.

Exit: reproducible end-to-end measurements support each performance claim, including cases where CPU is faster.

## Reference map

- [CPU correctness procedures and historical results](milestone_4_cpu_correctness.md).
- [Internal VectorChord objective comparison](internal_objectives.md).
- [Earlier roadmap, proof candidates, and research backlog](roadmap_reference.md).
- [Standalone library implementation record](milestone_1_standalone.md).
- [Bounded data-path implementation record](milestone_2_data_path.md).
- [CPU PostgreSQL integration record](milestone_3_cpu_integration.md).
- [Naming and rename-validation record](rename_vxvector.md).
- [Step 1 CPU baseline record](step1_cpu_baseline_2026-11-09.md).
- [Step 2 data path record](step2_data_path_vx_2026-11-11.md).
- [November 8 Vx usage review](vx_usage_review_2026-11-08.md) and [Vx issue drafts](vx_issue_drafts_2026-11-08.md).

These references supply procedures and evidence. They do not add prerequisites or override this sequence.
Record new validation under its exact source/compiler/backend versions before marking an active step complete.
