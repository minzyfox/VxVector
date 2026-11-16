# VxVector active roadmap

Updated November 15, 2026.
This is the single source of truth for next steps, priorities, and milestone completion.
[gpu_design.md](gpu_design.md) describes the GPU design and the rules its results must meet.
The [earlier roadmap and research assessments](roadmap_reference.md) preserve design ideas without defining additional execution gates.
The dated validation records of each step are in the git history of `docs/`.

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

The project pins Vx `889a1fbb` (main, November 15) in [vx-revision.txt](../vx-revision.txt).
The previous pin, `423a10db`, was a branch commit that is not on `main`.
At `889a1fbb`, the compiler writes the same native `kmeans.o`, Metal object, Metal source, and A100 PTX as at `423a10db`, byte for byte.
The kernel and data path were validated on native macOS arm64 at `df51035d`, and again at `423a10db`, which emits the same `kmeans.o`.

The CPU core, bounded sampling adapter, separate worker, and PostgreSQL/VectorChord integration are implemented.
The adapter's logic is in Vx since Step 2. Its kernel ABI is `vxvector_kmeans_v2`.
The Linux worker, the extension, and the PostgreSQL image are rebuilt at `423a10db`. The extension links the Vx object.
The PostgreSQL suites, snapshot, synthetic recall, and real-data recall pass at that revision.
Internal objective results remain evidence for their recorded source and compiler revisions.
PostgreSQL sends protocol-v2 requests with `auto`, `cpu`, or `gpu` backend
selection. Native macOS can select Metal when the GPU library is available;
the Linux Docker image cannot use Metal and currently falls back to CPU.
The NVIDIA GPU library builds from the same Vx kernels as Metal, but has not
run on NVIDIA hardware, and it is not part of the Linux PostgreSQL build or image.
The Step 3 kernels split their work into launches of at most about 2 ms, because macOS can stop a long launch and the Vx runtime then aborts. On SIFT-128 they train 24 to 28 times faster than the single-threaded `-O3` CPU kernel, with the same bits.

NVIDIA GPUs now use the same Vx source as Metal, [gpu_kmeans.vx](../src/gpu_kmeans.vx), with the Vx CUDA runtime in place of a hand-written launcher.
The kernels and the training loop are shared. Each GPU adds one small C file behind [gpu_backend.h](../src/gpu_backend.h): allocate, free, copy in, copy out, wait, and the device name.
[gpu_backend_metal.m](../src/gpu_backend_metal.m) is the Apple one, and [gpu_backend_cuda.c](../src/gpu_backend_cuda.c) the NVIDIA one.
The Metal library gives the same bits as before the split, for all 902 outputs of the GPU checks.
The CUDA library, built by [build_gpu_cuda.sh](../scripts/build_gpu_cuda.sh), compiles and links with `-z defs` against the Vx CUDA runtime on Linux, but has not run on an NVIDIA GPU.
Spherical centers are normalized on the host in float64, as on Metal, after a copy of the k centers.
The earlier Driver API prototype ([gpu_cuda.cpp](../src/gpu_cuda.cpp), [gpu_lloyd.vx](../src/gpu_lloyd.vx)) stays, built by `build_gpu_lloyd_cuda.sh`, until the shared build passes on an A100.

## Backend selection contract

The extension exposes these modes through `vxvector.backend_mode`; the setting
is sent to the worker in protocol v2.

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

Complete November 9, 2026, for Vx `7e3cf6a3` on macOS arm64.

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

Native work complete November 11, 2026, for Vx `df51035d` on macOS arm64.
Native checks repeated November 14, 2026, for Vx `423a10db`.

- [x] Convert `vxvector_plan_cpu_v1` to Vx: option validation, overflow-checked buffer sizing, and the memory-budget check.
- [x] Convert the sampler to Vx: SplitMix64, Floyd's selection with its hash table, and the position sort.
- [x] Remove the stride permutation from `train`, so that the converted sampler is the only one.
- [x] Convert row validation (null rows, type, dimensions, non-finite values), batch copying, and the final centroid finiteness check.
- [x] Prove that sampled row indices are less than `rows`. Add a mutation that fails each proof.
- [ ] Prove that buffer sizes do not overflow. Tested, not proved: the prover ignores `/` and cannot use the `u64` constants.
- [x] Remove `batch_size` from the kernel ABI. Keep it in the row reader. Bump the ABI version if the kernel signature changes.
- [x] Keep the `vxvector_plan_cpu_v1`, `vxvector_run_cpu_v1`, and `vxvector_free_cpu_result_v1` entry points, error codes, and messages, so that `worker.c` and `bridge.c` stay unchanged.
- [x] Keep in C only what Vx cannot express, or record why not. Allocation, the clock, and the message text stay in C by choice, for the reasons in the header comment of `src/data_path.c`. The row-reader callback works from Vx.
- [x] Rerun the Step 1 checks and the data-path tests, including allocation failure and sanitizers, and record new hashes.
  Sample rows and plans match the C implementation exactly. Centroids change, because the kernel's own stride sampler is gone.
- [x] Build the Linux worker and the extension, which now links the Vx object, and rerun the PostgreSQL suite and recall.
  Done November 14, 2026, at Vx `423a10db`.
- [x] Check the Vx finding that an assigned call result loses its `ensures`. It is [vx-lang/Vx#3040](https://github.com/vx-lang/Vx/issues/3040), already open.
- [ ] Low priority, does not block Step 3: remove [src/data_path.c](../src/data_path.c) by moving its last parts to Vx.
  - Export the `vxvector_*_v1` functions from Vx, and write the error messages into `vxvector_data_error` from Vx.
  - Allocate with Vx `std::alloc`, and rework the allocation-failure test in [tests/data_path_memory.c](../tests/data_path_memory.c), which replaces `malloc` in C today.
  - Get the clock without the Vx runtime library in the extension and Linux worker. Either `worker.c` passes it in, or both link the library.

Out of scope: [pg_vxvector/src/bridge.c](../pg_vxvector/src/bridge.c) and the Rust wrapper, which depend on PostgreSQL internals, and the process and pipe code in [src/worker.c](../src/worker.c).

Exit: `src/data_path.c` contains only Vx gaps that have recorded issues, or exceptions with recorded reasons. The array API and the streaming adapter choose the same sample for the same seed and rows.
PostgreSQL and recall were revalidated with the refreshed worker on November 14, 2026.

## Step 3: execute standalone L2 clustering on Apple Metal

Run this directly on macOS, outside PostgreSQL and Linux Docker.

Complete November 14, 2026, for Vx `423a10db` on an Apple M2 Pro.

- [x] Define the GPU precision policy, tolerances, finite-value handling, tie policy, and empty-cluster replacement behavior.
  float32 distances are the only difference from the CPU. Fixed inputs must match bit for bit. The near-tie tolerance is left to Step 5.
- [x] Preserve the CPU f64-distance baseline. Validate any f32 GPU policy explicitly rather than silently changing CPU semantics.
- [x] Implement bounded point assignment and centroid updates using supported Apple backend operations.
- [x] Validate actual launch parallelism and race-free reductions. Choose partial buffers and merging if atomics are unavailable.
  Metal has no float atomics. One block owns each center and one lane owns each dimension, and the `across` checker proves no shared writes.
- [x] Keep samples, centroids, assignments, and reusable scratch resident across Lloyd iterations.
- [x] Record device completion before host reads, buffer reuse, and release.
- [x] Compare one fixed L2 input with the CPU reference and independent objective calculations.
  Eleven inputs, including 8,192 SIFT-128 vectors, match the CPU kernel and a NumPy reference bit for bit. Five kernel mutations are caught.
- [x] Require Metal dispatch evidence. A host copy or CPU fallback cannot satisfy this step.

Exit: one real L2 training operation executes on Apple GPU with correct bounded results and cleanup.
Do not add transfer overlap, multiple devices, or general tuning infrastructure to this first result.

## Step 4: automatic selection and CPU fallback

Complete November 14, 2026, for the standalone worker on an Apple M2 Pro.

- [x] Add a backend capability/availability probe and implement the three selection modes above.
  Selection is in Vx. The worker loads the GPU library with `dlopen`, so CPU-only installs need nothing new.
- [x] Test `auto` with GPU present, unavailable, and incompatible with a requested capability.
- [x] Test forced CPU and required GPU behavior without depending on the developer machine's device availability.
  A fake GPU library stands in for Metal. Two more tests use the real library where it is present.
- [x] Report the selected backend and reason from the standalone worker interface.
  Worker protocol version 2. Version 1 requests still train on the CPU, so the extension is unchanged until Step 6.
- [x] Verify completion, cancellation, and resource cleanup on success and failure.
  A GPU failure after training starts is reported and not rerun on the CPU.

Exit: the same training request runs on a supported GPU or selects CPU before training when GPU execution is unavailable.

## Step 5: complete standalone GPU semantics and validation

Complete November 14, 2026, for the standalone worker on an Apple M2 Pro, except GPU allocation-failure injection.

- [x] Extend the supported path to cosine, inner product, normalization, sampling, restarts, and flat/two-level output.
  Roots and spherical normalization run on the CPU between GPU launches, with the CPU's own arithmetic.
- [x] Keep unsupported combinations on the CPU in `auto`; reject them in required GPU mode.
  Every option is now supported. What remains is the GPU library, the device, the `i32` limit, and the budget.
- [x] Check zero vectors, empty clusters, duplicate points, non-finite inputs, awkward dimensions, and repeated execution.
- [x] Compare centroids, assignments, independently scored objectives, and hierarchy structure under declared tolerances.
  Bit for bit, or at float32 near-ties: nearest labels up to the tie, and an objective within 1e-3. Cosine diverges often on nearly parallel vectors.
- [ ] Test buffer limits, allocation failure behavior, and cancellation with outstanding device work.
  The `i32` limit and cancellation are tested. GPU allocation failure is reported as status 2, but it is not injected.
- [x] Record transfer, allocation, training, and total time separately, including dispatch evidence and backend versions.

Exit: supported GPU options preserve the application's defined behavior, with explicit CPU selection for unsupported requests.
No performance advantage is assumed from successful device execution.

## Step 6: PostgreSQL integration and final-index quality

The development database runs in a Linux Docker container, which cannot directly
execute a native macOS Metal worker. The Apple GPU path targets native macOS
PostgreSQL and worker execution. The Linux Docker setup remains CPU-only.

| Path | Work required |
| --- | --- |
| Native macOS PostgreSQL and worker (chosen for Apple GPU) | Validate the PostgreSQL extension and its dependencies natively. Invoke the Metal worker as a native process. |
| Existing Docker PostgreSQL and macOS worker service | Not required for the native macOS target; would need a separate host-service transport. |

- [x] Choose the native macOS path for Apple GPU execution and keep the Linux CPU setup usable.
- [x] Send protocol-v2 `auto` requests from SQL, expose `vxvector.backend_mode` (`auto`, `cpu`, `gpu`), and report the worker's backend, reason, device, and GPU timings.
  The bridge and protocol are covered by CPU fallback SQL tests and simulated GPU worker tests, and by native PostgreSQL on macOS.

Complete November 15, 2026, on native macOS 27.2.1 with an Apple M2 Pro, except the instrumented objective comparison.

- [x] Resolve the native VectorChord loading blocker and validate PostgreSQL extension loading on macOS.
  dyld rejected stripped bundles on macOS 27.2.1 (`mis-aligned LINKEDIT string pool`). The strip step causes it, not LTO.
  Build VectorChord with `CARGO_PROFILE_RELEASE_STRIP=none`. [build_macos_extension.sh](../scripts/build_macos_extension.sh) now sets `CARGO_PROFILE_DEV_STRIP=none` for `pg_vxvector`.
- [x] Exercise native SQL-to-sampling-to-GPU-training-to-centroid-table-to-VectorChord-index execution.
  The README walkthrough selects the GPU in `auto` mode and is served by an index scan.
  Over a socketpair, the worker sometimes missed the backend's `shutdown(SHUT_WR)` and hung until the timeout, about one call in seventy, apparently a lost wakeup in the macOS kernel. The worker now polls before its final read, and a regression test covers it.
- [x] Recheck exact-scan ground truth and held-out recall using the existing declared gates.
  The synthetic gate and the SIFT-128 and GloVe-25 comparison against VectorChord's trainer pass, with VxVector training on the GPU.
- [ ] Recheck objective quality with the instrumented VectorChord build (`check_real_recall.py --capture-objectives`). Not run natively.
- [x] Test `auto`, forced CPU, required GPU, worker failure, timeout, cancellation, and transaction rollback through native macOS PostgreSQL.
  `cpu`, `gpu`, and `auto` give identical centroids. Each failure leaves no outputs, and the session keeps working.
- [x] Verify unsupported or missing GPU capability selects CPU without requiring application SQL changes in default mode.
- [ ] Report the strip alignment issue to VectorChord and the socket wakeup issue to Apple.

Exit: the PostgreSQL workflow uses the GPU when supported and CPU otherwise, with observable selection and validated index quality.

## Step 7: performance and later backends

- [ ] Establish an optimized CPU baseline with `-O3` and inspect its generated instructions.
  Started November 14, 2026: the builds use `-O3`, with the same bits as `-O0` and about 6x faster CPU training.
  Instruction inspection, threads, and SIMD remain.
- [ ] Profile GPU assignment, centroid updates, transfers, allocation, and launch overhead on native hardware.
- [ ] Improve measured bottlenecks through tiling, fusion, reductions, and workspace reuse.
- [ ] Recheck numerical behavior and final-index recall after arithmetic or reduction-order changes.
- [ ] Add completion-safe overlap only with device timelines and simultaneous-memory accounting.
- [ ] Develop CPU SIMD as fallback optimization independently of GPU availability.
- [ ] Add NVIDIA execution and CUDA integration after the Apple path is validated.
- [x] Check whether the NVIDIA worker callbacks can be implemented in Vx using its existing host-side CUDA dispatch runtime, rather than the prototype's direct CUDA Driver API wrapper. Verify exported C ABI, runtime linkage, and recoverable worker error behavior on Linux.
  Done November 15, 2026, without a GPU. [gpu_kmeans.vx](../src/gpu_kmeans.vx) builds for the A100 and links with the Vx CUDA runtime, which `vxc` builds only where the CUDA toolkit is installed. On Linux without a GPU, the library exports every worker symbol, probes as unavailable, and reports a failed restart as status 2.
- [ ] Run the shared CUDA library on an A100: `check_gpu_l2.py`'s CPU comparison and near-tie tolerance, every metric, and spherical centers. Confirm that the Vx runtime runs launches in order, and that a runtime abort reaches SQL as a failed GPU. Then remove the Driver API prototype.
- [ ] Compare against cuVS only on supported NVIDIA hardware with matched semantics and timing boundaries.

Exit: reproducible end-to-end measurements support each performance claim, including cases where CPU is faster.

## Reference map

- [GPU training design](gpu_design.md): layout, backend selection, numeric policy, tolerance, and open questions.
- [Earlier roadmap, proof candidates, and research backlog](roadmap_reference.md).
- [November 8 Vx usage review](vx_usage_review_2026-11-08.md): how VxVector uses Vx, and what it should use next.
- Vx issue drafts not yet filed: [November 8](vx_issue_drafts_2026-11-08.md) (prover and raw-pointer facts) and [November 14](vx_issue_drafts_2026-11-14.md) (Metal, linking, and the C boundary).
- [Internal VectorChord objective comparison](internal_objectives.md): the procedure for the open objective-quality check in Step 6.
- [PostgreSQL development image](postgres.md).

These references supply design and procedures. They do not add prerequisites or override this sequence.
Record new validation, with its exact source, compiler, and backend versions, in the commit that marks an active step complete.
