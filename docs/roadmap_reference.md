# Reference: earlier roadmap and research assessments

Archived November 9, 2026. This document preserves earlier plans, research, and recorded milestone statements.
It is reference material, not an execution checklist or a current capability guarantee.
The [active roadmap](pg_vxvector.md) is the single source of truth for priorities, sequencing, and completion gates.
Its Apple-first GPU sequence supersedes the earlier CPU/SIMD/NVIDIA ordering below.
Earlier completion markers are rendered as recorded statements, not current validation of the latest source.
Unfinished items are historical candidates, not additional prerequisites.
Revision, installation, and capability claims retain their historical scope.

---

# Earlier plan: Vx k-means for VectorChord

Oct 20, 2026 · @desia

Plan updated Oct 26, 2026: validate CPU correctness, add SIMD with Vx compiler and standard-library improvements,
enable GPU execution, then compare performance against cuVS.

## Goal and scope

Build a Vx k-means implementation for VectorChord's external build, wrapped in a pgpu-style Postgres function.
Develop VxVector and Vx together. Application kernels should expose compiler and standard-library gaps as each stage progresses.
The remaining work follows this order:

1. End-to-end CPU tests and correctness validation (M4).
2. SIMD kernels, alongside Vx compiler and SIMD standard-library enhancements (M5).
3. GPU enablement, alongside the required Vx compiler and runtime enhancements (M6).
4. Performance and recall comparison against cuVS (M7).

Small CPU integration checks already pass. They do not replace representative end-to-end correctness validation in M4.
GPU enablement follows CPU correctness and SIMD work. The cuVS comparison follows working GPU execution.

The flow is: read vectors, run Vx k-means on the CPU, write a centroid table, then build a VectorChord index with `build.external`.
The later GPU path will use the same flow and centroid-table format. The reference integration is pgpu's ([pgpu](https://github.com/EnterpriseDB/pgpu), [VectorChord external build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)).

- In scope: batched, sampled k-means (flat and two-level), distance operators ip / l2 / cos, spherical centroids, centroid table output, VectorChord handoff.
- Out of scope for now: query-time GPU work, replacing VectorChord's index, other index types.
- First success: build and query a VectorChord index from Vx CPU centroids without GPU hardware or CUDA libraries. Validate flat and two-level trees.
- Later success: add optional GPU acceleration and compare recall and build time against pgpu/cuVS on the same dataset (threshold TBD).

## Core library and integration boundary

The reusable k-means core uses native Vx calls and contracts. It does not require a C ABI between its internal functions.
Keep clustering, distances, normalization, and hierarchy construction independent of PostgreSQL and VectorChord's table representation.
The C ABI is an integration adapter. In the current Postgres flow, the separate C worker calls the exported Vx entry point through it.
Keep foreign-buffer validation, IPC, and SQL concerns at those integration boundaries.
A future standalone Vx caller should use the core directly, without the Postgres adapter.

## Milestone 0: groundwork and feasibility

Settle the questions that could block everything else before writing any k-means code.

Feasibility decisions and the deferred GPU reference run are recorded in
[Milestone 0 findings](milestone_0_feasibility.md).
GPU availability does not block the CPU data path, Postgres integration, or CPU validation.
The GPU reference run and hardware validation belong to the later GPU milestone.

- Recorded: Confirm whether Vx can export a C ABI (a shared library with `extern "C"` entry points). Everything in Milestone 3 depends on this. Verified with a macOS arm64 C caller; Linux CUDA verification remains.
- Recorded: Decide the integration boundary. The C ABI works, and the k-means worker will run in a separate process because the current Vx runtime can abort.
- Recorded: Define the error boundary: Vx code must never unwind or longjmp into a Postgres backend. Write down how failures cross the ABI (return codes, error structs).
- Recorded: Check how Vx selects its dispatch backend. The README says `build.rs` picks it from what the build machine has, so a portable build made without CUDA ships only the CPU backend. Decide how an extension package should handle this.
- Recorded: Prepare the CPU development environment, Linux Vx compiler, and worker. See [M3 implementation and checks](milestone_3_cpu_integration.md).
- Recorded: Read pgpu's source for the centroid table schema and the exact `CREATE INDEX` options it passes to VectorChord, so our output matches the contract. Confirm the remaining root-parent discrepancy in a live run.

## Milestone 1: standalone k-means in Vx

Get a correct, benchmarkable k-means running with no Postgres involved. The parameters mirror pgpu's function signature, which is the practical spec.

The CPU library, its checks, and the compile-only GPU prototype are described
in [Milestone 1 notes](milestone_1_standalone.md).

- Recorded: Implement the core loop: assign points to nearest centroid, recompute centroids, repeat for `kmeans_iterations` (pgpu default 10).
- Recorded: Support the three distance operators: `ip`, `l2`, `cos`.
- Recorded: Add `spherical_centroids`: normalize centroids to the unit sphere (needed for `ip` and for unit-normalized datasets).
- Recorded: Add sampling: take up to `sampling_factor` distinct samples per centroid (pgpu default 256; its docs advise against values below 40).
- Recorded: Add `kmeans_nredo` restarts and keep the best run on the same sample.
- Recorded: Add two-level (hierarchical) clustering: `lists` as one value (flat) or two (root level, leaf level). The standalone API uses `root_count = 0` for flat.
- Recorded: Process sampled rows in batches when `batch_size` is smaller than the sample count. Input stays resident until Milestone 2 adds streaming.
- Recorded: Write correctness tests against an independent exhaustive CPU reference on small data: centroid quality, assignment agreement, determinism with a fixed seed.
- Recorded: Handle edge cases: empty clusters, fewer rows than centroids, duplicate vectors, dimensions 2000 and 3072.
- Recorded: Expose a plain library API (arrays in, centroids out) so it is reusable outside Postgres.

GPU execution and optimization are deferred to Milestone 6.
The current GPU prototype compiles to A100 code but emits no parallel launch metadata.

## Milestone 2: CPU host data path — complete

The host adapter streams source rows into bounded row-major float32 batches and trains Vx on a bounded sample.
It exposes a C reader callback for the M3 database wrapper. PostgreSQL itself is not implemented in this milestone.
See [Milestone 2 implementation and checks](milestone_2_data_path.md).

- Recorded: Implement contiguous float32 read batches, capped by `batch_size` and the declared row count.
- Recorded: Collect `min(row_count, leaf_count * sampling_factor)` distinct source rows without materializing the table.
- Recorded: Make seeded sampling independent of batch size. Document stable scan order and a shared count/scan snapshot as reader requirements.
- Recorded: Call the existing CPU Vx library with the collected sample. Match direct calls on that same sample for all metrics and tree shapes.
- Recorded: Validate dense `vector` rows, dimensions, distance options, null vectors, and nonfinite components, including unsampled rows.
- Recorded: Reject unsupported row types and source counts that change during the scan.
- Recorded: Check allocation arithmetic and enforce a budget before allocation, source reads, or Vx calls.
- Recorded: Budget sample, read batch, centroid output, scratch, assignments, source positions, and sampling hash storage.
- Recorded: Test batches of one row, batches larger than the source, partial batches, repeatability, and distinct sampling.
- Recorded: Measure actual allocations with injected failures. Verify cleanup and constant buffer storage as source size grows.
- Recorded: Expose read/validation, sampling, and clustering timings. Add a streaming benchmark and record local smoke results.

M3 implements database snapshot handling, SQL type checks, worker communication, and database-stage timings.
Additional vector types are follow-up work outside the M4–M7 priority sequence. GPU acceleration remains deferred to M6.

## Milestone 3: Postgres integration — CPU implemented

Wrap the CPU k-means in a pgrx extension that writes the centroid table and invokes VectorChord.
Implemented and validated without GPU hardware or CUDA libraries. See [M3 details](milestone_3_cpu_integration.md).

- Recorded: Build and validate a PostgreSQL 17 development image with pgvector, VectorChord, C/Rust tools, and PostgreSQL headers. See [setup and checks](postgres.md).
- Recorded: Create a pgrx extension crate targeting PostgreSQL 17 first. Extend version support after the first working integration.
- Recorded: Run the Vx CPU library in a separate worker through the existing C ABI. Return centroids or structured errors to Postgres.
- Recorded: Ensure extension loading and CPU worker startup work without CUDA libraries.
- Recorded: Route requests to the installed CPU worker. GPU selection and execution come in Milestone 6.
- Recorded: Emit a PostgreSQL WARNING when a build uses CPU because GPU acceleration is unavailable. Test successful CPU index creation and the warning.
- Recorded: Implement a SQL function modeled on `pgpu.create_vector_index_on_gpu(...)`: table, column, `lists`, `sampling_factor`, `batch_size`, `kmeans_iterations`, `kmeans_nredo`, `distance_operator`, `skip_index_build`, `spherical_centroids`, `residual_quantization`.
- Recorded: Implement the M2 row-reader callback for pgvector `vector` columns. Validate the declared SQL type and dimensions before scanning.
- Recorded: Count and scan rows in one database snapshot with a stable row order. Keep database reader storage bounded alongside the M2 buffer budget.
- Recorded: Measure table reads, worker communication, clustering, centroid writes, and index builds through the SQL flow.
- Recorded: Write the centroid table in the format VectorChord's external build expects (confirm against the VectorChord docs and pgpu's output).
- Recorded: Issue `CREATE INDEX ... USING vchordrq` with `build.external` pointing at the centroid table; support `skip_index_build` to stop after the centroids.
- Recorded: Convert invalid input, memory-budget rejection, worker crashes, and worker timeouts into Postgres errors.
- Recorded: Respect query cancellation and statement timeouts during CPU runs. Stop the worker and discard partial results.
- Recorded: Create centroid tables and indexes in one transaction. Verify rollback leaves no partial results after failure.
- Recorded: Keep CPU and later GPU options and centroid-table formats compatible. Numerical results need not be byte-identical.
- Recorded: Add executable SQL regressions through `scripts/check_extension.sh` and concurrent snapshot checks through `scripts/check_snapshot.py`. The current harness uses psql assertions.
- Recorded: Verify with `EXPLAIN` that queries use the resulting index.
- Recorded: Test the complete CPU flow for flat and two-level trees: source vectors, centroids, index build, and nearest-neighbor queries.
- Recorded: Test `skip_index_build` and a source column whose name is not `embedding`.
- Recorded: Document query probe settings matching the centroid tree depth. The image checks use `4` for flat and `2,4` for two-level trees.

## Milestone 4: end-to-end CPU tests and correctness validation

Updated November 4, 2026: 48 matched Faiss cases, 96 independent Lloyd cases,
25 native tests, and six zero/duplicate cases pass. The corrected worker passes
120 synthetic SQL recall queries and 3,072 real-data queries across 24 paired
Vx/internal index builds. See [M4 evidence and limits](milestone_4_cpu_correctness.md).
The internal-trainer objective comparison now passes six quality groups and 3,072
recall queries in an isolated instrumented image. See [objective evidence](internal_objectives.md).
M4 remains open for full kernel proofs and a hosted CI result.

The November 4 CPU/database results used Vx `7cc3d2d9`.
See [the historical results](evidence/vx_main_test_summary_2026-11-04.json).
Native builds, Linux builds, and CI now pin `ded61615` through [vx-revision.txt](../vx-revision.txt).
The local database runs that worker image and reports healthy.
Its data volume and database cluster identifier are unchanged.
Native correctness, both platforms' contract/code-generation checks, PostgreSQL integration,
snapshot behavior, and 120 synthetic recall queries pass on the new pin.
Real-data and internal-objective suites retain their November 4 evidence.
See [the November 5 audit and validation](vx_main_update_2026-11-05.md).

### Compiler upgrade and local database handoff

- Recorded: Pin native builds and CI to upstream Vx main at `ded616150af9a3372ad7e6badbc68ebb824c5c8b`. Preserve earlier evidence under its original revision.
- Recorded: Rebuild native and Linux kernels, install the new worker, and verify unchanged database identity on November 5. Record the capability audit and focused validation separately.
- Recorded: Build the native compiler/library and Linux kernel object from that pin. Record compiler, source, and object hashes.
- Recorded: Use borrowed metric enums and explicit validity, initialization, and copy non-overlap contracts. Require the flat code generator.
- Recorded: Reject packaging when the Linux object, kernel source, or compiler pin differs from its build manifest.
- Recorded: Build and install the PostgreSQL image with the new Linux object and packaged provenance. Verify startup, installed revision, and unchanged database identity. See [installation evidence](evidence/vx_main_install_2026-11-04.json).
- Recorded: Run the Linux/macOS contract and code-generation checks and the complete CPU/database pipeline against the upgraded worker. Preserve the new results separately from earlier evidence.


Establish correctness before changing arithmetic or adding GPU execution.
M1 through M3 provide reference checks and small integration smoke tests. Extend that evidence to representative inputs and final-index quality.
Collect timings for diagnosis and repeatability. The cuVS performance comparison belongs to M7.

- Recorded: Run small CPU extension checks for centroid output, flat/two-level indexes, actual index scans, snapshot consistency, and failure handling. See [M3 evidence](evidence/m3_cpu_2026-10-26.txt).
- Recorded: Extend end-to-end coverage across L2, inner product, cosine, spherical centroids, seeds, sampling factors, restarts, and batch sizes.
- Recorded: Cover representative dimensions, including 128, 768, 2000, and 3072, with duplicate vectors, empty clusters, and zero vectors.
- Recorded: Add [Faiss CPU k-means](https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering) as the primary external correctness reference. Keep performance comparisons outside this stage.
- Recorded: Extend known-answer synthetic checks with unequal-norm distributions and real GloVe-25/SIFT-128 subsets. Record source files and selected-array hashes.
- Recorded: Supply exactly the same float32 sample to both implementations. Disable additional reference-library subsampling or match it explicitly.
- Recorded: Match the distance metric, input normalization, spherical-centroid behavior, iterations, restarts, and stopping rules. Record differences that cannot be matched.
- Recorded: Address initialization explicitly. The current VxVector API accepts a seed, not initial centroids; equal seeds across libraries do not imply equal initialization.
- Recorded: Reconstruct Vx sample ordering and initialization for strict Faiss comparisons. Supply identical initial centroids to Faiss. No additional public initialization API is needed for these fixtures.
- Recorded: Match centroids up to permutation before comparing coordinates and assignments. Handle equal-distance ties explicitly.
- Recorded: Recompute clustering objectives independently from the returned centroids. Compare objective values, assignments, and centroids using declared numerical tolerances.
- Recorded: Validate the centroid tree separately: parent assignments, equal leaf depth, node counts, dimensions, and normalization. Reproduce VxVector's leaves-first hierarchy rather than assuming another hierarchical algorithm is equivalent.
- Recorded: Check empty clusters, duplicate vectors, and zero vectors separately. Align policies where possible and document expected differences.
- Historical candidate: Optionally cross-check L2 against [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with Lloyd's algorithm and explicit initialization.
- Recorded: Compute exact nearest neighbors with PostgreSQL sequential scans on the same source snapshot. Verify the ground-truth query plan does not use an approximate index.
- Recorded: Compare the final VectorChord index against that ground truth at fixed probes and query settings. Define tie handling, correctness tolerances, and a minimum recall target before SIMD work.
- Recorded: Compare Vx external centroid training with VectorChord's internal CPU trainer on the same real datasets, source snapshots, and held-out queries. Match available training and index settings; record differences in sampling, initialization, stopping rules, and empty-cluster handling rather than assuming equal seeds imply equal starts.
- Recorded: Repeat paired index builds with Vx seeds 0 and 17 and independent internal builds. Compare tie-aware recall across four probe/epsilon settings against exact scans. Apply predefined per-query conservative and mean relative quality gates. Record raw output and diagnostic timings.
- Recorded: Compare independently recomputed clustering objectives with VectorChord's internal trainer. An opt-in, source-hash-checked test image exports unrotated centroids. All six common-source leaf-objective groups pass the predefined 10% gate; coarse objectives remain diagnostics on differing leaf sets. See [procedure, results, and limits](internal_objectives.md). Initialization and sampling remain unmatched.
- Recorded: Preserve scalar reference results and fixtures for later SIMD and GPU regression checks.
- Recorded: Validate larger source tables and bounded sampling through SQL, including memory-budget rejection and concurrent writes.
- Recorded: Record datasets, seeds, versions, settings, and raw results. Distinguish native hardware from emulated development environments.
- Recorded: Implement CPU library, worker, contract-example, and end-to-end Postgres checks in CI without GPU or CUDA dependencies. Pin the compiler revision and dataset hashes.
- Historical candidate: Record a successful hosted run of the CPU CI workflow. Local component checks passed.

Different centroid coordinates alone do not prove a defect. K-means can reach different local optima when initialization or algorithm policies differ.
Strict differential checks require matched conditions. Broader quality comparisons must distinguish valid alternative solutions from correctness failures.
Faiss spherical clustering is a reference only where its normalization and assignment semantics match VxVector's chosen mode.

### Targeted compile-time verification in the Vx core

Use Vx invariants and Z3 where a proof prevents a realistic error or supplies a useful contract to callers.
Establish the reference-test baseline first. Add small structural proofs alongside that validation, with SIMD bounds proofs continuing in M5.
Do not make whole-algorithm verification a prerequisite for CPU correctness work.

- Recorded: Check native Vx contracts on both available compilers. Record the initial bounded-index, fixed-offset, and positive-count helpers. Preserve historical failure evidence for the nearest-loop and variable-offset examples.
- Recorded: Require a positive centroid count in `nearest` and prove that its result is a valid centroid index. This requires the Vx proof fixes described in [the proof update](m4_index_proofs_2026-11-05.md).
- Recorded: Carry the result contract into `accumulate_point`. Its numeric requirements check the centroid index at the real caller. Foreign allocation and initialization requirements remain trusted.
- Historical candidate: Prove distance and centroid buffer indices stay within the native buffer/view bounds, given valid dimensions and row/centroid indices.
- Historical candidate: Finish checking multiplication, addition, and offsets for machine-integer overflow. The widened `buffer_index` helper now proves row-major centroid offsets under bounded dimensions. Source sampling and metadata offsets remain open.
- Historical candidate: Establish that centroid division occurs only when the cluster count is positive. Keep the empty-cluster replacement path explicit.
- Historical candidate: Prove sampled row indices are within the source range. Consider sampling-distinctness proofs later if their benefit justifies the complexity.
- Recorded: State pointer preconditions on internal unsafe helpers separately from C ABI shape checks and caller allocation obligations. Foreign allocation sizes and lifetimes remain trusted obligations.
- Recorded: Add accepted and rejected helper-contract examples. The contract harness now checks six mutations. Kernel checks also reject an invalid nearest result, a missing caller bound, and an invalid buffer offset. Remaining kernel contracts stay open.
- Recorded: Record solver timings and unsupported obligations. The helper examples replace no kernel or foreign-buffer runtime checks. Retain those checks until actual kernel contracts pass.

Prefer these integer, indexing, and control-flow properties over attempting to prove clustering quality in Z3.
Finite inputs can still overflow floating-point accumulators. SIMD reassociation and FMA can also change rounding and assignments.
Continue independent objective calculations and final-index recall checks for numerical behavior.
Do not claim global optimality, convergence, or monotonic objectives across all metrics and empty-cluster policies without separate justification.
Defer annotations that add solver complexity without protecting a meaningful boundary or simplifying callers.

Exit criterion: the complete CPU flow passes the defined correctness and recall checks with reproducible evidence.

## Milestone 5: SIMD and Vx compiler/standard-library enhancements

Improve CPU computation while improving Vx's SIMD facilities alongside the application.
The current worker build omits `-O3`, and Vx defaults to `-O0`.
Earlier Linux inspection found scalar distance and centroid loops.
The upgraded kernel has no recorded profiling or packed-instruction inspection yet.
Inspect the new pinned build before drawing optimization conclusions.

- Historical candidate: Build the existing kernels with `-O3`. Inspect generated instructions and measure against the current scalar baseline before rewriting kernels.
- Historical candidate: Profile CPU clustering to identify the loops that justify SIMD work.
- Historical candidate: Compare scalar/SIMD assignment, bounded tiled GEMM plus argmin, and fused tiled distance-and-argmin. Inspect whether lowering materializes distance intermediates; see [Flash-KMeans assessment](#flash-kmeans-fused-assignment-and-grouped-updates).
- Recorded: Separate L2, inner-product, and cosine loops through borrowed metric-enum dispatch. Each computes only the required quantities. Performance measurement remains open.
- Historical candidate: Evaluate automatic vectorization, native Vx vector values, and inline `mlir!` for the hot loops. Record limitations with small compiler examples.
- Historical candidate: Enhance Vx vector loads/stores for existing float buffers with explicit alignment, bounds, and aliasing contracts.
- Historical candidate: Improve `std::simd` with useful load/store, arithmetic, reduction, and scalar-tail operations. Support appropriate lane widths and target capabilities.
- Historical candidate: Audit the current Rust-backed SIMD helpers for actual packed instructions, FFI overhead, pointer aliasing, and documented fused-operation behavior.
- Historical candidate: Prefer vector operations that remain in registers across the kernel. Avoid a separate FFI call for every small arithmetic operation.
- Historical candidate: Implement vectorized distance, centroid accumulation, and normalization where profiling supports them. Handle dimensions that are not lane multiples.
- Historical candidate: Apply targeted Vx bounds proofs to full SIMD loads and stores, their alignment requirements, and the scalar or masked tail. Use runtime checks where proof is unavailable.
- Historical candidate: Define floating-point reduction order, reassociation, and FMA behavior explicitly. Keep a scalar reference and avoid silently enabling broad fast-math assumptions.
- Historical candidate: Add Vx compiler and standard-library checks for generated SIMD instructions, scalar tails, unaligned buffers, numerical behavior, and supported CPU targets.
- Historical candidate: Rerun M4's correctness and recall checks. Measure kernel and total-build improvements without claiming cuVS competitiveness yet.
- Historical candidate: Record compiler and library changes with their application examples, validation, and remaining gaps. Carry the [concrete improvement candidates](#concrete-vx-compiler-and-standard-library-improvement-candidates) into this milestone; record evidence when an existing API already satisfies a candidate.

Exit criterion: verified SIMD execution and measured CPU improvement, while retaining the M4 correctness and recall targets.

## Milestone 6: GPU enablement and Vx enhancements

Enable GPU execution after CPU correctness and SIMD validation.
Develop the required compiler, runtime, and library support alongside the VxVector kernels.
GPU availability does not block the preceding CPU stages.

- Historical candidate: Adopt and validate existing `across` execution for VxVector assignment and reduction loops. CPU block parallelism and an Apple GPU backend now exist upstream. Record application launch metadata and native execution evidence.
- Historical candidate: Resolve tiled indexing, placed-buffer views, and pointer-offset gaps with focused Vx examples and compiler tests.
- Historical candidate: Add or improve device reductions and atomics where the chosen kernels require them.
- Historical candidate: Compare private partial sums, direct atomics, and assignment-grouped reductions, including sorting/index scratch and skewed cluster sizes. Audit reusable grouping and segmented-reduction support.
- Historical candidate: Evaluate points per thread and batched restarts; record registers, spills, shared memory, launch configuration, and numerical behavior. See [thread-coarsening experiments](#thread-coarsening-and-batched-clustering-runs).
- Historical candidate: Validate completion-safe double buffering with frozen centroids throughout each Lloyd iteration. Reusable async-copy tokens now compile upstream. Validate native overlap, transfer lifetimes, and cancellation separately. Apple GPU completion improvements do not establish portable asynchronous execution.
- Historical candidate: Link and call the Vx library on a Linux CUDA host. Validate execution on an NVIDIA target such as A100 or H100.
- Historical candidate: Add GPU initialization, sampling, restarts, and host-to-device transfers with the CPU path's options and centroid-table format.
- Historical candidate: Load GPU support optionally. Preserve CPU execution when GPU hardware or required GPU libraries are unavailable.
- Historical candidate: Select a supported GPU before execution. Otherwise select CPU and emit the fallback warning.
- Historical candidate: Report failures after GPU execution starts. Do not silently restart failed GPU work on CPU.
- Historical candidate: Test device-memory limits, GPU memory exhaustion, worker crashes, cancellation, and transaction rollback.
- Historical candidate: Validate GPU centroids and final-index recall against the CPU reference and M4 targets. Do not require byte-identical floating-point results.
- Historical candidate: Run the complete GPU SQL-to-index-to-query flow and retain raw evidence with driver, CUDA, GPU, and Vx versions.
- Historical candidate: Add GPU checks on an appropriate runner. Record the compiler and runtime gaps exposed by those checks.
- Historical candidate: Measure transfer and execution stages to guide pinned-memory, double-buffering, and other data-path improvements.

Exit criterion: working and validated GPU execution, with the CPU fallback still working independently of CUDA.

## Milestone 7: performance comparison against cuVS

Compare performance after GPU enablement passes the correctness checks.
Use the same datasets, hardware, clustering parameters, and query settings for Vx and cuVS.
Use [vsbt](https://github.com/EnterpriseDB/vsbt) or a compatible harness where practical.

- Historical candidate: Prepare pgpu/cuVS on the validated GPU host. Capture centroid rows, index definitions, query plans, and reference versions.
- Historical candidate: Add a Vx benchmark configuration with comparable sampling, iterations, restarts, metrics, and spherical-centroid settings.
- Historical candidate: Compare Vx GPU clustering against cuVS through pgpu or a controlled direct cuVS harness.
- Historical candidate: Separate clustering time from table reads, transfers, centroid writes, and index creation using `skip_index_build` and stage timings.
- Historical candidate: Compare final-index recall at fixed settings and performance at comparable recall. Retain the CPU implementation as a baseline.
- Historical candidate: Include VectorChord's internal CPU build and optionally faiss-gpu as additional baselines.
- Historical candidate: Cover datasets of different shapes: 1M x 128 (SIFT), 1M x 960 (GIST), and 5M x 768 (LAION, dot product), then larger.
- Historical candidate: Run repeated measurements on native hardware. Record memory use, variation, warm-up policy, versions, settings, and raw output.
- Historical candidate: Publish results, including regressions and cases where Vx loses. Turn measured bottlenecks into the next Vx compiler and library tasks.

Exit criterion: reproducible performance and recall comparisons against cuVS, with evidence supporting each claim.

## Additional input types — follow-up work

- Historical candidate: Add `halfvec` conversion to float32. Verify centroid output and VectorChord index compatibility.
- Historical candidate: Evaluate bounded `sparsevec` storage, distance behavior, and VectorChord compatibility before adding SQL support.
- Historical candidate: Evaluate `bit` separately. Define Hamming or Jaccard clustering and verify VectorChord support before promising a binary index path.
- Historical candidate: Test each added input type through the SQL function, centroid table, index build, and nearest-neighbor query.

## Vx capabilities to evaluate and gaps to propose

This assessment uses the [pinned Vx source](https://github.com/vx-lang/Vx/tree/ded616150af9a3372ad7e6badbc68ebb824c5c8b), updated November 5, 2026.
See [the November 5 capability audit](vx_main_update_2026-11-05.md) for evidence and limits.
Unchecked items describe remaining VxVector work. They do not imply that Vx lacks the feature.
Check existing APIs and backend support before proposing a language or library change.
Being expressible is only the starting point: this project should also produce
demonstrated Vx compiler and standard-library improvements in efficiency, safety,
resource control, and usability.

### November 5 main update: implemented features and remaining work

The new pin is `ded616150af9a3372ad7e6badbc68ebb824c5c8b`.
Upstream implementation and application adoption are separate milestones.
The checks below establish specific capabilities, not completed VxVector GPU support.

- Recorded: Verify safe pointer requirements, pointer postconditions, and `!frees` tracking with existing Vx fixtures. Use these facilities when extending buffer contracts.
- Recorded: Record a kernel baseline with `vxc --audit`. It reports 28 trusted pointer clauses at unsafe calls. Reducing those trust boundaries remains application work.
- Recorded: Verify `on Topology::Any` helpers and CPU block-parallel execution using existing execution fixtures. Kernel parallelization and deterministic centroid merging remain open.
- Recorded: Verify flat-codegen support for tensor dimensions that name local variables, and concurrent tensor-valued `tasks` blocks. The task fixture also passes with one shared-runtime worker. Use these for runtime-sized tiles and batched work before proposing replacements.
- Recorded: Verify reusable `lanes::copy_async` tokens with the upstream pipeline's PTX and FileCheck checks. Native CUDA overlap remains untested here.
- Recorded: Verify the Apple GPU backend with MSL generation and native Metal execution. The existing GPU fixture and its four-worker CPU copy produce identical checked output.
- Historical candidate: Evaluate an optional Apple GPU clustering experiment. The backend currently rejects f64 device types, atomics, dynamic shared memory, and vector operations. Preserve the f64-distance baseline rather than silently changing precision. This experiment does not replace M6's NVIDIA validation.
- Recorded: Resolve the reduced nearest-index and variable-offset proof gaps on the Vx proof branch. Adopt the nearest result contract and centroid offset helper in the kernel. See [validation and remaining boundaries](m4_index_proofs_2026-11-05.md).
- Historical candidate: Track pointer clauses in loop invariants and type invariants as proposals. The November 5 design documents are not implemented features.

The tensor, SIMD, and collective API files did not change across this upgrade.
Workspace reuse, fallible matrix allocation, producer/reduction fusion, grouped
reductions, and broader resource diagnostics remain open investigations.
The new pointer audit reports trusted facts, not allocation or register/spill statistics.

### Existing capabilities to adopt or validate

- Recorded: Use pointer validity, initialization, and copy non-overlap contracts in the CPU helpers. Use borrowed metric enums and require flat code generation. These unsafe contracts retain the C caller's allocation obligations.
- Historical candidate: Evaluate existing atomic types and lane reductions for GPU assignment and centroid accumulation. Check supported types, memory spaces, synchronization, and reduction scope. Validate cross-block counters, histograms, and argmin/argmax separately.
- Historical candidate: Evaluate `transfer_async`, its completion tokens, and `lanes::copy_async` for the GPU data path. Token reuse for tiled pipelines now compiles. Check host pinned-memory requirements, buffer lifetimes, native completion, and double buffering on the selected backend.
- Historical candidate: Evaluate the seeded generators in `std::rand`, including `Rng` and `SplitMix64`. Define parallel stream assignment and validate CPU/GPU reproducibility. Preserve the current sampling baseline until any change has matched reference evidence.
- Historical candidate: Audit existing `String`, host `HashMap`/`HashSet`, and lane sorting before requesting replacements. Assess generic types, device support, sort scope, and top-k requirements independently.

A lane is one participant in a parallel computation. A lane reduction combines the
participants' values into one result, such as a sum or maximum. For k-means,
lanes can compute partial distance sums and then combine them with `lanes::sum`.
The participating group determines the scope. A local reduction does not automatically
combine every GPU block's results. Cross-block aggregation needs another stage or
suitable atomics. Floating-point reduction order also affects rounding.

### Research work and language capability requirements

The research strategies below primarily need application algorithms and measurement.
Simple stopping, norm caching, recursive partitioning (including an explicit worklist),
local refinement, and validation bookkeeping can be expressed with existing loops,
scalar arithmetic, pointers, and budgeted buffers. These algorithms provide concrete
workloads for improving the compiler and library even when no new syntax is needed.
SOAR's immediate limitation is the VectorChord membership interface.

The pinned source already provides dynamic tensor views, `matmul_into`, and packed
CPU matrix kernels for f32/f64 in `std::tensor`. Backend examples cover caller-owned
output buffers. This is source-level evidence, not validation of the VxVector flat-codegen
path or a performance result. The matrix kernel also allocates packing scratch space;
caller-owned output alone does not make the operation allocation-free.

- Historical candidate: Validate existing tensor views and `matmul_into` with runtime-sized VxVector buffers, flat code generation, non-square tiles, transposed centroid layouts, and partial tiles. Local-variable tensor dimensions now pass an upstream flat-path execution fixture. Inspect application instructions and data copies.
- Historical candidate: Compare existing tensor matrix kernels, direct Vx SIMD loops, and a coarse C-ABI BLAS call as experimental baselines. Record backend dispatch, linking, packing, scratch allocation, and thread counts. An FFI baseline does not establish Vx-generated kernel performance.
- Historical candidate: Audit f64 SIMD widening/reductions, accumulation precision, FMA, and reassociation for the established distance contract; current four-lane f32 helpers alone do not cover it.
- Historical candidate: Determine whether reusable caller-budgeted packing scratch and recoverable matrix-allocation errors need library APIs. Document concrete gaps after testing existing paths.
- Historical candidate: Validate seeded random rotations using existing `std::rand` and matrix operations. Account for rotation construction/application costs, storage, inverse transformation, and CPU/GPU numerical differences separately from PRNG reproducibility.
- Historical candidate: For parallel assignment or centroid updates, evaluate existing CPU block workers (`VX_HOST_THREADS`), tensor-valued `tasks`, scheduling and reductions. Budget per-worker scratch and test deterministic merging and thread oversubscription before proposing concurrency changes.

Research-specific capability work extends the M5 SIMD and memory audits. Recall
validation inputs and checkpoint policy belong to VxVector's worker protocol; the
remaining buffer, sampling, and division proofs remain separate M4 verifier questions.

### Concrete Vx compiler and standard-library improvement candidates

Prioritize these application-driven candidates alongside the capability audits.
The observed needs are concrete; the exact API or compiler changes still require
small reproductions against the pinned compiler. An existing workaround does not
by itself establish that the language or library provides an efficient, safe interface.

| Candidate | Evidence or application need | Proposed Vx outcome |
| --- | --- | --- |
| Reusable, caller-budgeted matrix scratch | The inspected matrix kernel allocates packing storage internally; VxVector needs reuse across iterations and explicit memory admission | A way to determine scratch requirements and supply/reuse workspace, with documented ownership, alignment, and aliasing rules |
| Recoverable matrix-allocation failure | The worker must report resource exhaustion; a caller-owned output buffer does not control internal packing allocations | A fallible matrix operation or workspace path with defined error returns and partial-output behavior |
| Efficient SIMD over borrowed buffers | Distance loops need f32-to-f64 widening, reductions, masks, and tails; the existing four-lane f32 helpers do not establish that complete path | Safe or explicitly contracted buffer operations that keep intermediate vectors in registers and preserve the selected numerical behavior |
| Stronger loop and pointer verification | The proof branch resolves branch-result facts, loop-result invariants, and nonlinear integer offsets | Validate remaining source-buffer, sampling, and foreign allocation obligations |
| Dispatch and allocation observability | Performance comparisons must identify the actual backend, packing, copies, and scratch use | Reusable compiler/runtime diagnostics for the executed path and resource use, separated from VxVector's experiment metadata |
| Producer/reduction fusion | A matrix-based assignment path could introduce distance intermediates absent from scalar `nearest()` | Validate existing fusion; extend lowering or provide a tiled distance-and-argmin primitive where needed |
| Budgeted grouped reductions | Centroid updates combine integer-key grouping, indirect reads, and reductions | Reusable grouping/segmented sum-and-count operations with explicit workspace, precision, and reduction-order contracts |
| Scheduling and backend resource feedback | Thread coarsening and batched runs trade data reuse against register pressure and occupancy | Audit scheduling controls and expose actual registers, spills, shared memory, and launch metadata |
| Layout-aware composition and collectives | A legal device placement can still incur expensive redistribution or incompatible ownership layouts | Investigate layout/partition contracts and collective composition after auditing existing tensor and collective APIs |

- Historical candidate: Create a minimal Vx reproduction for each prioritized candidate. Record current behavior, the required behavior, and any workaround or hidden cost.
- Historical candidate: For matrix workspace and allocation errors, exercise repeated calls and constrained memory; determine whether existing APIs meet reuse, admission, and failure requirements before designing extensions.
- Historical candidate: For borrowed-buffer SIMD, compare generated instructions and numerical results with the scalar kernel. Include widening, unaligned input, partial tiles/tails, and near ties.
- Historical candidate: Resolve the verifier candidates using the M4 gap fixtures; separate invalid application contracts from missing compiler reasoning.
- Historical candidate: Audit existing dispatch/allocation diagnostics and identify missing information before proposing instrumentation APIs. Use the new trusted-pointer audit and Apple GPU launch/wait diagnostics where applicable. Allocation, register, and spill reporting remain separate requirements.
- Historical candidate: Build a minimal distance-producer/argmin-consumer example and inspect allocation, fusion, generated code, and peak scratch use before selecting a compiler or stdlib change.
- Historical candidate: Test integer-key grouping plus segmented sums/counts under skew and partial segments. Keep thread-local, block-wide, device-wide, and distributed operations distinct.
- Historical candidate: Audit current scheduling and backend resource-reporting controls with coarsened and batched kernels; distinguish declared memory capacity from measured register allocation and spills.
- Historical candidate: Audit `std::collective`'s existing f32/bf16 sum all-reduce and all-gather before proposing extensions. Investigate integer counts, precision, subgroups, reduce-scatter, argmin/index pairs, matching shapes, collective ordering, and failure behavior for an explicit distributed workload.
- Historical candidate: Implement confirmed compiler/library improvements with an application example and upstream regression checks; record measured performance or resource-control benefits where applicable.

Placement-aware scratch lifetimes and completion-safe asynchronous dispatch are
larger compiler/runtime investigations for M6. Apple GPU completion and command batching
now have upstream implementations. Portable overlap and VxVector integration remain incomplete;
these should extend that work rather than be presented as newly discovered features.
New syntax is an option only where existing types, APIs, and contracts cannot express
the required guarantees clearly. A useful outcome can instead be a better stdlib API,
stronger verifier reasoning, improved lowering, or observable resource behavior.

### Remaining requirements and candidate proposals

- Recorded: Investigate and resolve the nearest-index loop and variable-dimension offset proof gaps. The accepted examples now live directly under `tests/contracts/`. Vx changes include branch and loop facts, nonlinear arithmetic, and the value contract for `i64::from(i32)`. Remaining integration work is tracked above.
- Historical candidate: Complete the SIMD capability audit in M5: native vector loads/stores, arithmetic, reductions, alignment, aliasing, and scalar tails. The current `std::simd` helpers alone do not establish an efficient complete kernel.
- Historical candidate: Identify missing device-wide reductions, sorting, top-k, or hash operations after the GPU capability audit. Keep local lane operations separate from whole-device requirements.
- Historical candidate: Define concrete ragged and offsets-plus-values workloads before proposing additional irregular-data support. Strings already exist. Assess device support and bounded storage separately.
- Historical candidate: Evaluate a columnar / Arrow-style buffer module for a concrete database workload. Specify ownership, validity, layout, and conversion costs. Keep this follow-up work separate from the current dense-vector path.
- Historical candidate: Document C ABI compatibility, versioning, error conventions, and upgrade checks. Basic C export already works through VxVector's versioned `v1` API. Separate VxVector's policy from guarantees supplied by the compiler.
- Historical candidate: Define recoverable allocation failure where the application needs it. Low-level `std::alloc` allocation functions return null, while `Vec` can panic on allocation failure. VxVector already checks explicit buffer budgets and returns adapter allocation errors. Broader collection/device recovery and total process-memory limits remain separate requirements.
- Historical candidate: Evaluate the existing transfer-cost reasoning as an application planner input. It selects legal routes across declared edges. Compare its estimates with measured transfer costs before using them for runtime planning.

Prioritize the M4 proof gaps, M5 SIMD requirements, and M6 GPU data path.
Keep broader database-library proposals in the follow-up backlog.
For each confirmed gap, retain a small application example and an upstream regression check.
Record the existing capability, missing behavior, proposed change, and maintainer feedback.

## Backends and portability

Start with CPU support and a complete CPU index build. Add NVIDIA acceleration later, then evaluate other GPU backends.
Backend selection must preserve CPU execution without requiring CUDA libraries.

- Historical candidate: Ship and validate the CPU integration first. GPU support remains deferred until Milestone 6.
- Historical candidate: Keep the k-means code free of backend-specific calls so a new backend needs no algorithm changes.
- Historical candidate: Define a small backend test matrix: which operations (matmul, reductions, atomics, sorting) the algorithm needs, and check each against every new backend as it lands.
- Recorded: Connect the existing CPU library to the worker and extension. Keep it available as the fallback when GPU support arrives.
- Recorded: When the extension selects the CPU fallback because no supported GPU is available, emit a PostgreSQL WARNING that GPU acceleration is unavailable and CPU k-means is running. Loading the extension alone should not require a GPU. Test the warning and successful CPU index build on a host without a GPU.
- Historical candidate: Run M4 correctness checks and M7 performance comparisons on each added backend. Compare centroid quality and recall alongside speed.
- Historical candidate: Define optional GPU loading around Vx's build-time backend selection. Keep the CPU library independently loadable without GPU dependencies.
- Historical candidate: Document support honestly per backend (experimental, supported, unsupported).

## Production hardening

Keep worker isolation, cancellation, and transaction rollback in the first CPU integration.
The remaining packaging and operational work follows the first working demo.

- Historical candidate: Harden the separate worker selected in Milestone 0. Keep CPU and later GPU failures outside the Postgres backend.
- Historical candidate: Version the Vx ABI and the SQL function signature, and test upgrades with `ALTER EXTENSION ... UPDATE`.
- Historical candidate: Package for common platforms with `cargo pgrx package`, and document the CUDA and driver requirements.
- Historical candidate: Add observability: progress reporting during long builds, timings per stage, GPU memory use, clear log lines for backend selected.
- Historical candidate: Make runs reproducible: seeded sampling and fixed iteration counts, with the seed logged.
- Historical candidate: Preserve transactional failure behavior for CPU and later GPU runs. Failed runs must leave no partial centroid table or index.
- Historical candidate: Test on managed or restricted setups where GPUs and custom extensions may not be available, and document the limits.
- Historical candidate: Run CPU unit and end-to-end Postgres tests in CI without GPU hardware or CUDA libraries.
- Historical candidate: Add GPU tests on a labeled runner in Milestone 6, plus a regression benchmark on a fixed small dataset.

## Licensing and contribution hygiene

pgpu is AGPL-3.0, and VectorChord is dual-licensed under AGPLv3 or Elastic License v2. These terms affect how code can be reused and how a Vx-based extension could be distributed. This is a checklist of questions to settle, not legal advice.

- Historical candidate: Read the licenses of pgpu, VectorChord, pgvector and cuVS, and record what each allows.
- Historical candidate: Decide the approach to pgpu: write k-means against the documented table contract and VectorChord's public docs, without copying pgpu's code.
- Historical candidate: Choose a license for our extension and check it is compatible with how it will be used and distributed.
- Historical candidate: Get a qualified legal opinion before any commercial or hosted use.
- Historical candidate: Agree with Vx's core developer where this work lives (inside the Vx repo, or a separate repo that depends on it), who reviews it, and how proposals are handled.
- Historical candidate: Check your employer's policy on open-source contributions, since you work in this space professionally.
- Historical candidate: Keep a short contributing note: how to build, run tests and add a backend.

## Open questions and risks

The immediate risks are bounded input memory, worker failure handling, and VectorChord accepting the generated centroid tree.
GPU performance against cuVS is a later question and does not block CPU integration.

- Recorded: Vx exports the C ABI used by the standalone CPU library. Linux CUDA validation remains in Milestone 6.
- Historical candidate: Compare the validated CPU centroid-table contract with pgpu output in Milestone 7. Live CPU flat/two-level builds already passed in M3.
- Historical candidate: In Milestone 7, profile pgpu/cuVS clustering and database reads and writes to explain the performance comparison.
- Historical candidate: In Milestone 7, validate performance claims against cuVS at comparable recall. Make no GPU speed claim before measurement.
- Historical candidate: How stable is Vx's language and standard library while this is built? Agree how breaking changes are communicated.
- Historical candidate: Does the pgpu README's flow (VectorChord still builds the index from our centroids) match what EDB's docs describe as the GPU running the entire build? Settle this from the source or a test.
- Historical candidate: Which new Vx backends land first, and are the operations this algorithm needs available on them?

## Sources

- [pgpu README](https://github.com/EnterpriseDB/pgpu) and [Cargo.toml](https://github.com/EnterpriseDB/pgpu/blob/main/Cargo.toml)
- [EDB PGPU documentation](https://www.enterprisedb.com/docs/edb-postgres-ai/1.3/ai-factory/pgpu)
- [VectorChord overview](https://docs.vectorchord.ai/vectorchord/getting-started/overview.html) and [external build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)
- [EDB vsbt benchmark suite](https://github.com/EnterpriseDB/vsbt)
- [Vx repository](https://github.com/vx-lang/Vx) and [project site](https://vxlang.org/)
- [VectorChord GPU sidecar RFC](https://github.com/supervc-stack/VectorChord/discussions/476)


## Research assessment: faster clustering and retrieval quality — November 4, 2026

This dated assessment records research candidates and the reasons for their priority.
The [November 5 audit](vx_main_update_2026-11-05.md) updates its Vx capability assumptions.
It preserves the preceding roadmap and does not mark these candidates as implemented.
The recommendations below come from comparing the cited sources with the current VxVector code.
They are proposals for experiments, not measured VxVector improvements.
M4 correctness remains the prerequisite for the optimization work.

The strongest fits are faster exhaustive assignment, memory-efficient updates,
simple stopping, and recursive training for large leaf counts. The additional papers
make fusion, work per thread, and composition across operations explicit experiments.
Recall-based stopping needs validation inputs; SOAR requires index support beyond
the current centroid-training interface. Distributed execution remains later research.

| Candidate | Fit for VxVector | Recommended placement |
| --- | --- | --- |
| Tiled distance computation and centroid layouts | Directly targets the exhaustive assignment loop | Evaluate in M5 |
| Flash-KMeans fused distance/argmin | Preserves the current streaming minimum while enabling tiled computation | CPU experiment in M5; GPU implementation in M6 |
| Assignment-grouped centroid updates | Trades sorting/index scratch for fewer contended global updates | Compare update strategies in M6 |
| Thread coarsening and batched restarts | Reuses input and amortizes reductions, subject to register/scratch limits | M6 experiment; Multi-K tuning is optional follow-up |
| Communication-aware layout composition | Avoids redistribution between successive distributed operations | Later multi-device research |
| Residency-aware plans with quality audits | Combines placement, measured costs, and bounded approximation checks | Application research hypothesis; not an implemented Vx planner |
| Simple convergence stopping | Uses existing loops, buffers, and scalar arithmetic | Evaluate with the optimized baseline |
| Stop when retrieval quality stabilizes | Requires a validation-data contract for the sample-only worker | Evaluate after simpler stopping; retain final-index checks |
| Recursive clustering with local refinement | Reduces training work as leaf count grows | Prioritize after the baseline when targeting thousands of leaves |
| SuperKMeans adaptive dimension pruning | Promising for large, high-dimensional workloads | Prototype after the optimized baseline |
| SOAR multiple partition assignments | Needs index storage and query support | Separate VectorChord integration investigation |

### Current implementation and integration boundary

[`nearest()` in the CPU kernel](../src/kmeans.vx) compares each point with every
centroid across every dimension. `train()` divides samples into batches but still
performs these comparisons individually. It runs a fixed iteration count and
selects the winning restart by the sample objective.

The two-level implementation trains all leaves globally, then clusters those leaves
into roots. This produces a hierarchical index description, but does not reduce
the cost of global leaf training. Its dominant leaf-assignment work is approximately:

```text
restarts × iterations × sampled rows × leaf count × dimensions
```

The sample limit grows with `leaf_count * sampling_factor` until it reaches the
source row count. Sampling therefore does not remove the scaling problem as leaf
count grows.

The [data-path adapter](../src/data_path.c) trains on a bounded sample.
Its assignments describe that sample, not the full source table.
The [worker](../src/worker.c) returns centroid vectors and parents, together with
status and timing metadata. It does not return the sample assignments.
The [PostgreSQL bridge](../pg_vxvector/src/bridge.c) writes the centroid table and
invokes VectorChord through `build.external` to index the full source table.
VectorChord owns index storage and query execution.

### Tiled exhaustive assignment and SuperKMeans pruning

SuperKMeans combines matrix multiplication over some dimensions with progressive
pruning over the remaining dimensions. It uses random orthogonal rotation, a
specialized centroid layout, and a threshold derived from the previous assignment.
Its adaptive pruning is probabilistic, not a guarantee of identical assignments.
See [SuperKMeans, section 3](https://arxiv.org/html/2603.20009v1#S3).

For VxVector, first evaluate tiled exhaustive assignment: reuse point and centroid
batches through SIMD or matrix multiplication while considering every centroid.
This establishes an optimized baseline before introducing probabilistic pruning.
The current build scripts omit `-O3`. M5 already calls for optimized-build inspection
and profiling before rewriting kernels.

Candidate work:

- Historical candidate: Build the existing kernel with optimization and inspect its generated instructions.
- Historical candidate: Profile assignment, centroid accumulation, and normalization independently; separate kernel time from sampling, worker transport, and full index construction.
- Historical candidate: Evaluate point and centroid tiling, packed centroid layouts, and SIMD or matrix multiplication. Audit existing Vx tensor views, `matmul_into`, and packed f32/f64 CPU kernels before proposing new matrix APIs.
- Historical candidate: Benchmark cached point norms and per-iteration centroid norms for cosine distance. Preserve zero-vector behavior and numerical tolerances.
- Historical candidate: Compare simple convergence stopping with fixed iterations: unchanged assignments or small centroid movement, with a maximum iteration count and explicit empty-cluster repair behavior.
- Historical candidate: Include distance tiles, packed centroids, cached norms, convergence state, and library-internal packing allocations in memory accounting. Check allocation failure and reuse of scratch buffers.
- Historical candidate: Retain the scalar reference and rerun the numerical and final-index recall checks.
- Historical candidate: Prototype adaptive pruning only after measuring the exhaustive baseline.

The current kernel uses float64 distance intermediates and float32 centroid storage
and accumulation. Matrix multiplication can change rounding and cancellation through distance
expansion, FMA, and reduction order even when every centroid is considered. Float32
intermediates can also change overflow behavior relative to the current float64 distances. Retain large-input, near-tie, normalization, and centroid-sum overflow checks.
Do not silently weaken the numerical behavior established in M4.

Start an adaptive-pruning experiment with L2. Arbitrary inner product and
non-spherical cosine need separate treatment and validation. Include rotation and
other scratch buffers in memory planning. Return centroids in the original coordinate
system before the VectorChord handoff.

### Flash-KMeans: fused assignment and grouped updates

[Flash-KMeans, arXiv:2603.09229v2](https://arxiv.org/html/2603.09229v2),
April 9, 2026, studies Euclidean Lloyd k-means. Its FlashAssign computes tiled
distances and running argmin together without storing the full distance matrix.
Its sort-inverse update sorts assignment indices, gathers points in logical group
order, and reduces locally. Segment results still use global atomic merges; the
method reduces contention rather than eliminating every atomic operation.

VxVector's scalar `nearest()` already keeps only the best distance/index. The adoption
question is how to retain that property while increasing throughput. A separate GEMM
and reduction may introduce intermediate storage that the current kernel avoids.

- Historical candidate: Compare optimized scalar/SIMD, bounded GEMM tiles plus argmin, and fused tiled assignment. Measure intermediate writes, copies, scratch, and execution time.
- Historical candidate: Investigate producer/consumer fusion or a reusable stdlib operation using runtime dimensions and borrowed buffers; first inspect existing lowering.
- Historical candidate: Preserve tie policy, finite-value checks, large-input behavior, and the selected precision. Exhaustive or mathematically exact assignment does not guarantee identical floating-point results.
- Historical candidate: Compare direct atomics, private partial sums, and grouped/segmented centroid updates on uniform and skewed clusters. Include grouping time and workspace, not just the final reduction.
- Historical candidate: Audit device-wide integer grouping, argsort, scans, and segmented reductions separately from local lane operations. Define workspace and allocation-error behavior.
- Historical candidate: Evaluate cache/shape-based tile selection against measured configurations. Record hardware, compiler, precision, and configuration in benchmark provenance.

The paper's GPU results are not expected VxVector gains. Generalization to inner
product and cosine requires metric-specific checks. Out-of-core overlap also needs
validated completion and lifetime behavior in Vx; keep old centroids fixed until all
chunks have contributed to an iteration's new sums and counts.

### Thread coarsening and batched clustering runs

[Barros et al.: Rethinking Per-Thread Computation for Machine Learning Design
Exploration](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.70744),
June 2, 2026, combines multiple points per thread, hierarchical local reductions,
and concurrent evaluation of several K values. It discusses register spilling and
partitions reductions when shared-memory demand exceeds capacity. Its low-dimensional,
small-K Multi-K experiments do not establish performance for VxVector's embedding shapes.

- Historical candidate: Sweep points per thread and launch geometry with register/spill, shared-memory, occupancy, bandwidth, and time measurements. Include tail and synchronization checks.
- Historical candidate: Compare thread-local accumulation followed by block/global merging against other update strategies. Preserve the chosen reduction-order and overflow contract.
- Historical candidate: Benchmark independent restarts sequentially and in batches over the same resident sample. Keep initialization, centroid state, accumulators, and best-run selection separate for each restart.
- Historical candidate: Budget batch size using dimensions, centroid count, precision, and scratch; avoid assuming all accumulators remain in registers. At K=32 and dimension=128, f32 sums plus one f32 count per centroid alone occupy 16,512 bytes per thread.
- Historical candidate: Evaluate Multi-K search only as optional index tuning. Preserve each configuration's sampling/initialization semantics; a shared maximum-sized sample would change smaller-K runs unless explicitly designed and validated.
- Historical candidate: Select K using held-out recall versus query latency/candidate count, build time, and memory. Raw objectives across different K values are not a sufficient selection criterion.

Batched restarts are the first application experiment because they reuse an existing
VxVector operation. Classifier encoders, Gini-based screening, and XGBoost pruning target
a different quality objective and remain outside this vector-index work.

### Communication-aware composition across operations

[Communication-Avoiding Linear Algebraic Kernel K-Means on GPUs,
arXiv:2601.17136v2](https://arxiv.org/html/2601.17136v2), January 28, 2026,
chooses distributions jointly for consecutive dense/sparse operations. Its 1.5D
strategy avoids redistributing the large kernel matrix and arranges outputs for
local assignment updates, using selective replication and collective reductions.

Kernel k-means is not a drop-in replacement for our ordinary centroid training:
its N-by-N kernel matrix changes the problem and resource requirements. Implicit
feature-space centroids do not generally provide ordinary input-space centroid
vectors for the current VectorChord handoff. Borrow the composition principle.

- Historical candidate: For a later multi-device Lloyd experiment, start with resident point shards, replicated centroids, and reduction of partial sums/counts. Establish a simple baseline before complex distributions.
- Historical candidate: Compare complete assignment/update plans including layout conversions, collectives, replication, synchronization, and peak per-device memory.
- Historical candidate: Audit existing Vx collectives and layout/partition representations with a small composed workload. Legal memory access alone does not establish compatible shard ownership or layout.
- Historical candidate: Investigate consumer-aware dispatch: whether a producer can retain a layout or fuse with its reduction instead of independently selecting each operation's fastest implementation.
- Historical candidate: Keep distributed scaling separate from the current single-device M6 exit criterion and from the later cuVS comparison. Do not transfer the paper's speedup claims to VxVector.

### Residency-aware dispatch, provenance, and quality audits

These are application research hypotheses extending the papers, not claims of new
research or implemented Vx capabilities. Memory provenance describes placement and
access legality; experiment provenance records inputs, builds, plans, and outcomes.
Neither alone supplies a runtime strategy selector or a quality guarantee.

- Historical candidate: Compare whole training-phase plans that retain samples and reusable layouts across iterations. Calibrate compute, packing, transfer, launch, and synchronization costs before switching devices or strategies.
- Historical candidate: For approximate pruning, periodically compare with exhaustive assignment on a deterministic audit subset. Record disagreements, distance penalties, objective changes, and held-out partition recall separately from final index recall.
- Historical candidate: Evaluate tightening pruning or returning to exhaustive assignment when predefined audit limits are exceeded. A subset audit is evidence, not a guarantee for all rows; keep this distinct from fallback after a GPU execution failure.
- Historical candidate: Evaluate recursive group sizes and tiles chosen jointly with explicit memory admission, including packing and validation/checkpoint scratch.
- Historical candidate: Record plan/configuration, compiler/kernel hashes, input hashes, seeds, precision, backend, transfers, and audit-triggered changes. Preserve enough information to replay strategy decisions.
- Historical candidate: Audit whether Vx needs stronger layout, scratch-lifetime, or completion contracts for these experiments. Extend existing overlap work rather than assuming `spawn on` already provides an asynchronous pipeline.

### Recall-based stopping

SuperKMeans offers early termination when recall improvements fall below a threshold,
with two iterations of patience. This requires queries and exact-neighbor ground
truth. See [SuperKMeans, section 3.3](https://arxiv.org/html/2603.20009v1#S3.SS3).

VxVector could evaluate partition recall inside the worker after each iteration.
Partition recall measures whether the searched partitions contain the exact nearest
neighbors. It does not require rebuilding a PostgreSQL index every iteration.
The current worker receives a bounded training sample, so full-source validation
requires additional inputs. Sample-only ground truth must be labeled as such; it
does not establish partition recall over the full database.

Candidate work:

- Historical candidate: Benchmark simple convergence stopping before adding recall-based stopping.
- Historical candidate: Define whether validation ground truth covers the training sample or the full source. For full-source checks, supply representative queries and exact-neighbor vectors or equivalent assignment information from the same source snapshot.
- Historical candidate: Design and version the worker validation-input contract; include query/neighbor buffers, lifetimes, and memory-budget checks.
- Historical candidate: Use a small representative validation query set with exact-neighbor ground truth.
- Historical candidate: Record partition recall, cluster sizes, and candidate counts after each iteration.
- Historical candidate: Add an improvement threshold and patience while retaining a maximum iteration count. Evaluate an absolute recall requirement so a low-quality plateau does not count as success.
- Historical candidate: Retain the best validation checkpoint and evaluate how to handle noisy or nonmonotonic recall. Account for its centroid storage.
- Historical candidate: Keep fixed-iteration training available for matched reference tests.
- Historical candidate: Use separate queries for stopping decisions and final quality evaluation.
- Historical candidate: Measure validation overhead as part of training time and buffer use.

Partition recall is a proxy for final index quality. VectorChord routing and
quantization also affect results. Validate the final index at practical probe and
epsilon settings before adopting a stopping policy. Preserve the existing sample
objective as a diagnostic; any change to restart selection needs its own comparison.

### Recursive training followed by local refinement

Elastic recursively splits groups, then refines assignments using nearby-centroid
shortlists. This refinement helps correct mistakes from early coarse partitions.
Balanced recursion improves scaling, although constructing centroid neighborhoods
can still involve quadratic work.
See [Elastic's recursive clustering and refinement](https://www.elastic.co/search-labs/blog/k-means-for-vector-indices#breaking-the-scaling-barrier).

This is a promising algorithmic experiment when VxVector targets thousands of leaves.
It differs from the current leaves-first construction described above. Prioritize
this experiment ahead of recall-based stopping when large leaf counts are the
target workload. With sample size approximately `sampling_factor * leaf_count`,
exhaustive assignment grows quadratically in leaf count until source size caps the
sample; improving SIMD throughput alone does not remove that growth.

Candidate work:

- Historical candidate: Add an optional recursive training strategy while preserving the existing Lloyd baseline.
- Historical candidate: Allocate leaf counts among groups explicitly to preserve the requested total.
- Historical candidate: Evaluate local refinement using nearby-centroid candidate lists.
- Historical candidate: Measure neighborhood construction time and memory as part of the strategy.
- Historical candidate: Initially export a flat set of leaf centroids to isolate training quality from query routing.
- Historical candidate: Evaluate two-level output separately after establishing the flat-index results.

Directly exporting an adaptive-depth training tree needs additional design.
VectorChord requires equal root-to-leaf depth in its external centroid tree.
See the [external-build format](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html#format).
Training recursion and exported index depth need not be identical.

### SOAR and the index integration boundary

SOAR assigns each vector to an additional partition. It chooses that assignment to
reduce correlated retrieval errors rather than simply selecting the second-nearest
centroid. Its derivation targets maximum inner product search. Overlapping partitions
require query-result deduplication.
See [SOAR, sections 2–3](https://arxiv.org/pdf/2404.00774).

Changing sample assignments in `kmeans.vx` would not enable SOAR in the current
PostgreSQL flow. Those assignments never cross the worker boundary. The centroid
handoff does not express multiple partition memberships for source rows.
The existing `residual_quantization` option does not supply that interface.

A separate VectorChord investigation would need to establish:

- Historical candidate: A supported path for computing and retaining multiple assignments for full-source vectors.
- Historical candidate: Index storage for additional partition entries and their memory cost.
- Historical candidate: Query deduplication and the cost of repeated candidates.
- Historical candidate: Consistent insert, update, delete, and index-maintenance behavior.
- Historical candidate: Metric-specific validation before extending the approach beyond its evaluated setting.

Keep SOAR separate from the immediate centroid-training work until this integration
path is established. Do not infer support from the presence of residual quantization.

### Benchmark scope and experiment order

The current real-data baseline uses 8,192 source rows, 32 leaves, and
GloVe-25/SIFT-128. It provides bounded correctness evidence, not evidence for large,
high-dimensional clustering performance. Database timings under x86 emulation remain
diagnostic. See [M4 validation and limits](milestone_4_cpu_correctness.md).

SuperKMeans reports little benefit at very small centroid counts and stronger gains
on larger workloads. Its headline speedups are not expected VxVector gains.
See [SuperKMeans evaluation](https://arxiv.org/html/2603.20009v1#S4.SS3).

Extend the experiments before selecting an algorithm:

- Historical candidate: Declare target source sizes, dimensions, leaf counts, memory limits, and recall/query-latency requirements before selecting an algorithm.
- Historical candidate: Use higher-dimensional real embeddings and increasing leaf counts.
- Historical candidate: Run performance measurements on native hardware with recorded compiler and library settings.
- Historical candidate: Measure clustering time and total SQL-to-index build time separately.
- Historical candidate: Compare recall against query latency and the number of candidates examined.
- Historical candidate: Record cluster-size distributions and peak memory alongside the explicit buffer budget.
- Historical candidate: Use repeated runs and retain datasets, seeds, settings, and raw results.
- Historical candidate: Preserve correctness checks independently of performance comparisons.

Preferred experiment order:

1. Establish the optimized exhaustive CPU baseline and native profiling through M5.
2. Evaluate tiling/SIMD, fused distance-and-argmin, cosine norm caching, and simple convergence stopping in M5.
3. In M6, compare update strategies, work per thread, and batched restarts against a validated single-device baseline.
4. Evaluate recursive training with local refinement when targeting large leaf counts; define the validation-input contract before recall-based stopping.
5. Prototype adaptive dimension pruning against optimized exhaustive assignment, then evaluate audit-driven approximation policies.
6. Evaluate residency-aware phase plans after measuring transfer and execution costs. Keep distributed layout/collective experiments as later research.
7. Investigate SOAR separately with VectorChord integration support.

These experiments supplement the existing M4–M7 sequence. They do not establish GPU
execution or replace the later cuVS comparison. Decide which candidates to retain
from measured build time, retrieval quality, query cost, and memory use.

### References for this assessment

External research and interface documentation:

- [Flash-KMeans: Fast and Memory-Efficient Exact K-Means, arXiv:2603.09229v2](https://arxiv.org/html/2603.09229v2) ([PDF](https://arxiv.org/pdf/2603.09229v2)), April 9, 2026.
- [Communication-Avoiding Linear Algebraic Kernel K-Means on GPUs, arXiv:2601.17136v2](https://arxiv.org/html/2601.17136v2) ([PDF](https://arxiv.org/pdf/2601.17136v2)), January 28, 2026.
- [Rethinking Per-Thread Computation for Machine Learning Design Exploration: A Work-Efficient GPU Strategy for K-Means and XGBoost](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.70744) ([PDF](https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.70744)), June 2, 2026, DOI: 10.1002/cpe.70744.

- [A Super Fast K-means for Indexing Vector Embeddings, arXiv:2603.20009v1](https://arxiv.org/html/2603.20009v1), March 20, 2026.
- [Elastic: K-means for building vector indices](https://www.elastic.co/search-labs/blog/k-means-for-vector-indices#prior-art), June 30, 2025.
- [SOAR: Improved Indexing for Approximate Nearest Neighbor Search, arXiv:2404.00774](https://arxiv.org/pdf/2404.00774).
- [VectorChord external index precomputation and centroid-table format](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html).

Local design, validation, and implementation references:

- [README: architecture, options, current limits, and test entry points](../README.md).
- [M1: standalone library and compile-only GPU prototype](milestone_1_standalone.md).
- [M2: bounded sampling and CPU data path](milestone_2_data_path.md).
- [M3: CPU worker and PostgreSQL integration](milestone_3_cpu_integration.md).
- [M4: CPU correctness, real-data recall, and evidence limits](milestone_4_cpu_correctness.md).
- [Vx main update: compiler pin, kernel contracts, and validation](vx_main_update_2026-11-04.md).
- [M5: SIMD and compiler work](#milestone-5-simd-and-vx-compilerstandard-library-enhancements).
- [M6: GPU enablement](#milestone-6-gpu-enablement-and-vx-enhancements).
- [M7: performance comparison against cuVS](#milestone-7-performance-comparison-against-cuvs).
- [CPU k-means kernel](../src/kmeans.vx), [native build](../scripts/build.sh), and [Linux worker build](../scripts/build_linux_worker.sh).
- [Data-path adapter](../src/data_path.c) and [public buffer and assignment contract](../include/vxvector_data_path.h).
- [Worker output](../src/worker.c) and [PostgreSQL centroid-table/index handoff](../pg_vxvector/src/bridge.c).
- [Real-data recall harness](../scripts/check_real_recall.py).

## Future experiments

These are optional research directions outside the current milestone commitments.

### Learned sparse multi-vector retrieval (SSR)

SSR learns sparse token representations and searches inverted posting lists rather
than training a centroid codebook. It is an alternative representation and retrieval
pipeline, requiring a learned projector and sparse scoring/index support beyond
VxVector's external-centroid handoff. Its retrieval results do not establish a replacement
for arbitrary dense-vector nearest-neighbor search.

- Historical candidate: Evaluate the authors' implementation with a frozen encoder/projector on a small document corpus before implementing kernels in Vx.
- Historical candidate: Establish exact sparse token-level MaxSim as a reference; compare indexed retrieval and pruning against it. Distinguish coordinate-wise document maxima, candidate scores, and exact token scores. Validate omitted-coordinate bounds before claiming safe pruning.
- Historical candidate: Measure relevance with held-out judgments separately from dense nearest-neighbor recall. Include encoding/projection, index construction, query latency, memory, and model training costs in comparisons.
- Historical candidate: Investigate Vx workloads for projection fused with top-k selection, borrowed sparse buffers (offsets/indices/values), sparse intersection, and segmented MaxSim. Audit existing facilities before proposing new APIs.
- Historical candidate: Record encoder/projector hashes, sparsity policy, and feature numbering with the index. Evaluate compatibility and rebuild requirements when the representation changes.

This experiment does not change the milestones or their exit criteria. Bounded
reconstruction and score-distortion claims depend on the paper's assumptions;
empirical relevance gains do not guarantee preservation of every dense-vector ranking.

References:

- [No More K-means: Single-Stage Sparse Coding for Efficient Multi-Vector Retrieval, arXiv:2605.30120v4](https://arxiv.org/html/2605.30120v4) ([PDF](https://arxiv.org/pdf/2605.30120v4)), September 28, 2026.
- [Official SSR implementation](https://github.com/Y-Research-SBU/SSR).
