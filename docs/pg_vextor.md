# Vx × Postgres: k-means for VectorChord — TODOs

Oct 20, 2026 · @desia

Plan updated Oct 26, 2026: validate CPU correctness, add SIMD with Vx compiler and standard-library improvements,
enable GPU execution, then compare performance against cuVS.

## Goal and scope

Build a Vx k-means implementation for VectorChord's external build, wrapped in a pgpu-style Postgres function.
Develop Vextor and Vx together. Application kernels should expose compiler and standard-library gaps as each stage progresses.
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

## Milestone 0: groundwork and feasibility

Settle the questions that could block everything else before writing any k-means code.

Feasibility decisions and the deferred GPU reference run are recorded in
[Milestone 0 findings](milestone_0_feasibility.md).
GPU availability does not block the CPU data path, Postgres integration, or CPU validation.
The GPU reference run and hardware validation belong to the later GPU milestone.

- [x] Confirm whether Vx can export a C ABI (a shared library with `extern "C"` entry points). Everything in Milestone 3 depends on this. Verified with a macOS arm64 C caller; Linux CUDA verification remains.
- [x] Decide the integration boundary. The C ABI works, and the k-means worker will run in a separate process because the current Vx runtime can abort.
- [x] Define the error boundary: Vx code must never unwind or longjmp into a Postgres backend. Write down how failures cross the ABI (return codes, error structs).
- [x] Check how Vx selects its dispatch backend. The README says `build.rs` picks it from what the build machine has, so a portable build made without CUDA ships only the CPU backend. Decide how an extension package should handle this.
- [x] Prepare the CPU development environment, Linux Vx compiler, and worker. See [M3 implementation and checks](milestone_3_cpu_integration.md).
- [x] Read pgpu's source for the centroid table schema and the exact `CREATE INDEX` options it passes to VectorChord, so our output matches the contract. Confirm the remaining root-parent discrepancy in a live run.

## Milestone 1: standalone k-means in Vx

Get a correct, benchmarkable k-means running with no Postgres involved. The parameters mirror pgpu's function signature, which is the practical spec.

The CPU library, its checks, and the compile-only GPU prototype are described
in [Milestone 1 notes](milestone_1_standalone.md).

- [x] Implement the core loop: assign points to nearest centroid, recompute centroids, repeat for `kmeans_iterations` (pgpu default 10).
- [x] Support the three distance operators: `ip`, `l2`, `cos`.
- [x] Add `spherical_centroids`: normalize centroids to the unit sphere (needed for `ip` and for unit-normalized datasets).
- [x] Add sampling: take up to `sampling_factor` distinct samples per centroid (pgpu default 256; its docs advise against values below 40).
- [x] Add `kmeans_nredo` restarts and keep the best run on the same sample.
- [x] Add two-level (hierarchical) clustering: `lists` as one value (flat) or two (root level, leaf level). The standalone API uses `root_count = 0` for flat.
- [x] Process sampled rows in batches when `batch_size` is smaller than the sample count. Input stays resident until Milestone 2 adds streaming.
- [x] Write correctness tests against an independent exhaustive CPU reference on small data: centroid quality, assignment agreement, determinism with a fixed seed.
- [x] Handle edge cases: empty clusters, fewer rows than centroids, duplicate vectors, dimensions 2000 and 3072.
- [x] Expose a plain library API (arrays in, centroids out) so it is reusable outside Postgres.

GPU execution and optimization are deferred to Milestone 6.
The current GPU prototype compiles to A100 code but emits no parallel launch metadata.

## Milestone 2: CPU host data path — complete

The host adapter streams source rows into bounded row-major float32 batches and trains Vx on a bounded sample.
It exposes a C reader callback for the M3 database wrapper. PostgreSQL itself is not implemented in this milestone.
See [Milestone 2 implementation and checks](milestone_2_data_path.md).

- [x] Implement contiguous float32 read batches, capped by `batch_size` and the declared row count.
- [x] Collect `min(row_count, leaf_count * sampling_factor)` distinct source rows without materializing the table.
- [x] Make seeded sampling independent of batch size. Document stable scan order and a shared count/scan snapshot as reader requirements.
- [x] Call the existing CPU Vx library with the collected sample. Match direct calls on that same sample for all metrics and tree shapes.
- [x] Validate dense `vector` rows, dimensions, distance options, null vectors, and nonfinite components, including unsampled rows.
- [x] Reject unsupported row types and source counts that change during the scan.
- [x] Check allocation arithmetic and enforce a budget before allocation, source reads, or Vx calls.
- [x] Budget sample, read batch, centroid output, scratch, assignments, source positions, and sampling hash storage.
- [x] Test batches of one row, batches larger than the source, partial batches, repeatability, and distinct sampling.
- [x] Measure actual allocations with injected failures. Verify cleanup and constant buffer storage as source size grows.
- [x] Expose read/validation, sampling, and clustering timings. Add a streaming benchmark and record local smoke results.

M3 implements database snapshot handling, SQL type checks, worker communication, and database-stage timings.
Additional vector types are follow-up work outside the M4–M7 priority sequence. GPU acceleration remains deferred to M6.

## Milestone 3: Postgres integration — CPU implemented

Wrap the CPU k-means in a pgrx extension that writes the centroid table and invokes VectorChord.
Implemented and validated without GPU hardware or CUDA libraries. See [M3 details](milestone_3_cpu_integration.md).

- [x] Build and validate a PostgreSQL 17 development image with pgvector, VectorChord, C/Rust tools, and PostgreSQL headers. See [setup and checks](postgres.md).
- [x] Create a pgrx extension crate targeting PostgreSQL 17 first. Extend version support after the first working integration.
- [x] Run the Vx CPU library in a separate worker through the existing C ABI. Return centroids or structured errors to Postgres.
- [x] Ensure extension loading and CPU worker startup work without CUDA libraries.
- [x] Route requests to the installed CPU worker. GPU selection and execution come in Milestone 6.
- [x] Emit a PostgreSQL WARNING when a build uses CPU because GPU acceleration is unavailable. Test successful CPU index creation and the warning.
- [x] Implement a SQL function modeled on `pgpu.create_vector_index_on_gpu(...)`: table, column, `lists`, `sampling_factor`, `batch_size`, `kmeans_iterations`, `kmeans_nredo`, `distance_operator`, `skip_index_build`, `spherical_centroids`, `residual_quantization`.
- [x] Implement the M2 row-reader callback for pgvector `vector` columns. Validate the declared SQL type and dimensions before scanning.
- [x] Count and scan rows in one database snapshot with a stable row order. Keep database reader storage bounded alongside the M2 buffer budget.
- [x] Measure table reads, worker communication, clustering, centroid writes, and index builds through the SQL flow.
- [x] Write the centroid table in the format VectorChord's external build expects (confirm against the VectorChord docs and pgpu's output).
- [x] Issue `CREATE INDEX ... USING vchordrq` with `build.external` pointing at the centroid table; support `skip_index_build` to stop after the centroids.
- [x] Convert invalid input, memory-budget rejection, worker crashes, and worker timeouts into Postgres errors.
- [x] Respect query cancellation and statement timeouts during CPU runs. Stop the worker and discard partial results.
- [x] Create centroid tables and indexes in one transaction. Verify rollback leaves no partial results after failure.
- [x] Keep CPU and later GPU options and centroid-table formats compatible. Numerical results need not be byte-identical.
- [x] Add executable SQL regressions through `scripts/check_extension.sh` and concurrent snapshot checks through `scripts/check_snapshot.py`. The current harness uses psql assertions.
- [x] Verify with `EXPLAIN` that queries use the resulting index.
- [x] Test the complete CPU flow for flat and two-level trees: source vectors, centroids, index build, and nearest-neighbor queries.
- [x] Test `skip_index_build` and a source column whose name is not `embedding`.
- [x] Document query probe settings matching the centroid tree depth. The image checks use `4` for flat and `2,4` for two-level trees.

## Milestone 4: end-to-end CPU tests and correctness validation

Establish correctness before changing arithmetic or adding GPU execution.
M1 through M3 provide reference checks and small integration smoke tests. Extend that evidence to representative inputs and final-index quality.
Collect timings for diagnosis and repeatability. The cuVS performance comparison belongs to M7.

- [x] Run small CPU extension checks for centroid output, flat/two-level indexes, actual index scans, snapshot consistency, and failure handling. See [M3 evidence](evidence/m3_cpu_2026-10-26.txt).
- [ ] Extend end-to-end coverage across L2, inner product, cosine, spherical centroids, seeds, sampling factors, restarts, and batch sizes.
- [ ] Cover representative dimensions, including 128, 768, 2000, and 3072, with duplicate vectors, empty clusters, and zero vectors.
- [ ] Compare clustering objectives and assignments against independent reference calculations on controlled datasets.
- [ ] Calculate exact nearest neighbors and measure final-index recall at fixed query probes and settings.
- [ ] Define correctness tolerances and a minimum recall target before evaluating optimized kernels.
- [ ] Validate larger source tables and bounded sampling through SQL, including memory-budget rejection and concurrent writes.
- [ ] Record datasets, seeds, versions, settings, and raw results. Distinguish native hardware from emulated development environments.
- [ ] Automate CPU library, worker, and end-to-end Postgres checks in CI without GPU or CUDA dependencies.

Exit criterion: the complete CPU flow passes the defined correctness and recall checks with reproducible evidence.

## Milestone 5: SIMD and Vx compiler/standard-library enhancements

Improve CPU computation while improving Vx's SIMD facilities alongside the application.
The current worker build omits `-O3`, and Vx defaults to `-O0`.
The inspected Linux object uses scalar arithmetic in the distance and centroid loops.
That result does not establish what an optimized build can generate.

- [ ] Build the existing kernels with `-O3`. Inspect generated instructions and measure against the current scalar baseline before rewriting kernels.
- [ ] Profile CPU clustering to identify the loops that justify SIMD work.
- [ ] Separate L2, inner-product, and cosine kernels so each computes only the required quantities.
- [ ] Evaluate automatic vectorization, native Vx vector values, and inline `mlir!` for the hot loops. Record limitations with small compiler examples.
- [ ] Enhance Vx vector loads/stores for existing float buffers with explicit alignment, bounds, and aliasing contracts.
- [ ] Improve `std::simd` with useful load/store, arithmetic, reduction, and scalar-tail operations. Support appropriate lane widths and target capabilities.
- [ ] Audit the current Rust-backed SIMD helpers for actual packed instructions, FFI overhead, pointer aliasing, and documented fused-operation behavior.
- [ ] Prefer vector operations that remain in registers across the kernel. Avoid a separate FFI call for every small arithmetic operation.
- [ ] Implement vectorized distance, centroid accumulation, and normalization where profiling supports them. Handle dimensions that are not lane multiples.
- [ ] Define floating-point reduction order, reassociation, and FMA behavior explicitly. Keep a scalar reference and avoid silently enabling broad fast-math assumptions.
- [ ] Add Vx compiler and standard-library checks for generated SIMD instructions, scalar tails, unaligned buffers, numerical behavior, and supported CPU targets.
- [ ] Rerun M4's correctness and recall checks. Measure kernel and total-build improvements without claiming cuVS competitiveness yet.
- [ ] Record compiler and library changes with their application examples, validation, and remaining gaps.

Exit criterion: verified SIMD execution and measured CPU improvement, while retaining the M4 correctness and recall targets.

## Milestone 6: GPU enablement and Vx enhancements

Enable GPU execution after CPU correctness and SIMD validation.
Develop the required compiler, runtime, and library support alongside the Vextor kernels.
GPU availability does not block the preceding CPU stages.

- [ ] Fix useful parallel execution for the GPU assignment and reduction loops. Record generated launch metadata and native execution evidence.
- [ ] Resolve tiled indexing, placed-buffer views, and pointer-offset gaps with focused Vx examples and compiler tests.
- [ ] Add or improve device reductions and atomics where the chosen kernels require them.
- [ ] Link and call the Vx library on a Linux CUDA host. Validate execution on an NVIDIA target such as A100 or H100.
- [ ] Add GPU initialization, sampling, restarts, and host-to-device transfers with the CPU path's options and centroid-table format.
- [ ] Load GPU support optionally. Preserve CPU execution when GPU hardware or required GPU libraries are unavailable.
- [ ] Select a supported GPU before execution. Otherwise select CPU and emit the fallback warning.
- [ ] Report failures after GPU execution starts. Do not silently restart failed GPU work on CPU.
- [ ] Test device-memory limits, GPU memory exhaustion, worker crashes, cancellation, and transaction rollback.
- [ ] Validate GPU centroids and final-index recall against the CPU reference and M4 targets. Do not require byte-identical floating-point results.
- [ ] Run the complete GPU SQL-to-index-to-query flow and retain raw evidence with driver, CUDA, GPU, and Vx versions.
- [ ] Add GPU checks on an appropriate runner. Record the compiler and runtime gaps exposed by those checks.
- [ ] Measure transfer and execution stages to guide pinned-memory, double-buffering, and other data-path improvements.

Exit criterion: working and validated GPU execution, with the CPU fallback still working independently of CUDA.

## Milestone 7: performance comparison against cuVS

Compare performance after GPU enablement passes the correctness checks.
Use the same datasets, hardware, clustering parameters, and query settings for Vx and cuVS.
Use [vsbt](https://github.com/EnterpriseDB/vsbt) or a compatible harness where practical.

- [ ] Prepare pgpu/cuVS on the validated GPU host. Capture centroid rows, index definitions, query plans, and reference versions.
- [ ] Add a Vx benchmark configuration with comparable sampling, iterations, restarts, metrics, and spherical-centroid settings.
- [ ] Compare Vx GPU clustering against cuVS through pgpu or a controlled direct cuVS harness.
- [ ] Separate clustering time from table reads, transfers, centroid writes, and index creation using `skip_index_build` and stage timings.
- [ ] Compare final-index recall at fixed settings and performance at comparable recall. Retain the CPU implementation as a baseline.
- [ ] Include VectorChord's internal CPU build and optionally faiss-gpu as additional baselines.
- [ ] Cover datasets of different shapes: 1M x 128 (SIFT), 1M x 960 (GIST), and 5M x 768 (LAION, dot product), then larger.
- [ ] Run repeated measurements on native hardware. Record memory use, variation, warm-up policy, versions, settings, and raw output.
- [ ] Publish results, including regressions and cases where Vx loses. Turn measured bottlenecks into the next Vx compiler and library tasks.

Exit criterion: reproducible performance and recall comparisons against cuVS, with evidence supporting each claim.

## Additional input types — follow-up work

- [ ] Add `halfvec` conversion to float32. Verify centroid output and VectorChord index compatibility.
- [ ] Evaluate bounded `sparsevec` storage, distance behavior, and VectorChord compatibility before adding SQL support.
- [ ] Evaluate `bit` separately. Define Hamming or Jaccard clustering and verify VectorChord support before promising a binary index path.
- [ ] Test each added input type through the SQL function, centroid table, index build, and nearest-neighbor query.

## Vx language and library gaps to propose

Compiler and library enhancements are part of the main implementation sequence, especially M5 and M6.
For each gap, keep an application example and an upstream regression check alongside the proposed improvement.

Vx's core work is inference-focused, so database-style needs are the contribution. Keep a running list here as the milestones expose gaps, and turn each into a short upstream proposal early so library work and language direction stay aligned.

- [ ] Atomics and reductions on device (counters, histograms, argmin/argmax across blocks).
- [ ] Irregular and variable-length data: ragged arrays, offsets-plus-values layouts, strings.
- [ ] Sorting, top-k and hash-table primitives as standard-library modules.
- [ ] A columnar / Arrow-style buffer module alongside the existing `tensor`, `simd` and `vec` modules.
- [ ] A stable C ABI export story, including versioning and error-return conventions.
- [ ] Streaming and async transfer primitives (pinned memory, double buffering, events).
- [ ] Seeded random number generation that gives reproducible results across backends.
- [ ] Memory limits and out-of-memory behavior as a recoverable error rather than an abort.
- [ ] Use the compiler's transfer-cost reasoning (it finds the cheapest legal route across declared bus edges) for a planner-style cost model of host-to-device movement.
- [ ] For each gap: write the proposal, link the failing example, and note whether the core developer agrees.

## Backends and portability

Start with CPU support and a complete CPU index build. Add NVIDIA acceleration later, then evaluate other GPU backends.
Backend selection must preserve CPU execution without requiring CUDA libraries.

- [ ] Ship and validate the CPU integration first. GPU support remains deferred until Milestone 6.
- [ ] Keep the k-means code free of backend-specific calls so a new backend needs no algorithm changes.
- [ ] Define a small backend test matrix: which operations (matmul, reductions, atomics, sorting) the algorithm needs, and check each against every new backend as it lands.
- [x] Connect the existing CPU library to the worker and extension. Keep it available as the fallback when GPU support arrives.
- [x] When the extension selects the CPU fallback because no supported GPU is available, emit a PostgreSQL WARNING that GPU acceleration is unavailable and CPU k-means is running. Loading the extension alone should not require a GPU. Test the warning and successful CPU index build on a host without a GPU.
- [ ] Run M4 correctness checks and M7 performance comparisons on each added backend. Compare centroid quality and recall alongside speed.
- [ ] Define optional GPU loading around Vx's build-time backend selection. Keep the CPU library independently loadable without GPU dependencies.
- [ ] Document support honestly per backend (experimental, supported, unsupported).

## Production hardening

Keep worker isolation, cancellation, and transaction rollback in the first CPU integration.
The remaining packaging and operational work follows the first working demo.

- [ ] Harden the separate worker selected in Milestone 0. Keep CPU and later GPU failures outside the Postgres backend.
- [ ] Version the Vx ABI and the SQL function signature, and test upgrades with `ALTER EXTENSION ... UPDATE`.
- [ ] Package for common platforms with `cargo pgrx package`, and document the CUDA and driver requirements.
- [ ] Add observability: progress reporting during long builds, timings per stage, GPU memory use, clear log lines for backend selected.
- [ ] Make runs reproducible: seeded sampling and fixed iteration counts, with the seed logged.
- [ ] Preserve transactional failure behavior for CPU and later GPU runs. Failed runs must leave no partial centroid table or index.
- [ ] Test on managed or restricted setups where GPUs and custom extensions may not be available, and document the limits.
- [ ] Run CPU unit and end-to-end Postgres tests in CI without GPU hardware or CUDA libraries.
- [ ] Add GPU tests on a labeled runner in Milestone 6, plus a regression benchmark on a fixed small dataset.

## Licensing and contribution hygiene

pgpu is AGPL-3.0, and VectorChord is dual-licensed under AGPLv3 or Elastic License v2. These terms affect how code can be reused and how a Vx-based extension could be distributed. This is a checklist of questions to settle, not legal advice.

- [ ] Read the licenses of pgpu, VectorChord, pgvector and cuVS, and record what each allows.
- [ ] Decide the approach to pgpu: write k-means against the documented table contract and VectorChord's public docs, without copying pgpu's code.
- [ ] Choose a license for our extension and check it is compatible with how it will be used and distributed.
- [ ] Get a qualified legal opinion before any commercial or hosted use.
- [ ] Agree with Vx's core developer where this work lives (inside the Vx repo, or a separate repo that depends on it), who reviews it, and how proposals are handled.
- [ ] Check your employer's policy on open-source contributions, since you work in this space professionally.
- [ ] Keep a short contributing note: how to build, run tests and add a backend.

## Open questions and risks

The immediate risks are bounded input memory, worker failure handling, and VectorChord accepting the generated centroid tree.
GPU performance against cuVS is a later question and does not block CPU integration.

- [x] Vx exports the C ABI used by the standalone CPU library. Linux CUDA validation remains in Milestone 6.
- [ ] Compare the validated CPU centroid-table contract with pgpu output in Milestone 7. Live CPU flat/two-level builds already passed in M3.
- [ ] In Milestone 7, profile pgpu/cuVS clustering and database reads and writes to explain the performance comparison.
- [ ] In Milestone 7, validate performance claims against cuVS at comparable recall. Make no GPU speed claim before measurement.
- [ ] How stable is Vx's language and standard library while this is built? Agree how breaking changes are communicated.
- [ ] Does the pgpu README's flow (VectorChord still builds the index from our centroids) match what EDB's docs describe as the GPU running the entire build? Settle this from the source or a test.
- [ ] Which new Vx backends land first, and are the operations this algorithm needs available on them?

## Sources

- [pgpu README](https://github.com/EnterpriseDB/pgpu) and [Cargo.toml](https://github.com/EnterpriseDB/pgpu/blob/main/Cargo.toml)
- [EDB PGPU documentation](https://www.enterprisedb.com/docs/edb-postgres-ai/1.3/ai-factory/pgpu)
- [VectorChord overview](https://docs.vectorchord.ai/vectorchord/getting-started/overview.html) and [external build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)
- [EDB vsbt benchmark suite](https://github.com/EnterpriseDB/vsbt)
- [Vx repository](https://github.com/vx-lang/Vx) and [project site](https://vxlang.org/)
- [VectorChord GPU sidecar RFC](https://github.com/supervc-stack/VectorChord/discussions/476)
