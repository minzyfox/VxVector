# Vx × Postgres: GPU k-means for VectorChord — TODOs

Oct 20, 2026 · @desia

## Goal and scope

Build a Vx implementation of the GPU k-means step that VectorChord's external build consumes, wrapped in a pgpu-style Postgres function, and use the work to grow Vx's libraries and language.

The reference flow is pgpu's: read vectors, run k-means on the GPU, write a centroid table, then call VectorChord's index build with `build.external` ([pgpu](https://github.com/EnterpriseDB/pgpu), [VectorChord external build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)).

- In scope: batched, sampled k-means (flat and two-level), distance operators ip / l2 / cos, spherical centroids, centroid table output, VectorChord handoff.
- Out of scope for now: query-time GPU work, replacing VectorChord's index, other index types.
- Success: a VectorChord index built from Vx centroids reaches recall comparable to a pgpu/cuVS-built index on the same dataset (threshold TBD), with build time measured and a written gap list for Vx.

## Milestone 0: groundwork and feasibility

Settle the questions that could block everything else before writing any k-means code.

Feasibility decisions and the deferred GPU reference run are recorded in
[Milestone 0 findings](milestone_0_feasibility.md).
The GPU run does not block standalone implementation. Assume the target GPU is
available, then verify the behavior on real hardware when a host is available.

- [x] Confirm whether Vx can export a C ABI (a shared library with `extern "C"` entry points). Everything in Milestone 3 depends on this. Verified with a macOS arm64 C caller; Linux CUDA verification remains.
- [x] Decide the integration boundary. The C ABI works, and the GPU worker will run in a separate process because the current Vx runtime can abort.
- [x] Define the error boundary: Vx code must never unwind or longjmp into a Postgres backend. Write down how failures cross the ABI (return codes, error structs).
- [x] Check how Vx selects its dispatch backend. The README says `build.rs` picks it from what the build machine has, so a portable build made without CUDA ships only the CPU backend. Decide how an extension package should handle this.
- [ ] When a CUDA host is available, stand up a dev environment: Postgres 17 or 18, pgvector, VectorChord, a CUDA-capable GPU, and the Vx toolchain. This does not block Milestone 1.
- [ ] When a CUDA host is available, run pgpu once end to end on a small table to capture reference output (centroid table, index, query plan with `EXPLAIN`). This does not block Milestone 1.
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
- [ ] Run and optimize the GPU path on NVIDIA hardware. The current GPU prototype compiles to A100 code but emits no parallel launch metadata.

## Milestone 2: data path

The flow reads a table, clusters, then writes centroids back, so moving vectors to the GPU efficiently is likely where a database background adds the most. This is my inference from the pipeline's shape, not something pgpu's README measures, so the first task is to measure it.

The host-side batch and sampling design is in [Milestone 2 data path](milestone_2_data_path.md).

- [ ] Profile pgpu's pipeline to see where time goes: table read, host-to-device transfer, clustering, centroid write, index build. Decide what to optimize from data.
- [x] Design the host-side batch format: contiguous row-major float32 buffers sized by `batch_size`. Pinned host memory needs GPU validation.
- [ ] Implement streaming: read the next batch while the GPU processes the current one (double buffering) so the GPU is not idle during reads.
- [ ] Handle sampling at read time: pick `lists * sampling_factor` rows without reading the whole table into memory.
- [ ] Support pgvector `vector` (float32) input first and validate its dimensions and distance operator.
- [ ] Add `halfvec` input: convert float16 components to float32 for clustering, then verify centroid output and VectorChord index compatibility.
- [ ] Evaluate `sparsevec` input with bounded storage. Verify distance behavior and VectorChord compatibility before adding it to the SQL API.
- [ ] Evaluate `bit` input separately. Define clustering for Hamming or Jaccard distance and verify VectorChord support before promising a binary index path.
- [ ] Test each supported input type through the SQL function, centroid table, index build, and nearest-neighbor query.
- [ ] Use Vx's placement types to make host/device pointer mistakes a compile error, and record where the types helped or got in the way.
- [ ] Define memory limits: what happens when the table is larger than device memory (batching) or host memory (streaming).

## Milestone 3: Postgres integration

Wrap the Vx k-means in a pgrx extension that produces the same centroid table pgpu does and hands it to VectorChord.

- [ ] Create a pgrx extension crate (pgrx supports Postgres 13 through 18; pgpu defaults to 17 and lists 14 to 18). Pick the supported version range.
- [ ] Link or load the Vx library through the C ABI (or the fallback chosen in Milestone 0).
- [ ] Implement a SQL function modeled on `pgpu.create_vector_index_on_gpu(...)`: table, column, `lists`, `sampling_factor`, `batch_size`, `kmeans_iterations`, `kmeans_nredo`, `distance_operator`, `skip_index_build`, `spherical_centroids`, `residual_quantization`.
- [ ] Read vectors from the table in batches and feed the Milestone 2 data path.
- [ ] Write the centroid table in the format VectorChord's external build expects (confirm against the VectorChord docs and pgpu's output).
- [ ] Issue `CREATE INDEX ... USING vchordrq` with `build.external` pointing at the centroid table; support `skip_index_build` to stop after the centroids.
- [ ] Convert Vx failures into Postgres errors through the pgrx error path. Test: out of device memory, bad arguments, and cancelled queries. A missing GPU selects the CPU path with the warning specified below.
- [ ] Respect query cancellation and statement timeouts during long GPU runs.
- [ ] Write `pg_regress` tests (pgpu has a `tests/pg_regress` directory to model on).
- [ ] Verify with `EXPLAIN` that queries use the resulting index.

## Milestone 4: benchmarks and validation

Measure Vx against real baselines so the result is credible either way. EDB's [vsbt](https://github.com/EnterpriseDB/vsbt) benchmark suite already runs pgvector, VectorChord and pgpu on datasets from 1M to 1B vectors, so reuse it instead of building a harness.

- [ ] Run vsbt's pgpu configuration on a small dataset to get a baseline on our hardware.
- [ ] Add a Vx configuration to vsbt (or a compatible one) so the same datasets and metrics apply.
- [ ] Compare against: cuVS k-means (via pgpu), faiss-gpu (VectorChord's older toolkit), and VectorChord's internal CPU build.
- [ ] Separate clustering time from total build time using `skip_index_build`, so the k-means step is measured on its own.
- [ ] Measure recall at fixed settings. Centroid quality drives recall, so recall on the final index is the real test, not centroid distance alone.
- [ ] Cover datasets of different shapes: 1M x 128 (SIFT), 1M x 960 (GIST), 5M x 768 (LAION, dot product), then larger.
- [ ] Record hardware, driver, CUDA and Vx versions with every result, and keep raw output in the repo.
- [ ] Publish results including where Vx loses, since the gaps feed the language wish list.

## Vx language and library gaps to propose

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

Start on NVIDIA, since pgpu and cuVS are NVIDIA-only today and give a direct baseline. Then use the new Vx backends to turn portability into a measured result. Vx's repo lists machine files for H100, A100, B200, MI300X and Apple M4, and new backends are being added.

- [ ] Ship the first working version on one NVIDIA target (A100 or H100) and record exact driver and CUDA versions (pgpu's LTS docs require driver 580 or newer; its build uses CUDA 13.1).
- [ ] Keep the k-means code free of backend-specific calls so a new backend needs no algorithm changes.
- [ ] Define a small backend test matrix: which operations (matmul, reductions, atomics, sorting) the algorithm needs, and check each against every new backend as it lands.
- [ ] Add a CPU fallback path with the same output, so the extension works without a GPU (slower) and tests run in CI without one.
- [ ] When the extension selects the CPU fallback because no supported GPU is available, emit a PostgreSQL WARNING that GPU acceleration is unavailable and CPU k-means is running. Loading the extension alone should not require a GPU. Test the warning and successful CPU index build on a host without a GPU.
- [ ] Re-run the Milestone 4 benchmarks on each new backend (AMD, Apple) and compare centroid quality and recall, not only speed.
- [ ] Decide the runtime policy: build one binary that detects the backend at run time, or one build per backend. Account for the build-time backend selection noted in Milestone 0.
- [ ] Document support honestly per backend (experimental, supported, unsupported).

## Production hardening

Nothing here blocks the first demo, but each item decides whether anyone could trust this inside a database host. Index builds are batch jobs, so a sidecar design is a reasonable lower-risk option to evaluate.

- [ ] Evaluate in-process (library inside the backend) versus sidecar (separate process that talks to Postgres). A sidecar contains GPU or kernel crashes outside the Postgres backend; the VectorChord GPU RFC proposes a similar optional sidecar.
- [ ] Version the Vx ABI and the SQL function signature, and test upgrades with `ALTER EXTENSION ... UPDATE`.
- [ ] Package for common platforms with `cargo pgrx package`, and document the CUDA and driver requirements.
- [ ] Add observability: progress reporting during long builds, timings per stage, GPU memory use, clear log lines for backend selected.
- [ ] Make runs reproducible: seeded sampling and fixed iteration counts, with the seed logged.
- [ ] Define failure behavior: a failed GPU run must leave no half-written centroid table or partial index.
- [ ] Test on managed or restricted setups where GPUs and custom extensions may not be available, and document the limits.
- [ ] Set up CI: unit tests without a GPU, GPU tests on a labeled runner, and a regression benchmark on a fixed small dataset.

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

The biggest risk is that Vx cannot yet match a tuned library like cuVS, so the first benchmark may show a gap. That is still a useful result, because it tells us which language or runtime features to work on.

- [ ] Can Vx export a C ABI today? If not, who builds it, and how long would it take?
- [ ] What exact centroid table schema and `CREATE INDEX` options does VectorChord's external build need? Confirm from VectorChord's docs and a pgpu run.
- [ ] Where does pgpu's time go: clustering, or reading and writing through Postgres? The answer decides whether the data path or the kernel is the real work.
- [ ] Is the plan to compete with cuVS on speed, or to offer portability and a clearer codebase? Pick one primary claim.
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
