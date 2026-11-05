# Vx main update: November 4, 2026

## Compiler revision

The previous Linux worker and CI used Vx `41e73e48191069ea4a026fbab473126759e5587c`.
The previous macOS compiler reported only `0.0.3`. Its exact source revision was not established.

The new pin is upstream `vx-lang/Vx` main at
`7cc3d2d95fb61f0b80d3b0bd1d8ff38b833f1c47`, checked on November 4.
The local Vx main checkout moved from `aaeee849` to that commit through a fast-forward.
Untracked local work stayed intact. No branch was pushed.

[vx-revision.txt](../vx-revision.txt) is the shared source for build scripts and CI.
Native builds check the checkout and compiler revision before compiling.
Linux builds use a separate source directory for each revision and reuse Cargo artifacts.
Build metadata records compiler, kernel source, and object hashes.
Packaging rejects a different pin or changed source/object bytes.
The runtime image includes `vx-build-info.json` and `vx-revision.txt` beside the worker.

The Linux compiler builds from a source archive without Git metadata.
Its version string may omit the commit. The packaged build manifest records the exact archive revision and compiler hash.

## Language features used

The internal metric is a `DistanceMetric` enum, borrowed by training and distance functions.
Matching through an enum reference uses the current flat code generator.
The public C ABI still accepts metric codes 0, 1, and 2.

Distance dispatch occurs before the coordinate loop.
L2 computes squared differences. Inner product computes a dot product.
Cosine computes the dot product and both norms.
All distance and norm intermediates remain float64.
Iteration counts, sampling, restarts, empty-cluster policy, and accumulation order stay the same.

The distance and normalization helpers state `valid` and `initialized` pointer requirements.
A shared centroid-copy helper also states `!overlaps`.
Each call relies on the existing C ABI allocation, initialization, alignment, and disjoint-buffer obligations.
These clauses describe trusted unsafe boundaries. They do not prove foreign allocations or the entire kernel.
Copies use separate scratch buffers, including when roots and leaves occupy the same node allocation.

Generated LLVM contains nonnull/alignment information, dynamic dereferenceable assumptions,
and alias scopes for the copy loop's loads and stores.
Both build scripts reject fallback to the legacy AST code generator.
The code-generation harness removes validity and alias clauses in separate mutations.
Both mutated programs compile, then fail their intended metadata checks.

The current `std::simd` module still uses small Rust-backed helpers.
The checked-allocation contract document remains a proposal.
Neither supplies a new verified SIMD or no-allocation guarantee for this kernel.
GPU lane operations, asynchronous transfers, and task wait budgets remain separate follow-up work.

## Validation

The native build passes 25 tests, 48 matched Faiss cases,
96 independent Lloyd cases, six zero/duplicate cases, and allocation/sanitizer checks.
The new library matches the previous M4 library bit-for-bit across 48 cases.
Those cases compare centroids, parents, and assignments across metrics, dimensions, seeds, and tree shapes.
The generated-code checks and their two mutations pass.
The nearest-loop and variable-offset proof examples still fail on current main.

The Linux compiler and kernel-object build completed at `7cc3d2d9`.
Its source and object hashes match the current kernel and recorded build manifest.
The full CPU/database pipeline and Linux/macOS contract/code-generation checks now pass.
The upgraded database passed 120 synthetic and 3,072 real-data recall checks.
See [the test summary](evidence/vx_main_test_summary_2026-11-04.json) and
[the complete log](evidence/vx_main_checks_2026-11-04.txt).
The development database now runs `vextor-postgres:17-vextor` with the new worker.
The container reports healthy. Its data volume and cluster identifier are unchanged.
The previous image is retained as `vextor-postgres:17-before-vx-7cc3d2d9`.
See [the installation record](evidence/vx_main_install_2026-11-04.json).
No performance improvement is claimed before measurement.

## Pending work

- [x] Package the new Linux object and build manifest in the PostgreSQL image.
- [x] Update the local development database to use that image. Verify health, revision, and unchanged data identity.
- [x] Run the deferred Linux/database validation and record the upgraded-worker results.
- [ ] Resolve the nearest-loop and variable-offset proof gaps.
- [ ] Add an internal VectorChord centroid export for objective comparisons.
- [ ] Record a hosted CPU CI run.

## Reproduce

Build the pinned Vx compiler and runtime using the README instructions. Then run:

```sh
./scripts/build.sh
bash scripts/check_cpu.sh
build/correctness-venv/bin/python -B scripts/check_kernel_codegen.py
build/correctness-venv/bin/python -B scripts/check_contracts.py
DOCKER_CONTEXT=colima ./scripts/build_linux_worker.sh
DOCKER_CONTEXT=colima ./scripts/build_extension.sh
```

Read `build/vx-build-info.json` for the native compiler.
Read `build/linux/vx-build-info.json` for the Linux compiler and kernel object.
The earlier M4 evidence remains a historical record of its original compiler builds.
