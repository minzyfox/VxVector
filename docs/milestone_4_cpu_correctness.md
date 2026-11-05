# Milestone 4: CPU correctness baseline

November 4, 2026. Synthetic and real-data CPU checks pass. The CI workflow and
targeted verification examples are implemented. M4 remains open for full kernel
proofs and a hosted CI result. The internal clustering-objective comparison now passes; see [results and limits](internal_objectives.md).
SIMD, GPU, and performance comparisons remain later milestones.

## Run the checks

Build the standalone library using the existing Vx checkout:

```sh
./scripts/build.sh
python3 -m venv build/correctness-venv
build/correctness-venv/bin/python -m pip install -r tests/requirements-correctness.txt
build/correctness-venv/bin/python -B scripts/check_cpu_correctness.py
VEXTOR_WORKER="$PWD/build/vextor-worker" \
  build/correctness-venv/bin/python -B -m unittest discover -s tests -v
./scripts/check_data_path.sh
```

The four worker tests require an existing native worker. To build it on macOS:

```sh
clang -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror \
  -I include src/worker.c -L build -lvextor_kmeans \
  -Wl,-rpath,"$PWD/build" -o build/vextor-worker
```

Start the existing PostgreSQL image using the README instructions. Then run:

```sh
export DOCKER_CONTEXT=colima
python3 -B scripts/check_recall.py
./scripts/check_extension.sh
python3 -B scripts/check_snapshot.py
```

`check_recall.py` accepts `--container` and `--epsilon`. It uses a unique schema
inside one repeatable-read transaction. Success rolls back every fixture.
On failure, psql exits and PostgreSQL rolls back the disconnected transaction.
The existing snapshot check uses its own schema and intentionally commits test
inserts and updates before removing that schema.

## November 4 extension

The corrected CPU library passes 25 native tests, 48 matched Faiss cases,
96 independent float64 Lloyd cases, and six zero/duplicate cases.
The independent cases vary iteration counts, restarts, sampling factors, and
spherical behavior separately. They include non-spherical cosine on inputs with
unequal norms. Dimensions 13 and 128 exercise non-aligned and aligned shapes.

Numerical tests exposed a defect: large finite vectors overflowed float32 norms
and became zero spherical centroids. Distance and normalization intermediates
now use float64. Centroid buffers and accumulation storage remain float32.
The function returns an error when a centroid sum overflows. SQL checks verify
unit norms for large inputs and rollback after a numerical error.
The old installed worker failed the new unit-norm SQL assertion.
Three isolated native mutations failed the intended norm, assignment, and error
status assertions. This changes the scalar baseline for later SIMD work.

The complete corrected run passed the synthetic SQL suite, image checks,
permissions, worker failures, cancellation, rollback, and concurrent snapshot checks.
It used a separate Linux amd64 database on port 15433 under x86 emulation.
The original development database on port 15432 kept its previous image.

### Real data and internal training comparison

[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) supplies GloVe-25
word embeddings and SIFT-128 image descriptors. Each fixture uses 8,192 distinct
training rows selected with a fixed NumPy seed and 32 separate test queries.
GloVe inputs are unit-normalized and cover cosine and inner product.
SIFT covers L2. The recorded hashes identify the files and exact selected arrays.
These subsets establish a bounded baseline, not full-dataset or production validation.

Each metric uses 32 leaves, flat and four-root trees, and Vx seeds 0 and 17.
Both trainers use sampling factor 40, ten iterations, spherical behavior where
applicable, and one training attempt. Internal training uses one thread and Lloyd.
Residual quantization is disabled for both. Every build receives the same source
snapshot and held-out queries. Exact results use sequential scans.
Only one approximate index exists during each query. Its plan must select that index.

VectorChord 1.1.1 fixes its clustering initializer to `[7; 32]` internally.
SQL does not expose that seed, initial centroids, or trained centroid rows.
Its source sampler differs from Vextor's sampler. Internal build records use the
`seed` field only to identify the paired Vx run. It does not control internal training.
The paired internal builds are independent repeats, not matched-initialization tests.
Empty-cluster repair and floating-point reduction order also differ.
Strict centroid equivalence is not required for these unmatched starts. An isolated
instrumented 1.1.1 source build now exports unrotated centroids for independent
common-source leaf-objective scoring. All six groups pass a predefined 10% relative
cost gate; the largest mean increase is 0.43%. This is a bounded quality comparison,
not matched-initialization equivalence. See [objective procedure and evidence](internal_objectives.md).

The harness defines gates before running: conservative recall@10 must reach 0.95
for every query. At each practical setting, Vx mean recall may trail internal
training by at most 0.05 across the two repeats and 32 queries.
This is a fixed regression gate, not a statistical equivalence claim.

All 24 builds and 3,072 queries passed. All-probe, epsilon-4 recall reached 1.0.
At epsilon 1.9 with 16 leaf probes, Vx mean recall reached 0.9984375 for flat
GloVe and 0.99375 for two-level GloVe. SIFT reached 1.0 at those settings.
The largest mean recall deficit against internal training was 0.0109375.
The lowest-probe two-level GloVe setting reached only 0.9328125 mean Vx recall.
Therefore that setting does not satisfy a 0.95 mean target on this fixture.
The earlier synthetic default-epsilon failure remains valid evidence.
No universal default-epsilon or low-probe recall guarantee follows from these runs.

Build timings and query timings remain diagnostic because the database is emulated.
Wrong index plans, duplicate IDs, and farthest-neighbor mutations failed the intended
assertions in separate rolled-back transactions.

### Automation and verification examples

[CPU CI](../.github/workflows/cpu-correctness.yml) builds the pinned Linux Vx
compiler, native worker, and PostgreSQL image without CUDA. It downloads datasets,
checks their hashes, runs the CPU pipeline, and preserves logs.
The local commands passed. Workflow YAML and shell syntax checks passed.
A hosted GitHub Actions run is not recorded yet.

Three helper properties pass on both tested compilers: choosing a bounded centroid
index, computing a bounded offset for dimension 128, and retaining a positive count.
Four mutations fail at the intended contract diagnostics.
Those two examples failed on the original compiler pin. The November 5 proof branch
now accepts them and integrates the nearest result and centroid-offset contracts
into the kernel. See [the proof update](m4_index_proofs_2026-11-05.md).
These proofs do not establish foreign buffer capacities or all source-buffer offsets.
Runtime checks remain necessary. Solver timings and diagnostics are recorded.

Reproduce the complete pipeline after building the library and database:

```sh
python3 -m venv build/correctness-venv
build/correctness-venv/bin/python -m pip install -r tests/requirements-correctness.txt
mkdir -p build/datasets
curl --fail --location https://ann-benchmarks.com/glove-25-angular.hdf5 -o build/datasets/glove-25-angular.hdf5
curl --fail --location https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o build/datasets/sift-128-euclidean.hdf5
shasum -a 256 --check tests/datasets.sha256
DOCKER_CONTEXT=colima VEXTOR_CHECK_POSTGRES=1 bash scripts/check_cpu.sh
build/correctness-venv/bin/python -B scripts/check_contracts.py
```

Set `VEXTOR_POSTGRES_CONTAINER` when checking another container.
The real-data fixtures roll back automatically. Downloaded data stays under ignored `build/`.

Evidence:

- [Complete corrected pipeline](evidence/m4_full_2026-11-04.txt)
- [Binary and source hashes, mutation results](evidence/m4_metadata_2026-11-04.json)
- [macOS verifier checks](evidence/m4_contracts_2026-11-04.jsonl)
- [Linux verifier checks](evidence/m4_contracts_linux_2026-11-04.jsonl)
- [Old worker failing the new SQL assertion](evidence/m4_old_worker_regression_2026-11-04.txt)
- [Real-data harness mutations](evidence/m4_real_mutations_2026-11-04.txt)

## Matched Faiss comparisons

The standalone suite compares the native library with Faiss CPU 1.15.1.
It uses 48 combinations of dimensions 128, 768, 2000, and 3072; L2, inner
product, and cosine; seeds 0 and 17; and flat or two-level trees.
The two option bundles exercise partial/full sampling, one/eight iterations,
one/three restarts, and batches of one or thirteen rows. Each case also compares
its result exactly with a batch of 1,000 rows.

The harness reconstructs the documented implementation's sample order and
initial centroid positions. Both implementations receive the same float32 sample
and initial centroid bytes. An equal random seed alone would not establish that.
Faiss subsampling is disabled. Each Faiss call runs one Lloyd iteration, repeated
for Vextor's requested count. This prevents Faiss early stopping from changing
the schedule. Restarts run separately and use independently recomputed sample
objectives to choose the best result.

Centroids are matched up to permutation. Coordinates and float64 sample
objectives use absolute and relative tolerances of `3e-4`. Assignment checks
accept any centroid tied within those tolerances. The harness checks finite
centroids, assignment bounds and distances, the zero root, parent ranges,
nearest-root choices, node counts, and unit norms for nonzero spherical nodes.
Root training uses the Vx leaves as input, preserving the leaves-first hierarchy.
Thus it checks each training stage separately rather than allowing small leaf
differences to change the root-training problem.

Inner-product and cosine reference cases use spherical centroids. Cosine uses
unit input vectors; its assignment ordering then agrees with Faiss inner-product
search. This does not validate arbitrary non-spherical cosine against Faiss.
Existing native tests also check directional and zero-vector cosine cases.

Faiss splits populated clusters to replace empty clusters. Vextor substitutes a
sampled row. The matched fixtures must report zero Faiss splits; the suite fails
if that policy difference enters a comparison. Six separate zero/duplicate cases
check finite, normalized, known-answer outputs and valid trees when empty clusters
occur. Those cases do not claim equivalence with Faiss splitting.

The suite records library and sample hashes, versions, settings, and objective
values as JSON lines. Dataset generation and initialization stay in the script.
These fixtures and recorded scalar results form a baseline for later SIMD work.

## SQL recall and bounded sampling

The SQL suite builds 24 indexes: four dimensions, three metrics, and both tree
shapes. It uses deterministic 256-row directional fixtures and five held-out
queries per index. The target is tie-aware recall@10 of at least `0.95` for every
query. Exact ground truth uses a checked sequential-scan plan. Approximate
queries must name the intended Vextor-built index in their plan.

Exact queries sort by distance and ID. An approximate result must contain ten
distinct IDs. All exact neighbors strictly below the cutoff are required; tied
cutoff neighbors can substitute for one another. The boundary tolerance is
`1e-6 + 1e-5 * abs(cutoff)`. Each record includes both sets of IDs and the cutoff.

Every leaf is probed (`4` or `2,4`). Epsilon defaults explicitly to `4.0`, the
installed VectorChord maximum. Scan limits are disabled. These are conservative
correctness settings; they do not establish useful speed or low-probe recall.
All 120 queries returned recall 1.0 in the recorded run.

The initial experiment used VectorChord's default epsilon of 1.9. The L2,
768-dimensional flat case returned recall 0.90 for query 2. The target stayed at
0.95. The failing output is retained separately. Reproduce it with:

```sh
python3 -B scripts/check_recall.py --epsilon 1.9
```

The final SQL check grows a source from 256 to 8,192 rows with a fixed 32-row
sample. Both calls reported 103,864 buffer bytes. A one-byte budget failed and
left no centroid table. This establishes the reported explicit-buffer bound,
not a bound on total PostgreSQL or worker memory.

## Recorded validation

- Rebuilt macOS arm64 standalone library: 48 matched Faiss cases and six
  zero/duplicate cases passed.
- Existing native library, adapter, and worker tests: 22 passed without skips.
- Existing allocation-failure and address/undefined-behavior sanitizer checks passed.
- PostgreSQL 17.11, pgvector 0.8.2, VectorChord 1.1.1, pg_vextor 0.1.0:
  24 index builds, 120 recall queries, and larger-source budget checks passed.
- Existing SQL failure, cancellation, permissions, and rollback checks passed.
- Concurrent snapshot check passed.

The database ran Linux amd64 under x86 emulation on an arm64 Mac. The installed
Linux worker used Vx source revision `41e73e48191069ea4a026fbab473126759e5587c`.
The standalone library was rebuilt with the current local compiler. Its binary
hash is recorded separately; `vxc --version` reported only `0.0.3`, so its exact
source revision is not established by that version string. These are separate
platform checks, not a claim of identical compiler builds. No timings here are
performance evidence.

Raw evidence:

- [Faiss comparisons](evidence/m4_cpu_2026-11-03.jsonl)
- [Final SQL run](evidence/m4_recall_2026-11-03.txt)
- [Default-epsilon failure](evidence/m4_default_epsilon_2026-11-03.txt)
- [Regression and mutation results](evidence/m4_checks_2026-11-03.txt)

## Mutation checks

Four temporary Vx source variants compiled into separate libraries. Biasing
centroid division failed the centroid comparison. Changing final assignments
failed the nearest-leaf assertion. Writing leaf parent zero failed the parent
range assertion. Disabling normalization failed the unit-norm assertion.
Each variant failed at the intended assertion; repository source stayed intact.

Four temporary SQL harness variants also failed at the intended assertions:
using an index for ground truth, using a sequential scan for index results,
returning duplicate IDs, and returning the farthest neighbors.
An initial planner mutation only disabled index scans while sequential scans
were also disabled. PostgreSQL still selected the index. That attempt did not
prove the plan assertion; enabling sequential scans made the corrected mutation
fail at the intended assertion.

## Remaining M4 work

The November 4 compiler upgrade used Vx `7cc3d2d9` with borrowed metric enums,
pointer contracts, and recorded build hashes. Its
[installation evidence](evidence/vx_main_install_2026-11-04.json) remains historical.
On November 5, native and Linux builds moved to `ded61615`, and the local database
received the new worker image. Its data volume and cluster identity stayed unchanged.
The new-pin native, contract, code-generation, PostgreSQL integration, snapshot, and
synthetic recall checks pass. Real-data and internal-objective checks were not rerun.
See [the current compiler audit](vx_main_update_2026-11-05.md) and
[new validation summary](evidence/vx_main_test_summary_2026-11-05.json).

The complete November 4 CPU/database pipeline and both platforms' contract/code-generation
checks passed with that compiler and worker.
See [the new test summary](evidence/vx_main_test_summary_2026-11-04.json) and
[the full upgraded-worker log](evidence/vx_main_checks_2026-11-04.txt).

The internal-trainer objective comparison is complete for the bounded real-data
fixtures: 24 centroid captures, six quality groups, and 3,072 recall queries passed.
The reference uses an instrumented source build with a distinct build profile;
see [the manifest, measurements, and limits](internal_objectives.md).

- Complete the remaining kernel contracts. The nearest-index loop and widened
  centroid-offset arithmetic now pass on the proof branch. Source sampling, all
  source-buffer offsets, and the positive division-count proof remain open.
  Foreign allocations and lifetimes remain C caller obligations.
- Record a successful hosted CI run. The workflow and its component commands are
  implemented and locally validated. No remote workflow was triggered.

The optional scikit-learn cross-check remains optional. Higher-dimensional real
embeddings and larger samples would broaden confidence beyond this bounded baseline.

## Reference contracts

- [Faiss clustering API](https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h)
  describes supplied centroids, iteration settings, and subsampling limits.
- [VectorChord settings](https://github.com/tensorchord/VectorChord/blob/main/src/index/gucs.rs)
  defines epsilon's default and range. The installed database settings also
  reported default 1.9 and range 0 to 4.
