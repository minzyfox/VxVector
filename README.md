# Vextor

Vextor is a PostgreSQL extension that trains k-means centroids in Vx and uses them
to build a VectorChord vector-search index. The current implementation runs on CPU.
GPU acceleration will come later.

- **pgvector** provides the `vector` data type and distance operators.
- **Vextor** samples source vectors and trains centroids in a separate CPU worker.
- **VectorChord** builds the `vchordrq` index and serves nearest-neighbor queries.

The current extension is `pg_vextor` 0.1.0, validated on PostgreSQL 17 / Linux amd64.
It is intended for development and testing. Larger-data recall, performance,
installation packages, upgrades, and other platforms still need validation.

Start with [Build from source](#build-from-source) for a fresh checkout.
If the images already exist, go to [Start the existing Docker setup](#start-the-existing-docker-setup).
Then follow the SQL walkthrough below to create, index, and query a small table.
[Test instructions](#run-tests) cover database smoke checks and the complete CPU suite.

## Build from source

### Prerequisites

- Git, Docker, and Docker Compose with a running Docker engine.
- Python 3.14 for the host scripts and the pinned correctness-test environment.
- Internet access for source checkouts, container images, and optional test datasets.

The database image is validated on Linux amd64. On an ARM machine, Docker needs
x86 emulation for this setup. No GPU or CUDA installation is required.
The database build compiles Vx and the extension inside Docker.
Host Rust and LLVM are needed only for the native development/test path described below.

### Get Vextor and the pinned Vx source

These commands create new checkouts. Run them from the directory where you want
Vextor to live:

```sh
git clone https://github.com/minzyfox/vextor.git
cd vextor
git clone https://github.com/minzyfox/Vx.git ../Vx
export VX_ROOT="$(cd ../Vx && pwd)"
git -C "$VX_ROOT" fetch origin "$(cat vx-revision.txt)"
git -C "$VX_ROOT" checkout --detach "$(cat vx-revision.txt)"
cp .env.example .env
```

Edit `.env` and set `VEXTOR_POSTGRES_PASSWORD` to a local development password.
The default port is `15432`. Keep an existing `.env` when reusing a database volume.
If you already have both checkouts, set `VX_ROOT` to a checkout at the exact
commit in [vx-revision.txt](vx-revision.txt). Use a separate checkout if your
existing Vx tree contains work you want to keep.

The current pin is `c9612a1f1c8a2819258b159774a615f9aca298a7`.
It adds CPU proof fixes on top of main `ded61615`; see [the proof update](docs/m4_index_proofs_2026-11-05.md).
See [the proof update](docs/m4_index_proofs_2026-11-05.md) for validation and remaining work.
The scripts default to `../../Vx` when `VX_ROOT` is unset. The fresh-checkout
example overrides that default with its adjacent `../Vx` directory.

### Build the images

Run the remaining shell commands from the Vextor repository.
Use the Docker context containing your running engine. For Colima, select it with:

```sh
export DOCKER_CONTEXT=colima
```

Docker Desktop users can keep their active Docker context.
Build the tested amd64 images and start PostgreSQL:

```sh
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build -f "$VX_ROOT/docker/Dockerfile.build" -t vx-build:llvm22 "$VX_ROOT"
./scripts/build_postgres_image.sh
./scripts/build_linux_worker.sh
./scripts/build_extension.sh
docker-compose up -d --no-build --wait
```

Use `docker compose` in place of `docker-compose` if the Compose plugin is installed.
The first build can take many minutes under x86 emulation. Dedicated Docker volumes
retain compiler and Cargo caches for later builds.
Build logs are under `build/linux/`, including `vx-build.log`, `kmeans-build.log`,
and `extension-build.log`.

The complete image is `vextor-postgres:17-vextor`.
Its base development image is `vextor-postgres:17-dev`.
Both native and Linux builds use the Vx pin. Build manifests record compiler,
kernel source, and object hashes. The installed database image carries its manifest
at `/usr/local/libexec/vextor/vx-build-info.json`.

## Start the existing Docker setup

Run these shell commands from the Vextor repository using your Docker context.
On the current development machine, that context is `colima`.
Colima users can select it with `export DOCKER_CONTEXT=colima`.
The first command starts the service. The second opens the SQL prompt:

```sh
docker-compose up -d --no-build --wait
docker-compose exec postgres psql -U vextor -d vextor
```

Use `docker compose` instead of `docker-compose` if the Compose plugin is installed.
The database listens at `127.0.0.1:15432`, with database and user both named `vextor`.
The password is in the ignored `.env` file. This service uses its own data volume.
The existing database on port 5432 is separate.

If the images are unavailable, follow [Build from source](#build-from-source) first.
Run the following SQL sections in the same `psql` session.
Enter SQL at the database prompt, and shell commands in your terminal.

## Create the extensions

A fresh Vextor volume initializes these extensions automatically. These statements
also work when connecting to an existing Vextor database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vchord;
CREATE EXTENSION IF NOT EXISTS pg_vextor;
```

The Docker service preloads `vchord`. A PostgreSQL installation outside this image
also needs the extension files, worker, and preload configuration.
See the [PostgreSQL setup notes](docs/postgres.md).

## Create a table and insert dummy data

This example creates 1,000 three-dimensional vectors in four separated groups.
The data is deterministic, so repeating it gives the same input.
Use these example names in a database where they do not already exist.

```sql
CREATE TABLE public.vextor_demo_items (
    id integer PRIMARY KEY,
    label text NOT NULL,
    features vector(3) NOT NULL
);

INSERT INTO public.vextor_demo_items (id, label, features)
SELECT
    i,
    format('group-%s', i % 4),
    ARRAY[
        (i % 4) * 10 + (i % 17) * 0.01,
        (i % 7) * 0.01,
        (i % 11) * 0.01
    ]::real[]::vector(3)
FROM generate_series(1, 1000) AS i;

SELECT count(*) AS source_rows FROM public.vextor_demo_items;
```

The count should be `1000`. In an application, replace these dummy values with
embeddings from your model. All vectors must have the same dimension and finite
components.

## Train centroids and build an index

```sql
SELECT vextor.create_vector_index(
    'public.vextor_demo_items',
    'features',
    ARRAY[4],
    sampling_factor => 40,
    batch_size => 64,
    kmeans_iterations => 10,
    kmeans_nredo => 3,
    distance_operator => 'l2',
    seed => 42,
    centroid_table => 'vextor_demo_centroids',
    index_name => 'vextor_demo_idx'
);

SELECT id, parent, vector
FROM public.vextor_demo_centroids
ORDER BY id;

ANALYZE public.vextor_demo_items;
```

`ARRAY[4]` requests four leaf centroids. The function samples up to `4 * 40 = 160`
source vectors, trains centroids, then indexes all 1,000 source rows.
The centroid table contains five nodes: a zero-vector root and four leaves.
The root has `parent = NULL`; each leaf has parent `0`.

The function returns JSON with `backend = "cpu"`, output names, row/sample counts,
node count, buffer bytes, and stage timings. It emits this expected warning:

```text
GPU acceleration is unavailable; running CPU k-means
```

Each call creates new outputs. Reusing an existing centroid-table or index name
causes an error. The function does not replace existing objects.
A failed build rolls back the outputs created by that call.

## Query nearest neighbors

Set one probe count for this flat tree. The example searches all four partitions
and uses conservative reranking. These settings favor recall for the demo:

```sql
SET vchordrq.probes = '4';
SET vchordrq.epsilon = '4.0';

SELECT
    id,
    label,
    features,
    features <-> '[10,0,0]'::vector AS distance
FROM public.vextor_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;
```

For L2 distance, smaller values mean closer vectors. This query should find
vectors from `group-1`, whose first coordinate is near 10.

Inspect the query plan:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT id
FROM public.vextor_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;
```

PostgreSQL can prefer a sequential scan for a small table. To check index use
explicitly in this demo, temporarily discourage sequential scans:

```sql
SET enable_seqscan = off;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT id
FROM public.vextor_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;

RESET enable_seqscan;
```

The plan should include an index scan using `vextor_demo_idx`.
Keep the planner setting at its normal value for application queries.

## Optional: train centroids without an index

Use a different output name so this can run after the indexed example:

```sql
SELECT vextor.create_vector_index(
    'public.vextor_demo_items',
    'features',
    ARRAY[4],
    sampling_factor => 40,
    skip_index_build => true,
    centroid_table => 'vextor_demo_training_centroids'
);
```

The returned `index` field is `null`. The call creates only the centroid table.

## Options and current limits

| Option | Meaning |
| --- | --- |
| `lists = ARRAY[leaves]` | Flat centroid tree. |
| `lists = ARRAY[roots, leaves]` | Two-level centroid tree. Query probes need two values, such as `2,4`. |
| `distance_operator` | `l2`, `ip`, or `cos`; query with `<->`, `<#>`, or `<=>`, respectively. |
| `spherical_centroids` | Normalize centroids; useful for directional comparisons. |
| `sampling_factor` | Maximum sampled vectors per leaf. |
| `batch_size` | Worker read and computation batch size. |
| `kmeans_iterations`, `kmeans_nredo` | Training iterations and restarts. |
| `seed` | Repeatable sampling and initialization for a fixed source order. |
| `memory_budget_bytes` | Explicit buffer budget; default 256 MiB. It does not cap total PostgreSQL/worker memory use. |
| `worker_timeout_ms` | Worker deadline; default 300,000 milliseconds. PostgreSQL statement timeouts also apply. |
| `residual_quantization` | Forwarded to the VectorChord index options. |

The first three function arguments are source table, column name, and `lists`.
See the [complete SQL signature](pg_vextor/sql/pg_vextor--0.1.0.sql) for defaults.
Output names are individual identifiers in the source schema, with at most 63 bytes.

Current source support is dense pgvector `vector` columns in ordinary heap tables
without inheritance. The reader rejects NULL vectors, mixed dimensions, and active
row-level security. `halfvec`, `sparsevec`, and `bit` are not supported.
The caller needs SELECT permission on the source column and normal permissions for
creating the outputs. Installing the extension requires a superuser.

Probe settings must match the number of levels in the index being queried.
Use a separate demo table when experimenting with a different tree depth.
The CPU kernels use borrowed metric enums and explicit pointer contracts.
They compute only the quantities required by the selected metric.
Builds require Vx's flat code generator. Explicit SIMD tuning remains future work.
See [the Vx main update](docs/vx_main_update_2026-11-04.md) for rationale and validation.
See the [roadmap](docs/pg_vextor.md) for correctness validation, SIMD, GPU enablement,
and the later comparison against cuVS.

## Clean up the example

These commands remove only the example tables. Dropping the source table also
removes its index:

```sql
DROP TABLE public.vextor_demo_items;
DROP TABLE public.vextor_demo_centroids;
DROP TABLE IF EXISTS public.vextor_demo_training_centroids;
```

Exit `psql` with `\q`. To stop the Docker service while retaining the database:

```sh
docker-compose down
```

## Run tests

Run test commands from the Vextor repository in your terminal. Exit `psql` with
`\q` first, or open another terminal. Keep the Docker context used to start the service.

### Quick database checks

These require the running Docker service. They need no Faiss installation or dataset downloads:

```sh
./scripts/check_postgres.sh
./scripts/check_extension.sh
python3 -B scripts/check_snapshot.py
python3 -B scripts/check_recall.py
```

The checks cover installed extensions, index queries, worker failures, permissions,
cancellation, rollback, snapshot consistency, and synthetic recall against exact scans.
They use dedicated fixture schemas. Most fixtures roll back. The snapshot check
commits changes in its own schema and removes that schema afterward.
The synthetic suite reports 24 index builds and 120 recall queries on success.

The default container is `vextor-postgres-1`. If you changed the Compose project
name, set `VEXTOR_POSTGRES_CONTAINER` to its PostgreSQL container name.

### Prepare the native CPU tests

This step builds a library for your host operating system, separate from the Linux
worker inside Docker. Install a C/Clang compiler, Rust, LLVM/MLIR 22, and Z3 using
the [pinned Vx build instructions](https://github.com/minzyfox/Vx/tree/c9612a1f1c8a2819258b159774a615f9aca298a7#quickstart).
Use LLVM major version 22 even if a newer version is installed.

Set `VX_ROOT` to your pinned Vx checkout. It is already set if you followed the
fresh-checkout commands above. For another checkout, use
`export VX_ROOT=/absolute/path/to/Vx` with your actual path.
Build the compiler with its commit in the version string:

```sh
(
  cd "$VX_ROOT"
  ./setup.sh
  source config.local
  VX_VERSION_WITH_COMMIT=1 cargo build --locked -j 2 --bin vxc -p vxc
  cargo build --locked -j 2 -p vx_std_core
)
./scripts/build.sh
python3 -m venv build/correctness-venv
build/correctness-venv/bin/python -m pip install -r tests/requirements-correctness.txt
```

The subshell returns you to Vextor after building Vx.
Native builds require a clean tracked Vx checkout at the pinned revision.
The library is `build/libvextor_kmeans.dylib` on macOS or
`build/libvextor_kmeans.so` on Linux. Python commands use the environment directly;
you do not need to activate it.

Run the CPU-only suite without PostgreSQL or downloaded datasets:

```sh
bash scripts/check_cpu.sh
```

The runner builds the native worker, then runs 25 library/adapter/worker tests and ten objective-scorer/parser tests,
48 matched Faiss cases, 96 independent Lloyd cases, six zero/duplicate cases,
and allocation-failure/sanitizer checks. It exits nonzero if a check fails.

### Complete CPU and database suite

Prepare the native tests above and start PostgreSQL first.
Download the GloVe and SIFT datasets once, about 625 MB combined:

```sh
mkdir -p build/datasets
curl --fail --location --retry 3 https://ann-benchmarks.com/glove-25-angular.hdf5 -o build/datasets/glove-25-angular.hdf5
curl --fail --location --retry 3 https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o build/datasets/sift-128-euclidean.hdf5
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c tests/datasets.sha256
else
  shasum -a 256 -c tests/datasets.sha256
fi
```

Keep these files under ignored `build/datasets/`. Their recorded hashes identify
the fixtures used by the correctness suite.
Run the full suite in Bash or Zsh and save its output:

```sh
set -o pipefail
VEXTOR_CHECK_POSTGRES=1 bash scripts/check_cpu.sh 2>&1 | tee build/full-checks.log
```

This includes the native and database checks above, plus 24 paired Vx/VectorChord
internal index builds and 3,072 real-data recall queries. The tests use deterministic
subsets of 8,192 training rows and 32 held-out queries per dataset.
They apply declared recall gates; low-probe queries need not return exact results.
The full run can take several minutes under emulation.
Read `build/full-checks.log` when a check fails. `pipefail` preserves the failure status
when output passes through `tee`.

To rerun only the real-data comparison:

```sh
build/correctness-venv/bin/python -B scripts/check_real_recall.py
```

### Compiler contracts and generated code

After building the pinned native compiler, run these separately:

```sh
build/correctness-venv/bin/python -B scripts/check_contracts.py
build/correctness-venv/bin/python -B scripts/check_kernel_codegen.py
```

These check helper contracts, generated pointer/alias information, and mutations
that must fail at specific assertions. The nearest-index loop and variable-dimension
offset examples now pass. Kernel mutations check the result bound at the accumulation
caller and reject invalid buffer offsets.
These checks do not establish a proof of the complete kernel.
Use `"$VX_ROOT/target/debug/vxc" src/kmeans.vx --audit` to list trusted pointer clauses.
The audit identifies trust boundaries; it does not prove them.

The [November 4 upgraded-worker results](docs/evidence/vx_main_test_summary_2026-11-04.json)
and [full recorded log](docs/evidence/vx_main_checks_2026-11-04.txt) used Vx `7cc3d2d9`.
See [the proof update](docs/m4_index_proofs_2026-11-05.md) for the current pin.
See [M4 details](docs/milestone_4_cpu_correctness.md) for tolerances and remaining work.

### Internal trainer objectives

The optional [internal objective comparison](docs/internal_objectives.md) builds a
separate, instrumented VectorChord image, captures unrotated centroids, and scores
both trainers on the same real vectors. Follow that guide to run it in an isolated
Compose project. The ordinary database image does not export internal centroids.

## CPU correctness baseline

The M4 synthetic baseline compares matched samples and initial centroids with
Faiss CPU, then checks SQL index recall against exact PostgreSQL scans.
See [M4 checks and recorded results](docs/milestone_4_cpu_correctness.md) for
setup, tolerances, query settings, and remaining validation work.
The default-epsilon experiment missed the recall target. The conservative
all-probe, epsilon-4 run passed; this does not establish production recall or speed.

The November 4 extension adds real GloVe/SIFT comparisons against VectorChord's
internal CPU trainer: 24 builds and 3,072 queries passed the declared recall gates.
It also fixes extreme-input normalization and rejects centroid-sum overflow.
The optional internal clustering-objective comparison also passes six quality
groups using an isolated instrumented reference image; see [results and limits](docs/internal_objectives.md).
The CPU CI workflow is implemented. Full kernel proofs and a successful hosted CI
run remain open.

## Standalone library development

Follow [Prepare the native CPU tests](#prepare-the-native-cpu-tests) to build the
host library and Python environment. The following commands are optional benchmarks
and a GPU compilation check, rather than part of the correctness suite:

```sh
./build/benchmark_data_path 100000 16 8 40 64
build/correctness-venv/bin/python scripts/benchmark.py --rows 10000 --dims 128 --leaves 32
VX_CHECK_GPU=1 ./scripts/build.sh
```

The benchmark executables and libraries live in ignored `build/`.
The GPU command checks compilation only. It does not validate execution on a GPU.

## Library API

[`include/vextor_kmeans.h`](include/vextor_kmeans.h) defines the C ABI.
The caller supplies the input, output, and scratch arrays. The function returns
zero on success and one for invalid arguments or float32 centroid-sum overflow.
It makes one zero-vector root.
For `root_count = 0`, it trains flat leaves. For `root_count > 0`, it trains
leaves first, then roots from those leaves, and assigns each leaf to its nearest
root. This forms the centroid tree VectorChord accepts.

The distance codes are squared L2, negative inner product, and cosine distance.
The function samples up to `leaf_count * sampling_factor` distinct rows using a
seeded permutation. It processes sampled rows in batches. Each restart begins
from different sampled rows, and the lowest sample objective wins. Empty
clusters use a sampled row as their replacement. Zero-length vectors stay zero
when spherical centroids are enabled.

The original array API requires resident input. The CPU host adapter in
[`include/vextor_data_path.h`](include/vextor_data_path.h) accepts a streaming row
reader instead. It validates all rows, collects a repeatable bounded sample,
checks a buffer memory budget before allocation, and calls the same Vx library.
It reports read/validation, sampling, and clustering timings. Its sampler uses
Floyd's algorithm with SplitMix64 rather than the array API's permutation.
Compare the adapter against direct Vx calls on its collected sample.

The reader must provide a stable row order and count/scan snapshot. The adapter
checks the declared row count but cannot enforce a database snapshot itself.
The PostgreSQL extension supplies the reader, worker isolation, and index creation described above.
GPU acceleration comes later. See [M2 details](docs/milestone_2_data_path.md).

`./scripts/check_data_path.sh` runs native allocation-failure checks with address
and undefined-behavior sanitizers. `build/benchmark_data_path` generates rows one
at a time. Its positional arguments are rows, dimensions, leaves, sampling factor,
batch size, and optional buffer budget in bytes (default 64 MiB).
The reported budget covers adapter buffers. It excludes caller-owned source
storage, allocator overhead, and Vx runtime memory.

`src/gpu_lloyd.vx` is a separate GPU prototype that accepts device-resident
samples and initialized centroids. `VX_CHECK_GPU=1` checks that Vx generates
A100 code without requiring GPU hardware. The generated code has no parallel
launch metadata for the assignment and reduction loops. This path is untested
on an NVIDIA device and is not a performance implementation.
