# VxVector

VxVector is a PostgreSQL extension that trains k-means centroids in Vx and uses them
to build a VectorChord vector-search index.

- **pgvector** provides the `vector` data type and distance operators.
- **VxVector** samples source vectors and trains centroids in a separate worker process.
- **VectorChord** builds the `vchordrq` index and serves nearest-neighbor queries.

The extension is `pg_vxvector` 0.1.0. Its SQL API uses the `vxvector` schema.
See [the rename notes](docs/rename_vxvector.md) for changed names and validation.
The repository URL and checkout folder remain `vextor`.
It is intended for development and testing. Installation packages, upgrades, and
production-scale recall and performance still need validation.

## What works today

| Capability | Status |
| --- | --- |
| Training from SQL, then building a VectorChord index | Works on the **CPU** in the Docker PostgreSQL image below, and on the **Apple GPU** in native macOS PostgreSQL |
| Training on an Apple GPU (Metal) | Works in the standalone worker, for every option: L2, inner product, cosine, restarts, spherical centroids, and two-level trees. Native PostgreSQL uses the same worker selection path |
| Automatic GPU or CPU selection | `auto`, `cpu`, or `gpu`, in the standalone worker and from SQL through `vxvector.backend_mode`, with the reason reported |
| PostgreSQL using the GPU | Validated on native macOS 27.2.1 with an Apple M2 Pro: SQL to Metal training to a VectorChord index, failure handling, and the recall gates. Linux Docker falls back to the CPU. See the [Step 6 record](docs/step6_native_macos_2026-11-15.md) |
| Installing into your own PostgreSQL | Validated on one native macOS setup with Homebrew PostgreSQL 17. VectorChord must be built without stripping; see [native PostgreSQL setup](#install-into-an-existing-postgresql) |
| NVIDIA GPUs | Built from the same Vx kernels as Metal (`scripts/build_gpu_cuda.sh`). It compiles and links on Linux, but has not run on an NVIDIA GPU and is not in the PostgreSQL image; see [Library API](#library-api) |

## Choose the PostgreSQL deployment

The Docker setup below runs PostgreSQL and the VxVector worker inside a **Linux
container** (currently `linux/amd64`, including under Docker Desktop or Colima on
Apple Silicon). It is useful for the SQL/VectorChord workflow, but it cannot use
the Mac's Metal GPU; `auto` selects CPU there.

To use an Apple GPU from SQL, PostgreSQL and the worker must both run natively on
macOS, with the Metal library beside the worker. Follow
[Install into an existing PostgreSQL](#install-into-an-existing-postgresql)
instead. That path is validated on one machine; see the
[Step 6 record](docs/step6_native_macos_2026-11-15.md).

GPU results match the CPU bit for bit for L2 and the inner product in every test. Cosine,
and spherical centroids on some inputs, can differ at float32 near-ties within a
declared tolerance: an objective within 1e-3 of the CPU's, and at most 3.5e-4 observed.
See the [Step 5 record](docs/step5_gpu_options.md).

Measured on an Apple M2 Pro, 100,000 SIFT-128 rows and 256 leaves: GPU training takes
0.28 to 1.0 seconds, 24 to 28 times faster than the CPU kernel at `-O3`. The CPU kernel
uses one thread. These are single-machine measurements, not benchmarks.

The [active roadmap](docs/pg_vxvector.md) is the single source of truth for implementation order.
The dated records under `docs/` describe each step's validation. Their raw logs
(`docs/evidence/`) are kept outside the repository.

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

### Get VxVector and the pinned Vx source

These commands create new checkouts. Run them from the directory where you want
VxVector to live:

```sh
git clone https://github.com/minzyfox/vextor.git
cd vextor
git clone https://github.com/minzyfox/Vx.git ../Vx
export VX_ROOT="$(cd ../Vx && pwd)"
git -C "$VX_ROOT" fetch origin "$(cat vx-revision.txt)"
git -C "$VX_ROOT" checkout --detach "$(cat vx-revision.txt)"
cp .env.example .env
```

Edit `.env` and set `VXVECTOR_POSTGRES_PASSWORD` to a local development password.
The default port is `15432`. Keep an existing `.env` when reusing a database volume.
If you already have both checkouts, set `VX_ROOT` to a checkout at the exact
commit in [vx-revision.txt](vx-revision.txt). Use a separate checkout if your
existing Vx tree contains work you want to keep.

The current pin is `423a10dbbd91230df290494f9705a746dc444baf`, a commit on Vx `main`.
[The pin update record](docs/vx_pin_update_2026-11-14.md) validates the native build at this pin.
The same record covers the Linux worker, the extension, and the PostgreSQL image at this pin.
The scripts default to `../../Vx` when `VX_ROOT` is unset. The fresh-checkout
example overrides that default with its adjacent `../Vx` directory.

### Build the Linux Docker images

Run the remaining shell commands from the VxVector repository.
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

The complete image is `vxvector-postgres:17-vxvector`.
Its base development image is `vxvector-postgres:17-dev`.
Both native and Linux builds use the Vx pin. Build manifests record compiler,
kernel source, and object hashes. The installed database image carries its manifest
at `/usr/local/libexec/vxvector/vx-build-info.json`.

## Start the Linux Docker setup

Run these shell commands from the VxVector repository using your Docker context.
On the current development machine, that context is `colima`.
Colima users can select it with `export DOCKER_CONTEXT=colima`.
The first command starts the service. The second opens the SQL prompt:

```sh
docker-compose up -d --no-build --wait
docker-compose exec postgres psql -U vxvector -d vxvector
```

Use `docker compose` instead of `docker-compose` if the Compose plugin is installed.
The database listens at `127.0.0.1:15432`, with database and user both named `vxvector`.
The password is in the ignored `.env` file. This service uses its own data volume.
The existing database on port 5432 is separate.

If the images are unavailable, follow [Build from source](#build-from-source) first.
Run the following SQL sections in the same `psql` session.
Enter SQL at the database prompt, and shell commands in your terminal.

## Create the extensions

A fresh VxVector volume initializes these extensions automatically. These statements
also work when connecting to an existing VxVector database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vchord;
CREATE EXTENSION IF NOT EXISTS pg_vxvector;
```

The Docker service preloads `vchord`. A PostgreSQL installation outside this image
also needs the extension files, worker, and preload configuration.
See the [PostgreSQL setup notes](docs/postgres.md).

## Create a table and insert dummy data

This example creates 1,000 three-dimensional vectors in four separated groups.
The data is deterministic, so repeating it gives the same input.
Use these example names in a database where they do not already exist.

```sql
CREATE TABLE public.vxvector_demo_items (
    id integer PRIMARY KEY,
    label text NOT NULL,
    features vector(3) NOT NULL
);

INSERT INTO public.vxvector_demo_items (id, label, features)
SELECT
    i,
    format('group-%s', i % 4),
    ARRAY[
        (i % 4) * 10 + (i % 17) * 0.01,
        (i % 7) * 0.01,
        (i % 11) * 0.01
    ]::real[]::vector(3)
FROM generate_series(1, 1000) AS i;

SELECT count(*) AS source_rows FROM public.vxvector_demo_items;
```

The count should be `1000`. In an application, replace these dummy values with
embeddings from your model. All vectors must have the same dimension and finite
components.

## Train centroids and build an index

```sql
SELECT vxvector.create_vector_index(
    'public.vxvector_demo_items',
    'features',
    ARRAY[4],
    sampling_factor => 40,
    batch_size => 64,
    kmeans_iterations => 10,
    kmeans_nredo => 3,
    distance_operator => 'l2',
    seed => 42,
    centroid_table => 'vxvector_demo_centroids',
    index_name => 'vxvector_demo_idx'
);

SELECT id, parent, vector
FROM public.vxvector_demo_centroids
ORDER BY id;

ANALYZE public.vxvector_demo_items;
```

`ARRAY[4]` requests four leaf centroids. The function samples up to `4 * 40 = 160`
source vectors, trains centroids, then indexes all 1,000 source rows.
The centroid table contains five nodes: a zero-vector root and four leaves.
The root has `parent = NULL`; each leaf has parent `0`.

The function returns JSON with the selected `backend`, the selection reason, device
and GPU version when applicable, output names, row/sample counts, buffer bytes, and
stage timings. Backend mode defaults to `auto`: a ready GPU is used when it supports
the request, otherwise training uses the CPU. The Linux Docker image has no GPU
library and therefore selects CPU. A native macOS worker built with the Metal library
can select the Apple GPU.

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
FROM public.vxvector_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;
```

For L2 distance, smaller values mean closer vectors. This query should find
vectors from `group-1`, whose first coordinate is near 10.

Inspect the query plan:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT id
FROM public.vxvector_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;
```

PostgreSQL can prefer a sequential scan for a small table. To check index use
explicitly in this demo, temporarily discourage sequential scans:

```sql
SET enable_seqscan = off;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT id
FROM public.vxvector_demo_items
ORDER BY features <-> '[10,0,0]'::vector
LIMIT 5;

RESET enable_seqscan;
```

The plan should include an index scan using `vxvector_demo_idx`.
Keep the planner setting at its normal value for application queries.

## Optional: train centroids without an index

Use a different output name so this can run after the indexed example:

```sql
SELECT vxvector.create_vector_index(
    'public.vxvector_demo_items',
    'features',
    ARRAY[4],
    sampling_factor => 40,
    skip_index_build => true,
    centroid_table => 'vxvector_demo_training_centroids'
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
| `batch_size` | Rows the worker reads per batch. It does not change the result. |
| `kmeans_iterations`, `kmeans_nredo` | Training iterations and restarts. |
| `seed` | Repeatable sampling and initialization for a fixed source order. |
| `memory_budget_bytes` | Explicit buffer budget; default 256 MiB. It does not cap total PostgreSQL/worker memory use. |
| `worker_timeout_ms` | Worker deadline; default 300,000 milliseconds. PostgreSQL statement timeouts also apply. |
| `residual_quantization` | Forwarded to the VectorChord index options. |

The superuser-only `vxvector.backend_mode` setting accepts `auto` (default), `cpu`,
or `gpu`. Use `cpu` to force CPU training or `gpu` to require GPU training and return
an error if no compatible device is available. For example:

```sql
SET vxvector.backend_mode = 'gpu';
SELECT vxvector.create_vector_index(
    'public.items', 'embedding', ARRAY[4],
    skip_index_build => true
);
RESET vxvector.backend_mode;
```

The first three function arguments are source table, column name, and `lists`.
See the [complete SQL signature](pg_vxvector/sql/pg_vxvector--0.1.0.sql) for defaults.
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
Builds require Vx's flat code generator. The CPU kernel is compiled at `-O3`, which
gives the same bits as `-O0` ([record](docs/cpu_o3_2026-11-14.md)). It uses one thread,
and explicit SIMD and threading remain future work.
See [the Vx main update](docs/vx_main_update_2026-11-04.md) for rationale and validation.
See the [active roadmap](docs/pg_vxvector.md) for PostgreSQL GPU integration and the remaining work.

## Install into an existing PostgreSQL

This path is validated on macOS 27.2.1 with an Apple M2 Pro and Homebrew
`postgresql@17`; see the [Step 6 record](docs/step6_native_macos_2026-11-15.md).
On Apple Silicon, run the builds as arm64. Prefix them with `arch -arm64` if your
terminal runs under Rosetta.

You need PostgreSQL 17 with its server headers, pgvector, and VectorChord. The image
uses pgvector 0.8.2 and VectorChord 1.1.1, with `shared_preload_libraries = 'vchord'`.

On macOS 27.2.1, dyld refuses stripped extension bundles with `mis-aligned LINKEDIT
string pool`, so build VectorChord 1.1.1 from source with stripping disabled:

```sh
cd VectorChord-1.1.1
PG_CONFIG="$(command -v pg_config)" CARGO_PROFILE_RELEASE_STRIP=none \
  cargo run -p xtask -- build
make install
```

1. Build the native Vx CPU library and worker with `./scripts/build.sh`, then build
   the Metal library with `arch -arm64 ./scripts/build_gpu.sh`. The GPU build places
   `libvxvector_gpu.dylib` next to the worker, where it is discovered automatically.
2. Build the extension against the same PostgreSQL installation. Run this from the
   repository, with `pg_config` from your PostgreSQL:

   ```sh
   ./scripts/build_macos_extension.sh
   ```

3. Install the extension where your server looks for it:
   - `pg_vxvector/target/debug/libpg_vxvector.dylib` to `$(pg_config --pkglibdir)`, named
     `pg_vxvector.dylib`
   - `pg_vxvector/pg_vxvector.control` and `pg_vxvector/sql/pg_vxvector--0.1.0.sql` to `$(pg_config --sharedir)/extension/`
4. Point the extension at the native worker by setting `vxvector.worker_path` in
   `postgresql.conf` to the absolute path of `build/vxvector-worker`. The worker loads
   the adjacent `libvxvector_kmeans.dylib` and, if built, `libvxvector_gpu.dylib`.
   The libraries and the pinned Vx runtime must remain readable by the server's
   operating-system user.
5. Restart the server, then run `CREATE EXTENSION vector; CREATE EXTENSION vchord; CREATE EXTENSION pg_vxvector;`.

Then run the [SQL walkthrough](#create-a-table-and-insert-dummy-data). `auto` selects
Metal when the native worker can load the GPU library and probe a device; otherwise
it uses CPU. The returned JSON identifies the backend. Set `vxvector.backend_mode`
to `cpu` or `gpu` to force or require a mode.

## Clean up the example

These commands remove only the example tables. Dropping the source table also
removes its index:

```sql
DROP TABLE public.vxvector_demo_items;
DROP TABLE public.vxvector_demo_centroids;
DROP TABLE IF EXISTS public.vxvector_demo_training_centroids;
```

Exit `psql` with `\q`. To stop the Docker service while retaining the database:

```sh
docker-compose down
```

## Run tests

Run test commands from the VxVector repository in your terminal. Exit `psql` with
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

The default container is `vxvector-postgres-1`. If you changed the Compose project
name, set `VXVECTOR_POSTGRES_CONTAINER` to its PostgreSQL container name.

### Prepare the native CPU tests

This step builds a library for your host operating system, separate from the Linux
worker inside Docker. Install a C/Clang compiler, Rust, LLVM/MLIR 22, and Z3 using
the [pinned Vx build instructions](https://github.com/vx-lang/Vx/tree/423a10dbbd91230df290494f9705a746dc444baf#quickstart).
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

The subshell returns you to VxVector after building Vx.
Native builds require a clean tracked Vx checkout at the pinned revision.
The library is `build/libvxvector_kmeans.dylib` on macOS or
`build/libvxvector_kmeans.so` on Linux. Python commands use the environment directly;
you do not need to activate it.

Run the CPU-only suite without PostgreSQL or downloaded datasets:

```sh
bash scripts/check_cpu.sh
```

The runner builds the native worker and a stand-in GPU library, then runs 54 unit
tests: library, adapter, worker, objective-scorer, and backend-selection tests. Next
come 48 matched Faiss cases, 96 independent Lloyd cases, six zero/duplicate cases, and
allocation-failure and sanitizer checks. When `scripts/build_gpu.sh` has built the
Metal library, three of the unit tests also train on the real GPU. Otherwise they are
skipped. The runner exits nonzero if a check fails.

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
VXVECTOR_CHECK_POSTGRES=1 bash scripts/check_cpu.sh 2>&1 | tee build/full-checks.log
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

The audit lists 56 trusted clauses at the current pin, all at C boundaries
([Step 5 record](docs/step5_gpu_options.md#results)).
See [the proof update](docs/m4_index_proofs_2026-11-05.md) for the proof history.
See the [CPU correctness reference](docs/milestone_4_cpu_correctness.md) for recorded tolerances and evidence.
Use the [active roadmap](docs/pg_vxvector.md) for remaining work.

### Internal trainer objectives

The optional [internal objective comparison](docs/internal_objectives.md) builds a
separate, instrumented VectorChord image, captures unrotated centroids, and scores
both trainers on the same real vectors. Follow that guide to run it in an isolated
Compose project. The ordinary database image does not export internal centroids.

## CPU correctness baseline

The M4 synthetic baseline compares matched samples and initial centroids with
Faiss CPU, then checks SQL index recall against exact PostgreSQL scans.
See [M4 checks and recorded results](docs/milestone_4_cpu_correctness.md) for
historical setup, tolerances, query settings, and validation limits.
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

### Apple GPU (Metal) training

On macOS with Apple silicon, build the Metal library and check it against the CPU
kernel. Use `arch -arm64` if your shell runs under Rosetta. Download the SIFT dataset
first, as in [Complete CPU and database suite](#complete-cpu-and-database-suite),
or the SIFT cases are skipped:

```sh
arch -arm64 ./scripts/build.sh
arch -arm64 ./scripts/build_gpu.sh
arch -arm64 build/correctness-venv/bin/python -B scripts/check_gpu_l2.py
arch -arm64 bash scripts/check_cpu.sh
```

`check_gpu_l2.py` runs 97 GPU training runs: every metric, spherical centroids, and
split launches. It compares each with the CPU kernel, an independent NumPy reference,
and the Metal dispatch log. `check_cpu.sh` then also runs the real-GPU worker tests.

The native worker loads `libvxvector_gpu.dylib` from beside itself, or from
`VXVECTOR_GPU_LIBRARY`, so CPU-only installs need no GPU library. Worker protocol
version 2 takes a mode (`auto`, `cpu`, or `gpu`). It reports the backend, the reason
for it, the device, the library version, and the GPU's allocation, copy, and training
times. Version 1, which the extension sends, always trains on the CPU.

Each GPU launch is kept to about 2 ms of work. macOS can stop a long launch, and the Vx
runtime then aborts the process. Training runs in the worker process, so an abort ends
the worker, not PostgreSQL.

Details are in the [Step 3](docs/step3_metal_l2.md), [Step 4](docs/step4_backend_selection.md),
and [Step 5](docs/step5_gpu_options.md) records. Vx issues found along the way are drafted in
[docs/vx_issue_drafts_2026-11-14.md](docs/vx_issue_drafts_2026-11-14.md).

## Library API

[`include/vxvector_kmeans.h`](include/vxvector_kmeans.h) defines the C ABI.
The caller supplies the input, output, and scratch arrays. The function returns
zero on success and one for invalid arguments or float32 centroid-sum overflow.
It makes one zero-vector root.
For `root_count = 0`, it trains flat leaves. For `root_count > 0`, it trains
leaves first, then roots from those leaves, and assigns each leaf to its nearest
root. This forms the centroid tree VectorChord accepts.

The distance codes are squared L2, negative inner product, and cosine distance.
The function samples up to `leaf_count * sampling_factor` distinct rows. When that
is every row, it uses them in order. Otherwise SplitMix64 and Floyd's algorithm
choose them from the seed. Each restart begins from different sampled rows, and
the lowest sample objective wins. Empty
clusters use a sampled row as their replacement. Zero-length vectors stay zero
when spherical centroids are enabled.

The array API (`vxvector_kmeans_v2`) requires resident input. The CPU host adapter in
[`include/vxvector_data_path.h`](include/vxvector_data_path.h) accepts a streaming row
reader instead. It validates all rows, collects a repeatable bounded sample,
checks a buffer memory budget before allocation, and trains with the same kernel.
It reports read/validation, sampling, and clustering timings. Both APIs use one
sampler, so they train on the same rows and produce the same centroids for the
same seed. The adapter's logic is in Vx, in `src/kmeans.vx`. Its C side,
`src/data_path.c`, allocates the buffers, supplies the clock, and writes the
error messages.

The reader must provide a stable row order and count/scan snapshot. The adapter
checks the declared row count but cannot enforce a database snapshot itself.
The PostgreSQL extension supplies the reader, worker isolation, and index creation described above.
See [M2 details](docs/milestone_2_data_path.md).

`vxvector_select_backend_v1` and `vxvector_run_v2` add backend selection. Selection
happens before any row is read. With the GPU selected, training runs through the
functions of [`include/vxvector_gpu.h`](include/vxvector_gpu.h), and the result has the
CPU's layout. A GPU failure after training starts is reported, and the CPU does not
train again.

`./scripts/check_data_path.sh` runs native allocation-failure checks with address
and undefined-behavior sanitizers. `build/benchmark_data_path` generates rows one
at a time. Its positional arguments are rows, dimensions, leaves, sampling factor,
batch size, and optional buffer budget in bytes (default 64 MiB).
The reported budget covers adapter buffers. It excludes caller-owned source
storage, allocator overhead, and Vx runtime memory.

The GPU kernels and training loop are in `src/gpu_kmeans.vx`, for Apple and
NVIDIA GPUs alike. `src/gpu_l2.c` calls them, and one small backend file per GPU
(`src/gpu_backend_metal.m`, `src/gpu_backend_cuda.c`) supplies memory, copies,
and waits. `VX_CHECK_GPU=1 ./scripts/build.sh` compiles the kernels to A100 PTX
without NVIDIA hardware. On Linux with the CUDA toolkit,
`CUDA_HOME=/usr/local/cuda ./scripts/build_gpu_cuda.sh` builds
`build/linux/libvxvector_gpu.so`. Build Vx itself where the toolkit is
installed: only then does it build the CUDA dispatch library that runs the
kernels on the GPU, and the script refuses any other. The NVIDIA library has not
run on an NVIDIA GPU yet and is not part of the PostgreSQL image.
The earlier CUDA Driver API prototype (`src/gpu_lloyd.vx`, `src/gpu_cuda.cpp`,
`scripts/test_gpu_lloyd_cuda.sh`) remains until that run passes.
