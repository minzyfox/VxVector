# Milestone 3: CPU worker and PostgreSQL extension

The CPU integration uses PostgreSQL 17, pgvector, VectorChord, and the real Vx k-means implementation.
The pgrx extension is `pg_vextor` version 0.1.0. GPU acceleration remains M5 work.

## Build and run

Run these commands from the Vextor repository:

```sh
export DOCKER_CONTEXT=colima
./scripts/build_postgres_image.sh
./scripts/build_linux_worker.sh
./scripts/build_extension.sh
docker-compose up -d --no-build --wait
docker-compose exec postgres psql -U vextor -d vextor \
  -c 'CREATE EXTENSION IF NOT EXISTS pg_vextor;'
```

The base image is `vextor-postgres:17-dev`. The complete image is `vextor-postgres:17-vextor`.
Compose uses the complete image by default, with the existing Vextor volume and loopback port 15432.
The original database on port 5432 uses a separate container and volume.
The extension initializer runs automatically for a fresh Vextor database volume.
The explicit `CREATE EXTENSION` command also handles an existing volume.

The Linux compiler build uses the committed HEAD of `VX_ROOT`, which defaults to `../../Vx`.
It records that revision in `build/linux/vx-revision.txt`.
The script expects the cached `vx-build:llvm22` image from Vx's `docker/Dockerfile.build`.
If needed, build that prerequisite first:

```sh
docker build -f ../../Vx/docker/Dockerfile.build -t vx-build:llvm22 ../../Vx
```

The first compiler build is slow in the emulated x86 VM. Later builds reuse dedicated caches.
The compiler container is `vextor-vx-build`, backed by `vextor_vx_build`.
The pgrx container is `vextor-extension-build`, backed by `vextor_extension_build` and `vextor_extension_cargo`.
Both build scripts use two Cargo jobs. Logs and Linux artifacts remain under `build/linux/`.
The extension build script compiles pgrx inside its builder, then packages that shared library in `docker/Dockerfile.extension`.
That Dockerfile also links the generated Vx object and C adapter into the worker library.
The backend library does not load Vx, LLVM, CUDA, or the worker library.

## SQL API

For an existing table with a `vector(3)` column named `features`, train centroids without creating an index:

```sql
SELECT vextor.create_vector_index(
    'public.items', 'features', ARRAY[32],
    skip_index_build => true,
    centroid_table => 'items_training_centroids'
);
```

To train centroids and create a VectorChord index, choose a different output table:

```sql
SELECT vextor.create_vector_index(
    'public.items', 'features', ARRAY[32],
    centroid_table => 'items_index_centroids',
    index_name => 'items_vextor_idx'
);
SET vchordrq.probes = '8';
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM public.items
ORDER BY features <-> '[0.1,0.2,0.3]' LIMIT 5;
```

`lists = ARRAY[leaves]` trains a flat tree. `ARRAY[roots,leaves]` trains a two-level tree.
Queries need one probe value per tree level, such as `8` for flat or `2,8` for two levels.
Probe counts affect recall and query cost. The image tests use small trees and inspect every leaf.

The function also accepts `sampling_factor`, `batch_size`, `kmeans_iterations`, `kmeans_nredo`, and `distance_operator` (`l2`, `ip`, `cos`).
Other options are `spherical_centroids`, `residual_quantization`, `seed`, `memory_budget_bytes`, and `worker_timeout_ms`.
The default buffer budget is 256 MiB. The default worker deadline is 300,000 milliseconds.
See `pg_vextor/sql/pg_vextor--0.1.0.sql` for the complete signature and defaults.

Output names are individual identifiers in the source table's schema.
Defaults are `<source>_vextor_centroids` and `<source>_vextor_idx`.
Names must contain 1 through 63 bytes. Supply shorter names for a long source name.
Existing output tables and indexes cause an error. The function does not replace them.

Each successful call returns JSON containing the backend, output names, source/sample counts, dimensions, node count, seed, buffer bytes, and stage timings.
Every call uses CPU and emits `GPU acceleration is unavailable; running CPU k-means`.
GPU routing and optional loading remain deferred to M5.

## Snapshot and memory behavior

The pgrx wrapper enters the C bridge through a guarded FFI boundary.
The bridge checks source-column permissions and requires dense pgvector `vector` input.
It currently supports ordinary heap tables without inheritance.
It rejects tables with active row-level security instead of bypassing their policies.
Callers need source-column SELECT permission and normal permissions for creating the outputs.
The function does not use `SECURITY DEFINER`.

The bridge registers one active snapshot for both counting and scanning.
Heap scans disable synchronized scan starts to preserve physical row order.
Concurrent inserts and updates remain outside that snapshot.
It sends one validated float32 row at a time over a local socket to a separate worker process.
The worker uses M2's bounded batches, seeded sampling, allocation checks, and Vx C ABI.
It returns centroids and parents only after training succeeds.

Planning reserves backend output buffers, row conversion space, and IPC storage before starting the worker.
The worker separately budgets its input row and every M2 buffer.
The budget bounds these explicit buffers. It does not bound total process RSS, PostgreSQL executor/index-build memory, or allocator overhead.
Source tables are never materialized in a resident float matrix.
The result includes count, transfer, worker-read, sampling, clustering, centroid-write, and index-build timings.
Worker-read and parent-transfer time overlap. They are not additive stage totals.

## Failure and cancellation

The bridge polls the worker socket and checks PostgreSQL interrupts during reads, writes, and waits.
A worker failure, malformed response, memory rejection, or deadline becomes a SQL error.
Statement timeouts and query cancellation stop and reap the worker.
A backend-exit callback also kills and reaps an active worker.
The server creates outputs only after receiving valid results and a successful worker exit.
Centroid writes and index creation use the caller's transaction.
A later index error rolls back centroid creation as well.

`vextor.worker_path` selects the installed worker executable.
Only superusers can change it. Regression checks use it to inject worker failure, abort, and delay.
The local IPC protocol is versioned and requires matching backend/worker architecture and ABI.
It is not a network protocol.

## Validation

```sh
./scripts/check_postgres.sh
./scripts/check_extension.sh
python3 -B scripts/check_snapshot.py
```

The SQL checks run in one transaction and roll back their fixtures.
They check centroid-only builds, batch repeatability, flat/two-level trees, actual index scans, all three metrics, invalid input, permissions, and row-level security rejection.
They also check the fallback warning, a worker abort, worker timeout, statement cancellation, process cleanup, and rollback after an index error.
The snapshot check commits concurrent source inserts and updates after counting, then compares centroids with the original source.
Four native worker tests check protocol errors, budget rejection, and centroid values from the real Vx implementation.
The M2 reference and allocation/sanitizer checks remain part of validation.

Validation targets linux/amd64 PostgreSQL 17. Other PostgreSQL versions and architectures need separate validation.
This establishes a working CPU integration. Recall and build-time comparisons remain M4 work.

See [recorded validation evidence](evidence/m3_cpu_2026-10-26.txt).
