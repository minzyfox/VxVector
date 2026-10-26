# Milestone 2: CPU host data path

Implemented and checked on 2026-10-26 on Apple M4 Pro with the local Vx compiler.
GPU acceleration will come later. This milestone requires neither PostgreSQL nor CUDA.

## Implemented API

`include/vextor_data_path.h` defines the host adapter alongside the existing Vx array ABI.
`src/data_path.c` implements source reading, sampling, allocation checks, and timing.
The clustering algorithm remains in `src/kmeans.vx`.

- `vextor_plan_cpu_v1` validates a request and calculates all adapter buffer sizes without allocating memory.
- `vextor_run_cpu_v1` reads the source, collects a sample, and calls `vextor_kmeans_v1` once.
- `vextor_free_cpu_result_v1` releases the result and clears its pointers. Releasing an empty result is safe.

The reader returns one dense float32 vector, EOF, or an error per call.
The adapter packs rows into contiguous row-major batches, capped by `batch_size` and the source count.
Reader storage must remain valid until the next reader call. The adapter copies each row before that call.
The result owns the sampled vectors, their source positions, centroids, parent IDs, and sample assignments.
Callers must release an existing result before reusing its destination.
Failures leave an empty result and return a status with an optional bounded error message.

## Input validation and source contract

The adapter accepts only dense `vector` rows represented as float32 components.
It rejects null vectors, mixed dimensions, unsupported row types, and nonfinite components before calling Vx.
It validates every source row, including rows outside the sample.
It also validates clustering options, including the distance code and spherical flag.
`halfvec`, `sparsevec`, and `bit` remain separate future work.

The caller supplies an exact 64-bit source count, capped at `i64::MAX`.
The adapter rejects early EOF, additional rows, and reader errors.
The reader must use a stable row order and the same snapshot for counting and scanning.
Count checks cannot detect reordered or replaced rows when the count stays unchanged.
M3 must enforce the snapshot contract in the PostgreSQL reader.
M3 must also validate the declared SQL column type before scanning.

Reader callbacks must return errors without unwinding or jumping across the C call.
A future database reader must translate cancellation into a reader error.
The separate worker and Postgres error conversion remain M3 work.

## Bounded sampling and training

Let `N` be source rows, `L` leaves, `F` the sampling factor, and `S = min(N, L * F)`.
The adapter retains `S` sampled vectors and one read batch, rather than `N` vectors.
It rejects `S > i32::MAX` because the Vx array ABI uses an `i32` sampled row count.

Floyd's algorithm selects distinct source positions with storage proportional to `S`.
SplitMix64 and rejection sampling define repeatable random draws without modulo bias.
An open-addressed hash table records selected positions.
An in-place heapsort orders scan positions without an additional allocation.
Each position retains its destination, so sampled vectors remain in seeded selection order.
Batch size does not affect the sample or clustering results.

The adapter calls the existing CPU library with the collected sample.
That library applies its own permutation for initialization and training order.
Adapter output therefore matches a direct call on the collected sample.
It need not match a direct call on the complete source table.
The array API's `batch_size` still batches computation rather than bounding its input storage.

## Memory budget

Planning checks every buffer size and their sum against `SIZE_MAX` and `PTRDIFF_MAX` before allocation.
The adapter rejects budget excess before allocation, source reads, or Vx calls.
Allocation failure returns a recoverable status and releases all partial allocations.

For `D` dimensions, `B = min(N, batch_size)`, and `C = 1 + root_count + L`, buffers contain:

| Buffer | Bytes |
| --- | ---: |
| Sample vectors | `4 * S * D` |
| Read batch | `4 * B * D` |
| Centroid output | `4 * C * D` |
| Parents | `4 * C` |
| Sample assignments | `4 * S` |
| Current centroids and sums | `8 * L * D` |
| Cluster counts | `4 * L` |
| Returned source positions | `8 * S` |
| Sorted positions and destinations | `16 * S` |
| Sampling hash table | `8 * H` |

`H` is the smallest power of two at least `2 * S`.
The implementation uses `sizeof` when calculating actual allocation sizes.
The budget covers all adapter heap buffers, which coexist during execution.
It excludes allocator overhead, caller-owned reader storage, stack storage, and Vx runtime memory.
M3 must account for database reader and worker transport storage separately.
The result retains sample buffers for inspection until the caller releases it.

## Verification

```sh
./scripts/build.sh
python3 -B -m unittest discover -s tests -v
./scripts/check_data_path.sh
./build/benchmark_data_path 1000 16 8 40 64
./build/benchmark_data_path 100000 16 8 40 64
```

The Python suite passes 18 tests: nine standalone checks and nine adapter checks.
Adapter checks compare direct Vx calls for every metric and flat/two-level trees.
They also cover deterministic distinct sampling, sample-to-source agreement, batch boundaries, invalid rows, count changes, overflow, and budget rejection.

The native harness tracks actual allocation sizes and injects failure into each allocation.
It checks cleanup, rejection before reader or Vx calls, and storage independent of total source rows.
It also checks Vx error returns and nonfinite centroid rejection.
Address and undefined-behavior sanitizer checks pass.

The benchmark generates one row at a time and reports read/validation, sampling, clustering, and total elapsed times.
These smoke runs used 16 dimensions, eight leaves, sampling factor 40, and batches of 64 rows.
Both retained 320 sampled rows and requested 43,396 bytes of adapter buffers.
The 1,000-row run took about 0.0039 seconds; the 100,000-row run took about 0.0086 seconds.
These are single local smoke measurements, not database or GPU performance claims.
Raw output and the Vx compiler revision are in [M2 smoke evidence](evidence/m2_cpu_2026-10-26.txt).

## Remaining integration work

M2 completes the reusable CPU host data path.
M3 supplies the PostgreSQL reader, snapshot enforcement, worker process, centroid-table writes, and VectorChord index creation.
Database reads, worker communication, centroid writes, and index builds need timings through that integration.
GPU transfers, pinned memory, double buffering, and hardware measurements remain in M5.
