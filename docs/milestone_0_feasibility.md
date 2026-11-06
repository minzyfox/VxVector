# Milestone 0: feasibility and reference contract

Checked on 2026-10-22. The pgpu source snapshot was commit
`b39cc5e3fa49efb673c524d722ea7c812228bc7e` (version 2.1.0).

## Decisions

1. **Use Vx object output and a C ABI.** A top-level, non-generic `pub fn` keeps
   its name and uses the platform C calling convention. `vxc --action emit-obj`
   produces an object that can be linked into a shared library. A C program called
   exported Vx functions successfully on macOS arm64. A scalar probe returned
   42 for input 41. A pointer probe wrote 42.0 and returned status 0. No fallback
   ABI is needed for this basic path. A Linux and CUDA link
   test remains necessary before integration.
2. **Keep GPU execution outside the Postgres backend.** The current Vx CUDA
   runtime contains `abort()` paths, and Vx assertions lower to `abort()` on the
   host. A status return cannot catch process termination. A pgrx function should
   read database rows and send float32 batches to a separate worker process. The
   worker should load the Vx library and return centroids or a structured error.
   A worker crash becomes a Postgres error without terminating the backend.
3. **Build one package per dispatch backend for now.** Vx `build.rs` chooses one
   dispatch library when Vx is built. A Linux CUDA build needs the CUDA toolkit.
   A build without it selects a CPU dispatch library. `VX_DISPATCH_LIB` can override
   the selected library, but it is not a tested universal package. Start with a
   Linux CUDA package for the GPU run and a separate CPU package for local tests.
4. **Target PostgreSQL 17 for the first integration.** It is pgpu's default feature
   and setup-script default. Extend the version range after the first working run.

## ABI and failure contract

The worker owns the Vx library and calls only exported C ABI functions. The
initial k-means entry point should accept primitive scalars and pointers to
contiguous float32 buffers. The caller owns input and output buffers. Vx must not
retain pointers after the call. Use a versioned ABI and explicit lengths.

For expected failures, return a status code and write a bounded error message
into a caller-owned buffer. Proposed codes are `OK`, `INVALID_INPUT`,
`BACKEND_UNAVAILABLE`, `OUT_OF_MEMORY`, `CANCELLED`, and `INTERNAL_ERROR`.
Validate lengths, dimensions, options, and output capacity before GPU work.
Do not use Vx `assert`, `panic`, or `abort` for input validation. No exception,
panic, or Postgres `longjmp` may cross the C ABI. The worker catches failures it
can recover from. The pgrx function treats worker EOF, signal exit, or timeout
as an error and rolls back its transaction. Query cancellation must stop the
worker and discard partial output. The centroid table and index should be
created within the same Postgres transaction.

This contract does not claim that the current Vx CUDA runtime is recoverable.
The worker process is required until its abort paths are removed or contained.

### Reproduce the ABI probe on this Mac

The probe used `/Users/gagan/dev/Vx/target/debug/vxc` after `source config.local`:

```vx
pub fn add_one(x: i32) -> i32 {
  return x + 1;
}
```

`vxc probe.vx --action emit-obj -o probe.o` produced an object with an exported
`add_one` symbol. `clang -dynamiclib probe.o -o libprobe.dylib` linked it. A C
caller used `dlopen`, `dlsym("add_one")`, and called it with 41; it got 42.
The source had no `main` and needed no runtime library for this scalar probe.
Another exported function accepted `*const f32` and `*mut f32`, wrote 42.0
from an input of 41.0, and returned status 0. A C caller used matching `float *`
arguments. These probes prove scalar and pointer export and shared-library
linking on macOS arm64. They do not prove CUDA linking or error handling.

## VectorChord handoff contract

VectorChord's external build consumes a table with these columns, in order:

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | `integer` | Unique centroid node ID. |
| `parent` | `integer` | Parent node ID. The root has `NULL`. |
| `vector` | pgvector `vector` | Centroid coordinates. |

The nodes form a tree whose leaves have the same depth. pgpu currently creates
`<source_table>_centroids` in the source schema as
`(id INT, parent INT, vector vector(<dimensions>))`. It writes an all-zero root
as `id = 0, parent = NULL`. Flat centroids have parent `0`. For two levels, root
centroids have parent `0`, and leaves name their root centroid's ID. pgpu assigns
IDs itself rather than relying on insertion order. Its table DDL does not declare
the constraints that VectorChord documents, so VxVector should declare `id` unique
and the non-null columns explicitly after validating compatibility.

The public VectorChord page has a contradiction: its example inserts a root
with `parent = NULL`, but its format text says `parent` cannot be null. pgpu's
root also uses `NULL`. Treat the root as the exception and verify against a
live VectorChord build before freezing the schema.

pgpu's index SQL uses `USING vchordrq`, a metric opclass such as
`vector_l2_ops`, and options equivalent to:

```toml
residual_quantization = false
build.pin = 2
[build.external]
table = 'public.example_centroids'
```

The source currently hard-codes the indexed column name `embedding` in its
`CREATE INDEX` statement despite accepting `column_name`. VxVector should use
the supplied column name and test a column with another name. `skip_index_build`
stops after writing the centroid table. The 2.1.0 source also adds a
`random_sampling` argument with default `true` that the README's function
signature does not list. Keep VxVector's public API choice explicit when the
Postgres wrapper is designed.

Sources: [pgpu centroid writer](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/vectorchord-indexing/src/centroids_table.rs),
[pgpu index statement](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/vectorchord-indexing/src/vectorchord_index.rs),
[pgpu function](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/src/vectorchord_indexing.rs),
and [VectorChord external build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html).

## Development environment and reference run

The local machine is an Apple M4 Pro running macOS. It has the Vx toolchain,
but no `psql`, `pg_config`, `nvcc`, or `nvidia-smi` on `PATH`. No NVIDIA GPU
environment is available yet. Therefore Postgres 17, pgvector, VectorChord,
and pgpu were not installed together, and no pgpu GPU reference run was made.
The absence of a GPU does not block writing and testing the standalone k-means
API. Assume the target CUDA GPU is present for the first GPU implementation.
Run the hardware checks when a suitable host becomes available.

pgpu does not appear to test device presence before clustering. Its
`pgpu.gpu_acceleration` setting defaults to `enable`. The clustering code calls
`Resources::new()` and device transfers with `expect`, so a returned CUDA or
cuVS error would cause a Rust panic. pgrx normally translates guarded Rust
panics into SQL errors. This is a source-based expectation, not an observed
no-GPU run. Missing CUDA libraries may prevent extension loading earlier, and
a native abort is not caught by pgrx. Do not promise one exact error message.

On a Linux NVIDIA host, use PostgreSQL 17, pgvector, VectorChord, a CUDA toolkit,
the NVIDIA driver, and the Vx toolchain. pgpu 2.1.0 pins pgrx 0.16.1 and cuVS
25.12. Its README calls for CUDA 13.1, while its setup script defaults to the
package suffix `13-0`; verify a compatible toolkit and cuVS environment before
building. The script enables `shared_preload_libraries = 'vchord'`. Record exact
versions, GPU model, driver, and Vx commit in the run log.

The first reference run should:

1. Start PostgreSQL 17 with `vector`, `vchord`, and `pgpu` installed.
2. Create the three extensions and a small table with a non-null `vector(8)`
   column named `embedding`. Insert at least 640 diverse rows.
3. Call `pgpu.create_vector_index_on_gpu` with `lists => ARRAY[16]`,
   `sampling_factor => 40`, `batch_size => 640`, `distance_operator => 'l2'`,
   and `skip_index_build => true`. Save the generated table schema and rows.
4. Run again with `skip_index_build => false`. Save the exact index definition
   from `pg_get_indexdef`, the table row count, and an `EXPLAIN (ANALYZE, BUFFERS)`
   for a nearest-neighbor query using `<->` and `LIMIT`.
5. Repeat with `lists => ARRAY[4, 16]`. Check root IDs, parent IDs, leaf depth,
   and that VectorChord accepts the table.

The reference output belongs under a dated benchmark or evidence directory.
Do not treat source inspection as a substitute for this run.

## Deferred hardware validation

- Link and call an exported Vx function on the Linux CUDA host. Inspect linked
  runtime libraries and verify that missing GPU, OOM, and worker crash become
  Postgres errors without terminating the backend.
- Complete the pgpu reference run and save centroid rows, index DDL, and
  `EXPLAIN` output.
- Check the live VectorChord version's treatment of a null root parent and
  declared table constraints.

These checks are required before claiming that the Postgres integration works.
They do not block Milestone 1 implementation.

Vx evidence: [ABI policy](/Users/gagan/dev/Vx/docs/lang/abi.md),
[GPU backend plan](/Users/gagan/dev/Vx/docs/gpu_backends.md), and
[`build.rs`](/Users/gagan/dev/Vx/build.rs). pgpu setup evidence:
[README](https://github.com/EnterpriseDB/pgpu/tree/b39cc5e3fa49efb673c524d722ea7c812228bc7e#building-and-running),
[setup script](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/scripts/setup_build.sh),
[GPU setting](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/vectorchord-indexing/src/guc.rs),
[clustering path](https://github.com/EnterpriseDB/pgpu/blob/b39cc5e3fa49efb673c524d722ea7c812228bc7e/vectorchord-indexing/src/clustering_gpu_impl.rs),
and [pgrx panic handling](https://docs.rs/pgrx/latest/pgrx/attr.pg_guard.html).
