# Milestone 2: host data path

This is the data path design that can be checked without a GPU or PostgreSQL.
The database reader, GPU transfers, and timing measurements remain implementation work.

## Input contract

The first database version accepts dense pgvector `vector` values. Each component
becomes an `f32`. Reject null vectors, nonfinite components, mixed dimensions,
and unsupported column types before calling Vx. `halfvec`, `bit`, and
`sparsevec` need separate conversion and compatibility work.

A host batch owns `rows * dims` contiguous `f32` values in row-major order.
Its byte size is `4 * rows * dims`. Use `batch_size` as a maximum row count,
and check every size multiplication before allocation. The database reader
fills the batch, and the worker copies selected rows into its sample matrix.
Release or reuse the read batch after those rows are copied.
Pinned memory is a later GPU-specific choice; ordinary host memory is enough
for the CPU path.

## Bounded centroid training

Let `N` be the number of source rows, `L` the leaf count, `F` the sampling
factor, and `S = min(N, L * F)`. Keep at most `S` sampled vectors in memory,
plus one database read batch. The present `vextor_kmeans_v1` function needs a
contiguous input matrix, but it does not need the complete table to train
centroids. Its assignments buffer can cover those `S` sampled rows. The
centroid table does not use assignments for all `N` source rows.

Use a stable source row order and the same database snapshot for an exact row
count and the sample scan. Compute `S` distinct row positions from the seed,
then collect those positions during the scan. Store the sampled vectors in
sample order. This retains a repeatable sample without materializing the
whole table. The existing Vx function will permute that sample again for
initialization. Its centroids therefore need not be byte-identical to a call
that receives every source row. Compare quality and recall with pgpu after a
GPU host becomes available.

For the first CPU integration, collect the sample and call the existing Vx
function once. Do not claim that its current `batch_size` argument bounds
input memory: it only divides the training loop into batches. A later worker
API can accept batches directly and overlap reads with GPU work.

The worker must reject a request before allocation if the checked total for
the sample, read batch, centroid output, scratch buffers, and index arrays
exceeds its configured memory budget. Use a wider row counter during the
database scan. The current Vx ABI uses `i32` for its sampled row count, so
reject `S > i32::MAX` until that ABI changes.

## Checks before the database integration

- Compare centroids from a collected sample with a call on the same sample
  supplied directly to Vx.
- Check that sampling is repeatable and produces distinct source row positions.
- Check batches of one row and batches larger than the table.
- Check allocation overflow and budget rejection before any Vx call.
- Check that a table larger than the sample does not allocate `N * dims` values.

GPU transfer overlap, pinned memory, and timing for each database stage need
the PostgreSQL wrapper and an NVIDIA host. Those checks remain open.
