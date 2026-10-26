# Vextor

Vextor is a Vx k-means implementation for VectorChord's external index build.
The current standalone library trains centroids over row-major float32 arrays.
It does not require Postgres. The current implementation runs on the CPU.

## Build and check

The build script expects the Vx checkout at `../../Vx` relative to this repo.
Set `VX_ROOT` to use another checkout. Build Vx first, including
`libvx_std_core`, and ensure its `config.local` exists.

```sh
./scripts/build.sh
python3 -B -m unittest discover -s tests -v
./scripts/check_data_path.sh
./build/benchmark_data_path 100000 16 8 40 64
python3 scripts/benchmark.py --rows 10000 --dims 128 --leaves 32
VX_CHECK_GPU=1 ./scripts/build.sh
```

The build creates `build/libvextor_kmeans.dylib` on macOS or
`build/libvextor_kmeans.so` on Linux. `build/` is ignored by Git.

## PostgreSQL development image

The [Docker setup](docs/postgres.md) supplies PostgreSQL 17, pgvector, VectorChord,
and C/Rust build tools in a separate development database on port 15432.
The pgrx extension and separate CPU worker now train centroids and create
VectorChord indexes through `vextor.create_vector_index(...)`.
See [M3 build steps, SQL API, and checks](docs/milestone_3_cpu_integration.md).
GPU acceleration comes later.

## Library API

[`include/vextor_kmeans.h`](include/vextor_kmeans.h) defines the C ABI.
The caller supplies the input, output, and scratch arrays. The function returns
zero on success and one for invalid arguments. It makes one zero-vector root.
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
PostgreSQL readers, worker isolation, and index creation are M3 work.
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
