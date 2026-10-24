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
python3 -m unittest discover -s tests -v
python3 scripts/benchmark.py --rows 10000 --dims 128 --leaves 32
VX_CHECK_GPU=1 ./scripts/build.sh
```

The build creates `build/libvextor_kmeans.dylib` on macOS or
`build/libvextor_kmeans.so` on Linux. `build/` is ignored by Git.

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

This API does not stream input from disk or transfer data to a GPU. The Vx
algorithm and its small-data reference checks establish a CPU baseline.

`src/gpu_lloyd.vx` is a separate GPU prototype that accepts device-resident
samples and initialized centroids. `VX_CHECK_GPU=1` checks that Vx generates
A100 code without requiring GPU hardware. The generated code has no parallel
launch metadata for the assignment and reduction loops. This path is untested
on an NVIDIA device and is not a performance implementation.
