# Milestone 1: standalone k-means

Checked on 2026-10-22 on Apple M4 Pro with the local Vx compiler.

## Implemented

`src/kmeans.vx` exports `vxvector_kmeans_v1` through a C ABI. The caller supplies
row-major float32 points, centroid output, assignments, and scratch buffers.
The header documents buffer sizes and status codes. No Postgres dependency is
needed. The implementation uses squared L2, negative inner product, or cosine
distance. It supports centroid normalization, distinct seeded sampling, multiple
restarts, and sample batches. A two-level tree trains leaves first and trains
roots from the leaf centroids. Empty clusters use a sampled vector. Invalid
shapes and nonfinite input return status 1.

The C ABI was built as a macOS shared library. Nine Python standard-library
checks passed. They include an exhaustive two-cluster CPU reference, assignment
agreement, repeatability, batch equivalence, two-level parent IDs, duplicate
vectors, zero cosine vectors, invalid input, and dimensions 2000 and 3072.
`scripts/benchmark.py` gives a repeatable CPU timing command. This is a CPU
baseline, not a claim about GPU speed.

## GPU compile result and language gap

`src/gpu_lloyd.vx` accepts device-resident samples and initialized centroids.
It expresses assignment and centroid recomputation without atomics by scanning
each cluster's assignments. `VX_CHECK_GPU=1 ./scripts/build.sh` generated A100
code with the local Vx compiler. No NVIDIA device ran this code.

The first raw-pointer version reported W1033 because the compiler could not
prove loop independence around unsafe writes and a helper call. The current
version uses placed tensor views and compiles without that warning. Its
generated code still has no `launch=` metadata for the assignment and
reduction loops. A known parallel Vx fixture does produce `launch=4096` and
uses the GPU thread ID, so the current prototype does not establish useful
parallel execution. This makes it unsuitable for the performance target.
An isolated probe produced a parallel launch for `view[lane][0]` but not for
`view[base + lane][0]`. The tiled kernel needs the latter form. A subview with
its own zero-based index may solve this, but creating that view from a placed
pointer also needs a supported pointer-offset operation. The first upstream
example should cover this exact tiled write. The current GPU
prototype also needs a host data transfer path, initialization, sampling,
restarts, and runtime tests.

## Verification commands

```sh
./scripts/build.sh
python3 -m unittest discover -s tests -v
python3 scripts/benchmark.py --rows 1000 --dims 16 --leaves 8
VX_CHECK_GPU=1 ./scripts/build.sh
```

The small CPU benchmark used 1000 rows, 16 dimensions, and 8 leaves. It took
about 0.014 seconds on this Mac. This one run is only a local smoke check.
