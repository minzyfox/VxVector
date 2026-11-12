"""Measure the standalone Vx CPU k-means call without Postgres."""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dims", type=int, default=128)
    parser.add_argument("--leaves", type=int, default=32)
    parser.add_argument("--sampling-factor", type=int, default=256)
    args = parser.parse_args()
    if min(args.rows, args.dims, args.leaves, args.sampling_factor) <= 0:
        parser.error("all sizes must be positive")
    if args.leaves > args.rows:
        parser.error("leaves cannot exceed rows")

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.fspath(project / "scripts"))
    from sampler_reference import table_slots
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = ctypes.CDLL(os.fspath(project / "build" / f"libvxvector_kmeans{suffix}"))
    run = library.vxvector_kmeans_v2
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int32)
    long_ptr = ctypes.POINTER(ctypes.c_int64)
    run.argtypes = ([float_ptr] + [ctypes.c_int32] * 10
                    + [float_ptr, int_ptr, int_ptr, float_ptr, float_ptr, int_ptr,
                       long_ptr, long_ptr])
    run.restype = ctypes.c_int32

    rows, dims, leaves = args.rows, args.dims, args.leaves
    # Four separated groups make the result easy to inspect.
    points = (ctypes.c_float * (rows * dims))()
    for row in range(rows):
        group = row % 4
        for d in range(dims):
            points[row * dims + d] = float(group * 10 + (row * 13 + d * 7) % 11) / 11.0
    nodes = (ctypes.c_float * ((1 + leaves) * dims))()
    parents = (ctypes.c_int32 * (1 + leaves))()
    assignments = (ctypes.c_int32 * rows)()
    current = (ctypes.c_float * (leaves * dims))()
    sums = (ctypes.c_float * (leaves * dims))()
    counts = (ctypes.c_int32 * leaves)()
    samples = min(rows, leaves * args.sampling_factor)
    sample_rows = (ctypes.c_int64 * samples)()
    sample_table = (ctypes.c_int64 * table_slots(samples))()

    start = time.perf_counter()
    status = run(points, rows, dims, 0, leaves, args.sampling_factor,
                 10, 1, 0, 0, 42, nodes, parents, assignments,
                 current, sums, counts, sample_rows, sample_table)
    seconds = time.perf_counter() - start
    if status != 0:
        raise RuntimeError(f"vxvector_kmeans_v2 returned {status}")
    print(f"rows={rows} dims={dims} leaves={leaves} seconds={seconds:.4f} "
          f"rows_per_second={rows / seconds:.1f}")


if __name__ == "__main__":
    main()
