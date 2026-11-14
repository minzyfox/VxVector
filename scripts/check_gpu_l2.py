"""Step 3: check L2 k-means on the Apple GPU against the CPU kernel and
independent float64 calculations, with evidence that Metal ran every kernel.

The parent process runs each case in a worker process with
VX_DISPATCH_VERBOSE=1 and reads the Metal runtime's log. The policy that these
gates check is in docs/step3_metal_l2.md.

Run with the arm64 Python of build/correctness-venv after scripts/build.sh and
scripts/build_gpu.sh.
"""

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CPU_LIBRARY = PROJECT / "build/libvxvector_kmeans.dylib"
GPU_LIBRARY = PROJECT / "build/gpu/libvxvector_gpu.dylib"
OK, OVERFLOW, INVALID, UNAVAILABLE = 0, 1, 2, 3

# name, n, dims, k, iterations, input
CASES = [
    ("blobs", 4096, 32, 16, 10, "blobs"),
    ("wide_rows", 1000, 300, 10, 8, "blobs"),
    ("one_past_a_block", 257, 3, 7, 6, "blobs"),
    ("uniform", 2048, 17, 33, 12, "uniform"),
    ("empty_clusters", 64, 4, 4, 5, "two_points"),
    ("empty_replacement", 64, 4, 4, 5, "duplicate_seeds"),
    ("one_center", 100, 8, 1, 4, "blobs"),
    ("every_sample_a_center", 9, 5, 9, 3, "blobs"),
    ("no_iterations", 300, 6, 5, 0, "blobs"),
    ("sum_overflow", 4, 2, 1, 2, "huge"),
    ("sift_8192", 8192, 128, 64, 10, "sift"),
]
SIFT = PROJECT / "build/datasets/sift-128-euclidean.hdf5"


def make_input(kind, n, dims, seed):
    rng = np.random.default_rng(seed)
    if kind == "blobs":
        centers = rng.normal(0.0, 10.0, size=(max(2, n // 64), dims))
        rows = centers[rng.integers(0, len(centers), size=n)]
        return (rows + rng.normal(0.0, 1.0, size=(n, dims))).astype(np.float32)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n, dims)).astype(np.float32)
    if kind == "two_points":
        rows = np.zeros((n, dims), dtype=np.float32)
        rows[n // 2:] = 5.0
        return rows
    if kind == "sift":
        import h5py
        with h5py.File(SIFT, "r") as dataset:
            return np.asarray(dataset["train"][:n], dtype=np.float32)
    if kind == "duplicate_seeds":
        # Distinct rows, except that the four seeds (rows 0, 16, 32 and 48)
        # are equal. Three centers start empty, and each replacement row differs.
        rows = np.repeat(np.arange(n, dtype=np.float32)[:, None], dims, axis=1)
        rows[::16] = 0.0
        return rows
    if kind == "huge":
        return np.full((n, dims), 3.0e38, dtype=np.float32)
    raise ValueError(kind)


def pointer(array, kind):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def run_gpu(library, samples, k, iterations):
    n, dims = samples.shape
    centroids = np.zeros((k, dims), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    counts = np.zeros(k, dtype=np.int32)
    status = library.vxvector_gpu_l2_v1(
        pointer(samples, ctypes.c_float), n, dims, k, iterations,
        pointer(centroids, ctypes.c_float), pointer(labels, ctypes.c_int32),
        pointer(counts, ctypes.c_int32))
    return status, centroids, labels, counts


def run_cpu(library, samples, k, iterations):
    """One restart of the CPU kernel on every sample in order."""
    n, dims = samples.shape
    factor = -(-n // k)
    nodes = np.zeros((1 + k, dims), dtype=np.float32)
    parents = np.zeros(1 + k, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int32)
    current = np.zeros((k, dims), dtype=np.float32)
    sums = np.zeros((k, dims), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int32)
    chosen = np.zeros(n, dtype=np.int64)
    slots = 1
    while slots < 2 * n:
        slots *= 2
    table = np.zeros(slots, dtype=np.int64)
    status = library.vxvector_kmeans_v2(
        pointer(samples, ctypes.c_float), n, dims, 0, k, factor, iterations,
        1, 0, 0, 0, pointer(nodes, ctypes.c_float),
        pointer(parents, ctypes.c_int32), pointer(labels, ctypes.c_int32),
        pointer(current, ctypes.c_float), pointer(sums, ctypes.c_float),
        pointer(counts, ctypes.c_int32), pointer(chosen, ctypes.c_int64),
        pointer(table, ctypes.c_int64))
    return status, nodes[1:].copy(), labels


def independent_checks(samples, centroids, labels):
    """float64 checks that use neither Vx kernel: the objective, and labels that
    are not the nearest center beyond float32 rounding."""
    wide = samples.astype(np.float64)
    centers = centroids.astype(np.float64)
    distances = ((wide[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    best = distances.min(axis=1)
    chosen = distances[np.arange(len(samples)), labels]
    # A label may differ from the float64 nearest center only at a near-tie
    # that float32 cannot separate.
    slack = 1e-5 * np.maximum(best, 1e-30) + 1e-30
    not_nearest = int((chosen - best > slack).sum())
    objective = float(chosen.sum())
    return objective, not_nearest


def nearest(samples, centers):
    """float64 squared L2 distances, in chunks. argmin keeps the lowest index."""
    centers = centers.astype(np.float64)
    labels = np.empty(len(samples), dtype=np.int32)
    for start in range(0, len(samples), 1024):
        wide = samples[start:start + 1024].astype(np.float64)
        distances = ((wide[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + 1024] = distances.argmin(axis=1)
    return labels


def reference_lloyd(samples, k, iterations):
    """The policy of docs/step3_metal_l2.md in NumPy, independent of both Vx
    kernels: seeds at c * n // k, float64 assignment, sequential float32 sums
    in sample order (cumsum adds in order), and empty-cluster replacement."""
    n = len(samples)
    centers = samples[[c * n // k for c in range(k)]].copy()
    for iteration in range(iterations):
        labels = nearest(samples, centers)
        for c in range(k):
            rows = samples[labels == c]
            if len(rows) == 0:
                centers[c] = samples[(c + iteration) % n]
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                running = np.cumsum(rows, axis=0, dtype=np.float32)
            if not np.isfinite(running).all():
                return None, None
            centers[c] = running[-1] / np.float32(len(rows))
    return centers, nearest(samples, centers)


def worker(args):
    gpu = ctypes.CDLL(os.fspath(GPU_LIBRARY))
    gpu.vxvector_gpu_l2_v1.restype = ctypes.c_int32
    if args.unavailable:
        samples = make_input("blobs", 64, 4, 1)
        status, *_ = run_gpu(gpu, samples, 4, 2)
        print(json.dumps(dict(case="metal_disabled", status=status)), flush=True)
        return
    cpu = ctypes.CDLL(os.fspath(CPU_LIBRARY))
    cpu.vxvector_kmeans_v2.restype = ctypes.c_int32
    for name, n, dims, k, iterations, kind in CASES:
        if args.case and name != args.case:
            continue
        if kind == "sift" and not SIFT.exists():
            print(json.dumps(dict(case=name, skipped="download the SIFT dataset as in README.md")), flush=True)
            continue
        samples = make_input(kind, n, dims, seed=n * 31 + dims)
        print(f"[case] {name}", file=sys.stderr, flush=True)
        status, centroids, labels, counts = run_gpu(gpu, samples, k, iterations)
        cpu_status, cpu_centroids, cpu_labels = run_cpu(cpu, samples, k, iterations)
        result = dict(case=name, n=n, dims=dims, k=k, iterations=iterations,
                      status=status, cpu_status=cpu_status)
        if status == OK and cpu_status == 0:
            objective, not_nearest = independent_checks(samples, centroids, labels)
            cpu_objective, _ = independent_checks(samples, cpu_centroids, cpu_labels)
            reference, reference_labels = reference_lloyd(samples, k, iterations)
            result.update(
                same_bits_as_reference=reference is not None and bool(np.array_equal(
                    centroids.view(np.uint32), reference.view(np.uint32))),
                labels_differ_from_reference=None if reference is None
                else int((labels != reference_labels).sum()),
                same_centroid_bits=bool(np.array_equal(
                    centroids.view(np.uint32), cpu_centroids.view(np.uint32))),
                label_differences=int((labels != cpu_labels).sum()),
                max_centroid_difference=float(np.abs(
                    centroids.astype(np.float64) - cpu_centroids).max()),
                counts_add_up=bool(counts.sum() == n),
                objective=objective, cpu_objective=cpu_objective,
                not_nearest=not_nearest,
                finite=bool(np.isfinite(centroids).all()))
        print(json.dumps(result), flush=True)


def kernel_launches(iterations):
    # The seed, then an assignment and an update per iteration, then the final assignment.
    return 2 + 2 * iterations


def gate(row):
    problems = []
    if "skipped" in row:
        return problems
    # The CPU returns 1 for invalid arguments and for a sum overflow.
    expected = {0: (OK,), 1: (OVERFLOW, INVALID)}.get(row["cpu_status"], ())
    if row["status"] not in expected:
        problems.append("status differs from the CPU")
    if row["status"] == OK and row["cpu_status"] == 0:
        if not row["finite"] or not row["counts_add_up"]:
            problems.append("non-finite centroids or counts that do not add up")
        if row["not_nearest"]:
            problems.append(f"{row['not_nearest']} labels are not the nearest center")
        # These inputs have no float32 near-ties, and an exact tie goes to the
        # lower index in float32 and float64 alike, so every bit must match.
        # A matching objective is not enough: equal-cost centroids in another
        # order would hide a wrong tie or empty-cluster rule.
        if not row["same_centroid_bits"] or row["label_differences"]:
            problems.append(f"differs from the CPU kernel: {row['label_differences']} labels, "
                            f"centroids up to {row['max_centroid_difference']:.3g} apart")
        if not row["same_bits_as_reference"] or row["labels_differ_from_reference"]:
            problems.append("differs from the NumPy reference")
        relative = abs(row["objective"] - row["cpu_objective"]) / max(row["cpu_objective"], 1e-30)
        if relative > 1e-12:
            problems.append(f"objective differs from the CPU by {relative:.3g}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--unavailable", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--log", type=Path, help="write the Metal dispatch log here")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    environment = dict(os.environ, VX_DISPATCH_VERBOSE="1")
    run = subprocess.run([sys.executable, "-B", __file__, "--worker"],
                         capture_output=True, text=True, env=environment)
    if args.log:
        args.log.write_text(run.stderr)
    if run.returncode != 0:
        sys.exit(f"the worker failed:\n{run.stdout}\n{run.stderr}")
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    failures = 0
    # Split the log by case and count the Metal launches of each.
    sections = re.split(r"^\[case\] (\S+)$", run.stderr, flags=re.M)
    logs = dict(zip(sections[1::2], sections[2::2]))
    for row in rows:
        log = logs.get(row["case"], "")
        launches = len(re.findall(r"^\[Vx Metal\] vx_npu_kernel_\d+: grid", log, flags=re.M))
        row["metal_launches"] = launches
        problems = gate(row)
        if row["status"] == OK and "skipped" not in row and launches != kernel_launches(row["iterations"]):
            problems.append(f"{launches} Metal launches, expected {kernel_launches(row['iterations'])}")
        if "is copied" in log or "Falling back" in log:
            problems.append("a tensor was copied, or a kernel ran on the CPU")
        row["result"] = "failed: " + "; ".join(problems) if problems else "passed"
        failures += bool(problems)
        print(json.dumps(row), flush=True)

    # With Metal turned off, the function must refuse rather than run on the CPU.
    off = subprocess.run([sys.executable, "-B", __file__, "--worker", "--unavailable"],
                         capture_output=True, text=True,
                         env=dict(environment, VX_METAL="0"))
    disabled = json.loads(off.stdout.splitlines()[-1]) if off.returncode == 0 else dict(status=None)
    disabled_ok = disabled["status"] == UNAVAILABLE and "Falling back" not in off.stderr
    print(json.dumps(dict(case="metal_disabled", status=disabled["status"],
                          result="passed" if disabled_ok else "failed")), flush=True)
    failures += not disabled_ok

    print(json.dumps(dict(result="passed" if failures == 0 else "failed",
                          cases=len(rows) + 1, failures=failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
