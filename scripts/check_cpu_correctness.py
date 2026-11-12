"""Compare the native Vx library with Faiss using matched samples and starts.

Run with the versions in requirements-correctness.txt. No GPU is used.
"""

import argparse
import ctypes
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import faiss
import numpy as np

from sampler_reference import table_slots, training_rows


PROJECT = Path(__file__).resolve().parents[1]
ATOL = 3e-4
RTOL = 3e-4


def distances(points, centers, metric):
    """Recompute in float64, independently of both native distance kernels."""
    x, c = points.astype('float64'), centers.astype('float64')
    if metric == 0:
        return ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    dot = x @ c.T
    if metric == 1:
        return -dot
    xn, cn = np.linalg.norm(x, axis=1), np.linalg.norm(c, axis=1)
    denom = xn[:, None] * cn[None, :]
    result = 1 - np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    result[(xn[:, None] == 0) & (cn[None, :] == 0)] = 0
    return result


def sample_indices(rows, k, factor, seed):
    return np.array(training_rows(rows, k, factor, seed), dtype='int64')


def normalize(centers):
    norm = np.linalg.norm(centers, axis=1, keepdims=True)
    return np.divide(centers, norm, out=np.zeros_like(centers), where=norm != 0)


def reference(points, k, factor, iterations, restarts, metric, spherical, seed):
    sample = np.ascontiguousarray(points[sample_indices(len(points), k, factor, seed)])
    candidates = []
    for restart in range(restarts):
        positions = (np.arange(k) * len(sample) // k + restart * 104729) % len(sample)
        initial = sample[positions].copy()
        if spherical:
            initial = normalize(initial)
        # One iteration per call prevents Faiss early stopping before Vx's fixed count.
        for _ in range(iterations):
            clustering = faiss.Clustering(points.shape[1], k)
            clustering.niter = 1
            clustering.nredo = 1
            clustering.spherical = spherical
            clustering.min_points_per_centroid = 1
            clustering.max_points_per_centroid = len(sample)
            faiss.copy_array_to_vector(initial.ravel(), clustering.centroids)
            index = (faiss.IndexFlatL2(points.shape[1]) if metric == 0
                     else faiss.IndexFlatIP(points.shape[1]))
            clustering.train(sample, index)
            if clustering.iteration_stats.size() != 1:
                raise AssertionError('Faiss did not run one requested Lloyd iteration')
            # Faiss replaces empty clusters by splitting populated clusters. Vx uses
            # a sampled row. Do not claim strict equivalence when that policy matters.
            splits = sum(clustering.iteration_stats.at(i).nsplit
                         for i in range(clustering.iteration_stats.size()))
            if splits:
                raise AssertionError('Matched Faiss fixture encountered its different empty-cluster policy')
            centers = faiss.vector_to_array(clustering.centroids).reshape(k, -1).copy()
            initial = centers
        objective = float(distances(sample, centers, metric).min(axis=1).sum())
        candidates.append((objective, centers))
    return min(candidates, key=lambda item: item[0])[1], sample


def lloyd_reference(points, k, factor, iterations, restarts, metric, spherical, seed):
    """Independent float64 Lloyd oracle, including VxVector's empty-cluster policy."""
    sample = points[sample_indices(len(points), k, factor, seed)].astype('float64')
    candidates = []
    for restart in range(restarts):
        positions = (np.arange(k) * len(sample) // k + restart * 104729) % len(sample)
        centers = sample[positions].copy()
        if spherical:
            centers = normalize(centers)
        for iteration in range(iterations):
            labels = distances(sample, centers, metric).argmin(axis=1)
            centers = np.stack([sample[labels == c].mean(axis=0) if np.any(labels == c)
                                else sample[(c + iteration) % len(sample)] for c in range(k)])
            if spherical:
                centers = normalize(centers)
        objective = float(distances(sample, centers, metric).min(axis=1).sum())
        candidates.append((objective, centers))
    return min(candidates, key=lambda pair: pair[0])[1], sample


def native_function(library):
    function = library.vxvector_kmeans_v2
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
    lp = ctypes.POINTER(ctypes.c_int64)
    function.argtypes = [fp] + [ctypes.c_int32] * 10 + [fp, ip, ip, fp, fp, ip, lp, lp]
    function.restype = ctypes.c_int32
    return function


def native(function, points, roots, leaves, factor, iterations,
           restarts, metric, spherical, seed):
    rows, dims = points.shape
    nodes = np.zeros((1 + roots + leaves, dims), dtype='float32')
    parents = np.zeros(len(nodes), dtype='int32')
    labels = np.zeros(rows, dtype='int32')
    current, sums = (np.zeros((leaves, dims), dtype='float32') for _ in range(2))
    counts = np.zeros(leaves, dtype='int32')
    samples = min(rows, leaves * factor)
    sample_rows = np.zeros(samples, dtype='int64')
    sample_table = np.zeros(table_slots(samples), dtype='int64')
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
    lp = ctypes.POINTER(ctypes.c_int64)
    status = function(points.ctypes.data_as(fp), rows, dims, roots, leaves,
                      factor, iterations, restarts, metric, int(spherical), seed,
                      nodes.ctypes.data_as(fp), parents.ctypes.data_as(ip),
                      labels.ctypes.data_as(ip), current.ctypes.data_as(fp),
                      sums.ctypes.data_as(fp), counts.ctypes.data_as(ip),
                      sample_rows.ctypes.data_as(lp), sample_table.ctypes.data_as(lp))
    if status != 0:
        raise AssertionError(f'Native call returned {status}')
    return nodes, parents, labels


def match_centers(actual, expected):
    # Fixtures use at most four centroids, so exhaustive matching is inexpensive.
    permutation = min(itertools.permutations(range(len(expected))),
                      key=lambda p: float(((actual - expected[list(p)]) ** 2).sum()))
    np.testing.assert_allclose(actual, expected[list(permutation)], atol=ATOL, rtol=RTOL,
                               err_msg='Centroids differ from reference training')


def check_structure(points, nodes, parents, labels, roots, leaves, metric, spherical):
    if not np.isfinite(nodes).all():
        raise AssertionError('Centroids must remain finite')
    np.testing.assert_array_equal(nodes[0], np.zeros(points.shape[1]))
    np.testing.assert_array_equal(parents[:1 + roots], [-1] + [0] * roots)
    leaf_nodes = nodes[1 + roots:]
    if len(leaf_nodes) != leaves or np.any(labels < 0) or np.any(labels >= leaves):
        raise AssertionError('Invalid leaf count or assignment index')
    costs = distances(points, leaf_nodes, metric)
    np.testing.assert_allclose(costs[np.arange(len(points)), labels], costs.min(axis=1),
                               atol=ATOL, rtol=RTOL, err_msg='Assignments must select nearest leaves')
    if roots:
        if np.any(parents[1 + roots:] < 1) or np.any(parents[1 + roots:] > roots):
            raise AssertionError('Leaf parents must identify root centroids')
        root_costs = distances(leaf_nodes, nodes[1:1 + roots], metric)
        selected = parents[1 + roots:] - 1
        np.testing.assert_allclose(root_costs[np.arange(leaves), selected], root_costs.min(axis=1),
                                   atol=ATOL, rtol=RTOL, err_msg='Leaf parents must be nearest roots')
    else:
        np.testing.assert_array_equal(parents[1:], np.zeros(leaves))
    if spherical:
        norms = np.linalg.norm(nodes[1:], axis=1)
        np.testing.assert_allclose(norms[norms != 0], 1, atol=ATOL, rtol=RTOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    parser.add_argument('--library', type=Path, default=PROJECT / 'build' / f'libvxvector_kmeans{suffix}')
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    library = ctypes.CDLL(str(args.library.resolve()))
    function = native_function(library)
    print(json.dumps(dict(python=platform.python_version(), platform=platform.platform(),
                          numpy=np.__version__, faiss=faiss.__version__, threads=1,
                          library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                          atol=ATOL, rtol=RTOL)), flush=True)
    cases = 0
    for dims, metric, seed, roots in itertools.product((128, 768, 2000, 3072), range(3), (0, 17), (0, 2)):
        # Continuous data avoids empty-cluster policy differences. All cases use
        # exactly the same float32 bytes and initialization in both libraries.
        points = np.random.default_rng(731 + dims).normal(size=(67, dims)).astype('float32')
        spherical = metric != 0
        if metric == 2:
            points = normalize(points)
        factor = 8 if seed == 0 else 32
        iterations, restarts = (1, 1) if seed == 0 else (8, 3)
        nodes, parents, labels = native(function, points, roots, 4, factor,
                                        iterations, restarts, metric, spherical, seed)
        expected, sample = reference(points, 4, factor, iterations, restarts, metric, spherical, seed)
        check_structure(points, nodes, parents, labels, roots, 4, metric, spherical)
        leaves = nodes[1 + roots:]
        match_centers(leaves, expected)
        if roots:
            # Train roots from Vx leaves, preserving the leaves-first hierarchy.
            expected_roots, _ = reference(leaves, roots, factor, iterations, restarts,
                                           metric, spherical, seed % 2147483646 + 1)
            match_centers(nodes[1:1 + roots], expected_roots)
        actual_objective = float(distances(sample, leaves, metric).min(axis=1).sum())
        expected_objective = float(distances(sample, expected, metric).min(axis=1).sum())
        np.testing.assert_allclose(actual_objective, expected_objective, atol=ATOL, rtol=RTOL,
                                   err_msg='Independent sample objectives differ')
        repeated = native(function, points, roots, 4, factor, iterations,
                          restarts, metric, spherical, seed)
        for actual, repeat in zip((nodes, parents, labels), repeated):
            np.testing.assert_array_equal(actual, repeat, err_msg='A repeated call changed the result')
        print(json.dumps(dict(dims=dims, metric=metric, seed=seed, roots=roots,
                              samples=len(sample), factor=factor, iterations=iterations,
                              restarts=restarts, objective=actual_objective,
                              reference_objective=expected_objective,
                              sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest())), flush=True)
        cases += 1
    independent_cases = 0
    for dims, metric, spherical, iterations, restarts, factor in itertools.product(
            (13, 128), range(3), (False, True), (1, 5), (1, 3), (3, 20)):
        points = np.random.default_rng(951 + dims).normal(size=(41, dims))
        points *= np.geomspace(0.1, 10, len(points))[:, None]
        points = points.astype('float32')
        seed = 17
        nodes, parents, labels = native(function, points, 0, 4, factor,
                                        iterations, restarts, metric, spherical, seed)
        expected, sample = lloyd_reference(points, 4, factor, iterations, restarts,
                                            metric, spherical, seed)
        check_structure(points, nodes, parents, labels, 0, 4, metric, spherical)
        match_centers(nodes[1:], expected)
        actual_objective = float(distances(sample, nodes[1:], metric).min(axis=1).sum())
        expected_objective = float(distances(sample, expected, metric).min(axis=1).sum())
        np.testing.assert_allclose(actual_objective, expected_objective, atol=ATOL, rtol=RTOL)
        print(json.dumps(dict(reference='independent-float64-lloyd', dims=dims, metric=metric,
                              spherical=spherical, iterations=iterations, restarts=restarts,
                              factor=factor, seed=seed, objective=actual_objective,
                              reference_objective=expected_objective)), flush=True)
        independent_cases += 1
    # Explicitly separate policies which cannot be equated with Faiss splitting.
    for metric in range(3):
        for points in (np.zeros((8, 128), dtype='float32'),
                       np.tile(np.array([[3., 4.] + [0.] * 126], dtype='float32'), (8, 1))):
            nodes, parents, labels = native(function, points, 2, 4, 8, 8, 2, metric, True, 42)
            check_structure(points, nodes, parents, labels, 2, 4, metric, True)
            expected = normalize(points[:1])[0]
            np.testing.assert_allclose(nodes[1:], np.tile(expected, (6, 1)), atol=ATOL, rtol=RTOL)
    print(json.dumps(dict(matched_faiss_cases=cases, independent_lloyd_cases=independent_cases, empty_cluster_cases=6, result='passed')), flush=True)


if __name__ == '__main__':
    main()
