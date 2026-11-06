"""Check the Vx C API against small, independently calculated answers."""

import ctypes
import itertools
import math
import os
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
LIBRARY = PROJECT / "build" / f"libvxvector_kmeans{SUFFIX}"


def squared_distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def best_two_cluster_cost(points):
    """Enumerate partitions, independent of the Vx Lloyd loop."""
    best = math.inf
    for labels in itertools.product((0, 1), repeat=len(points) - 1):
        groups = [[points[0]], []]
        for point, label in zip(points[1:], labels):
            groups[label].append(point)
        if not groups[1]:
            continue
        means = [
            [sum(point[d] for point in group) / len(group)
             for d in range(len(points[0]))]
            for group in groups
        ]
        cost = sum(squared_distance(point, means[label])
                   for label, group in enumerate(groups) for point in group)
        best = min(best, cost)
    return best


class KmeansTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LIBRARY.exists():
            raise RuntimeError("build the library with scripts/build.sh first")
        cls.library = ctypes.CDLL(os.fspath(LIBRARY))
        cls.run_function = cls.library.vxvector_kmeans_v1
        float_ptr = ctypes.POINTER(ctypes.c_float)
        int_ptr = ctypes.POINTER(ctypes.c_int32)
        cls.run_function.argtypes = (
            [float_ptr] + [ctypes.c_int32] * 11
            + [float_ptr, int_ptr, int_ptr, float_ptr, float_ptr, int_ptr]
        )
        cls.run_function.restype = ctypes.c_int32

    def run_kmeans(self, points, leaves, roots=0, sampling_factor=256,
                   batch_size=100000, iterations=10, restarts=1, metric=0,
                   spherical=False, seed=42):
        rows = len(points)
        dims = len(points[0])
        flat = [value for point in points for value in point]
        point_buffer = (ctypes.c_float * len(flat))(*flat)
        node_count = 1 + roots + leaves
        nodes = (ctypes.c_float * (node_count * dims))()
        parents = (ctypes.c_int32 * node_count)()
        assignments = (ctypes.c_int32 * rows)()
        current = (ctypes.c_float * (leaves * dims))()
        sums = (ctypes.c_float * (leaves * dims))()
        counts = (ctypes.c_int32 * leaves)()
        status = self.run_function(
            point_buffer, rows, dims, roots, leaves, sampling_factor,
            batch_size, iterations, restarts, metric, int(spherical), seed,
            nodes, parents, assignments, current, sums, counts,
        )
        vectors = [list(nodes[i * dims:(i + 1) * dims])
                   for i in range(node_count)]
        return status, vectors, list(parents), list(assignments)

    def test_l2_centroids_match_exhaustive_reference(self):
        points = [[0, 0], [0, 1], [10, 10], [10, 11]]
        status, nodes, parents, labels = self.run_kmeans(
            points, leaves=2, batch_size=2, restarts=3
        )
        self.assertEqual(status, 0)
        self.assertEqual(parents, [-1, 0, 0])
        cost = sum(squared_distance(point, nodes[1 + label])
                   for point, label in zip(points, labels))
        self.assertAlmostEqual(cost, best_two_cluster_cost(points), places=5)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_batch_size_preserves_the_result(self):
        points = [[float(i // 6 * 10), float(i % 6)] for i in range(12)]
        small = self.run_kmeans(points, leaves=2, batch_size=1, restarts=3)
        large = self.run_kmeans(points, leaves=2, batch_size=100, restarts=3)
        self.assertEqual(small, large)

    def test_sampling_uses_the_requested_rows(self):
        points = [[float(i)] for i in range(10)]
        few = self.run_kmeans(points, leaves=1, sampling_factor=2, seed=0)
        all_rows = self.run_kmeans(points, leaves=1, sampling_factor=10, seed=0)
        self.assertEqual(few[0], 0)
        self.assertAlmostEqual(few[1][1][0], 0.5)
        self.assertAlmostEqual(all_rows[1][1][0], 4.5)

    def test_restarts_do_not_increase_training_cost(self):
        points = [[float(i), float((i * 7) % 11)] for i in range(12)]
        one = self.run_kmeans(points, leaves=3, restarts=1)
        many = self.run_kmeans(points, leaves=3, restarts=4)
        def cost(result):
            _, nodes, _, labels = result
            return sum(squared_distance(point, nodes[1 + label])
                       for point, label in zip(points, labels))
        self.assertLessEqual(cost(many), cost(one) + 1e-5)
        self.assertEqual(many, self.run_kmeans(points, leaves=3, restarts=4))

    def test_inner_product_and_cosine_group_directions(self):
        points = [[2, 0], [3, 0], [-2, 0], [-3, 0]]
        for metric in (1, 2):
            with self.subTest(metric=metric):
                status, nodes, _, labels = self.run_kmeans(
                    points, leaves=2, metric=metric, spherical=True
                )
                self.assertEqual(status, 0)
                self.assertEqual(labels[0], labels[1])
                self.assertEqual(labels[2], labels[3])
                self.assertNotEqual(labels[0], labels[2])
                for center in nodes[1:]:
                    self.assertAlmostEqual(math.sqrt(sum(x * x for x in center)), 1.0)

    def test_two_levels_form_a_tree(self):
        points = [[float(x), float(y)] for x, y in (
            (0, 0), (0, 1), (1, 0), (1, 1),
            (10, 10), (10, 11), (11, 10), (11, 11),
        )]
        status, nodes, parents, labels = self.run_kmeans(
            points, leaves=4, roots=2, batch_size=3, restarts=2
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(nodes), 7)
        self.assertEqual(parents[:3], [-1, 0, 0])
        self.assertTrue(all(parent in (1, 2) for parent in parents[3:]))
        for point, label in zip(points, labels):
            self.assertEqual(label, min(range(4), key=lambda i:
                             squared_distance(point, nodes[3 + i])))

    def test_duplicate_vectors_and_zero_cosine_vectors_stay_finite(self):
        for points, metric in (([[3, 4]] * 4, 0),
                               ([[0, 0], [0, 0], [1, 0], [-1, 0]], 2)):
            with self.subTest(points=points):
                status, nodes, _, labels = self.run_kmeans(
                    points, leaves=2, metric=metric, spherical=True
                )
                self.assertEqual(status, 0)
                self.assertTrue(all(math.isfinite(x) for node in nodes for x in node))
                self.assertTrue(all(label in (0, 1) for label in labels))

    def test_high_dimensions(self):
        for dims in (2000, 3072):
            with self.subTest(dims=dims):
                points = [[float(value)] * dims for value in (0, 1, 10, 11)]
                status, nodes, _, labels = self.run_kmeans(points, leaves=2)
                self.assertEqual(status, 0)
                self.assertEqual(labels[0], labels[1])
                self.assertEqual(labels[2], labels[3])
                self.assertNotEqual(labels[0], labels[2])
                self.assertEqual(sorted(node[0] for node in nodes[1:]), [0.5, 10.5])

    def test_large_and_tiny_spherical_vectors_keep_their_direction(self):
        for magnitude in (1e-30, 1e30):
            for metric in range(3):
                with self.subTest(magnitude=magnitude, metric=metric):
                    points = ([[magnitude, magnitude]] * 2 if metric == 0 else
                              [[magnitude, magnitude], [-magnitude, -magnitude]])
                    status, nodes, _, labels = self.run_kmeans(
                        points, leaves=1 if metric == 0 else 2, metric=metric, spherical=True, seed=0
                    )
                    self.assertEqual(status, 0)
                    for node in nodes[1:]:
                        self.assertAlmostEqual(sum(x*x for x in node), 1.0, places=5)
                    if metric != 0:
                        self.assertNotEqual(labels[0], labels[1])
                    for point, label in zip(points, labels):
                        self.assertGreater(sum(x*y for x, y in zip(point, nodes[1+label])), 0)

    def test_cosine_large_vectors_have_distinct_assignments(self):
        status, _, _, labels = self.run_kmeans(
            [[1e30, 1e30], [-1e30, -1e30]], leaves=2,
            metric=2, spherical=True, seed=0, iterations=1
        )
        self.assertEqual(status, 0)
        self.assertNotEqual(labels[0], labels[1])

    def test_centroid_sum_overflow_returns_an_error(self):
        for metric in range(3):
            with self.subTest(metric=metric):
                status, _, _, _ = self.run_kmeans(
                    [[3e38, 3e38]] * 4, leaves=1, metric=metric, spherical=False
                )
                self.assertEqual(status, 1)

    def test_invalid_shapes_and_nonfinite_input(self):
        self.assertEqual(self.run_kmeans([[1], [2]], leaves=3)[0], 1)
        self.assertEqual(self.run_kmeans([[1], [2]], leaves=1, metric=9)[0], 1)
        self.assertEqual(self.run_kmeans([[math.nan], [2]], leaves=1)[0], 1)


if __name__ == "__main__":
    unittest.main()
