"""Exercise bounded source reads through the C adapter and real Vx library."""

import ctypes as c
import math
import unittest

import test_kmeans


class Options(c.Structure):
    _fields_ = [("row_count", c.c_uint64), ("memory_budget_bytes", c.c_uint64)] + [
        (name, c.c_int32) for name in (
            "dims", "root_count", "leaf_count", "sampling_factor", "batch_size",
            "iterations", "restarts", "metric", "spherical", "seed")]


class Plan(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in (
        "sample_count", "batch_rows", "node_count", "hash_slots", "buffer_bytes")]


class Error(c.Structure):
    _fields_ = [("code", c.c_int32), ("message", c.c_char * 256)]


class Row(c.Structure):
    _fields_ = [("values", c.POINTER(c.c_float)), ("dims", c.c_uint32),
                ("type", c.c_int32)]


class Result(c.Structure):
    _fields_ = [("plan", Plan)] + [
        (name, c.c_double) for name in (
            "read_seconds", "sample_seconds", "clustering_seconds")] + [
        ("sample", c.POINTER(c.c_float)), ("sample_rows", c.POINTER(c.c_uint64)),
        ("nodes", c.POINTER(c.c_float)), ("parents", c.POINTER(c.c_int32)),
        ("assignments", c.POINTER(c.c_int32))]


Next = c.CFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(Row))


def options(**overrides):
    values = dict(row_count=30, memory_budget_bytes=1024 * 1024, dims=2,
                  root_count=0, leaf_count=2, sampling_factor=3, batch_size=4,
                  iterations=4, restarts=2, metric=0, spherical=0, seed=42)
    values.update(overrides)
    return Options(**values)


class Reader:
    def __init__(self, points, null_at=None, unsupported_at=None, fail_at=None):
        self.points = iter(points)
        self.calls = 0
        self.null_at = null_at
        self.unsupported_at = unsupported_at
        self.fail_at = fail_at
        self.buffer = None
        self.callback = Next(self.next)

    def next(self, context, destination):
        index = self.calls
        self.calls += 1
        if index == self.fail_at:
            return -1
        try:
            point = next(self.points)
        except StopIteration:
            return 0
        self.buffer = (c.c_float * len(point))(*point)
        row = destination.contents
        row.values = (c.POINTER(c.c_float)() if index == self.null_at
                      else c.cast(self.buffer, c.POINTER(c.c_float)))
        row.dims = len(point)
        row.type = 1 if index == self.unsupported_at else 0
        return 1


class DataPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_kmeans.KmeansTests.setUpClass()
        cls.library = test_kmeans.KmeansTests.library
        cls.library.vxvector_plan_cpu_v1.argtypes = [c.POINTER(Options), c.POINTER(Plan), c.POINTER(Error)]
        cls.library.vxvector_plan_cpu_v1.restype = c.c_int32
        cls.library.vxvector_run_cpu_v1.argtypes = [c.POINTER(Options), Next, c.c_void_p, c.POINTER(Result), c.POINTER(Error)]
        cls.library.vxvector_run_cpu_v1.restype = c.c_int32
        cls.library.vxvector_free_cpu_result_v1.argtypes = [c.POINTER(Result)]
        cls.library.vxvector_free_cpu_result_v1.restype = None

    def plan(self, request, expected=0):
        plan, error = Plan(), Error()
        status = self.library.vxvector_plan_cpu_v1(c.byref(request), c.byref(plan), c.byref(error))
        self.assertEqual(status, expected, error.message.decode())
        self.assertEqual(error.code, expected)
        return plan

    def run_source(self, request, reader, expected=0):
        result, error = Result(), Error()
        status = self.library.vxvector_run_cpu_v1(c.byref(request), reader.callback, None,
                                                c.byref(result), c.byref(error))
        try:
            self.assertEqual(status, expected, error.message.decode())
            self.assertEqual(error.code, expected)
            if expected:
                self.assertFalse(result.sample)
                self.assertFalse(result.nodes)
                self.assertEqual(result.plan.buffer_bytes, 0)
                self.assertTrue(error.message)
                return None
            samples = result.plan.sample_count
            nodes = result.plan.node_count
            return dict(
                sample_rows=list(result.sample_rows[:samples]),
                sample=[list(result.sample[i * request.dims:(i + 1) * request.dims])
                        for i in range(samples)],
                nodes=[list(result.nodes[i * request.dims:(i + 1) * request.dims])
                       for i in range(nodes)],
                parents=list(result.parents[:nodes]),
                assignments=list(result.assignments[:samples]),
                timings=(result.read_seconds, result.sample_seconds, result.clustering_seconds))
        finally:
            self.library.vxvector_free_cpu_result_v1(c.byref(result))
            self.library.vxvector_free_cpu_result_v1(c.byref(result))

    @staticmethod
    def points(count):
        return ([float(i), float(i % 3)] for i in range(count))

    def test_collected_sample_matches_direct_vx_for_all_metrics_and_tree_shapes(self):
        reference = test_kmeans.KmeansTests()
        for metric in range(3):
            for roots in (0, 2):
                with self.subTest(metric=metric, roots=roots):
                    request = options(metric=metric, root_count=roots, spherical=int(metric != 0))
                    result = self.run_source(request, Reader(self.points(30)))
                    direct = reference.run_kmeans(
                        result["sample"], leaves=2, roots=roots, sampling_factor=3,
                        batch_size=4, iterations=4, restarts=2, metric=metric,
                        spherical=metric != 0, seed=42)
                    self.assertEqual(direct[0], 0)
                    self.assertEqual(result["nodes"], direct[1])
                    self.assertEqual(result["parents"], direct[2])
                    self.assertEqual(result["assignments"], direct[3])
                    source = list(self.points(30))
                    self.assertEqual(result["sample"], [source[i] for i in result["sample_rows"]])
                    self.assertTrue(all(math.isfinite(t) and t >= 0 for t in result["timings"]))
                    self.assertGreater(result["timings"][2], 0)

    def test_sample_is_distinct_repeatable_and_independent_of_batches(self):
        results = []
        for batch in (1, 4, 100):
            result = self.run_source(options(batch_size=batch), Reader(self.points(30)))
            self.assertEqual(len(set(result["sample_rows"])), 6)
            self.assertTrue(all(0 <= row < 30 for row in result["sample_rows"]))
            result.pop("timings")
            results.append(result)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        different = self.run_source(options(seed=43), Reader(self.points(30)))
        self.assertNotEqual(results[0]["sample_rows"], different["sample_rows"])

    def test_full_sample_and_single_row(self):
        for count in (1, 2, 7):
            result = self.run_source(options(row_count=count, leaf_count=1, sampling_factor=100),
                                     Reader(self.points(count)))
            self.assertEqual(sorted(result["sample_rows"]), list(range(count)))

    def test_sampling_handles_collisions_and_partial_final_batches(self):
        for count in (2, 3, 7, 17, 31):
            for seed in (0, 1, 42, 2**31 - 1):
                with self.subTest(count=count, seed=seed):
                    request = options(row_count=count, leaf_count=1,
                                      sampling_factor=count - 1, batch_size=3, seed=seed)
                    reader = Reader(self.points(count))
                    result = self.run_source(request, reader)
                    self.assertEqual(len(set(result["sample_rows"])), count - 1)
                    self.assertTrue(all(0 <= i < count for i in result["sample_rows"]))
                    source = list(self.points(count))
                    self.assertEqual(result["sample"], [source[i] for i in result["sample_rows"]])
                    self.assertEqual(reader.calls, count + 1)

    def test_all_rows_are_checked_including_rows_outside_the_sample(self):
        good = self.run_source(options(), Reader(self.points(30)))
        unsampled = next(i for i in range(30) if i not in good["sample_rows"])
        for bad in (None, "unsupported", [1.0], [math.nan, 0], [math.inf, 0]):
            with self.subTest(bad=bad):
                points = list(self.points(30))
                kwargs = {}
                if bad is None:
                    kwargs["null_at"] = unsampled
                elif bad == "unsupported":
                    kwargs["unsupported_at"] = unsampled
                else:
                    points[unsampled] = bad
                self.run_source(options(), Reader(points, **kwargs), expected=1)

    def test_source_count_changes_and_reader_errors_leave_no_result(self):
        for count, fail_at in ((29, None), (31, None), (30, 4), (30, 30)):
            with self.subTest(count=count, fail_at=fail_at):
                self.run_source(options(), Reader(self.points(count), fail_at=fail_at), expected=5)

    def test_budget_rejected_before_reader_and_exact_budget_succeeds(self):
        plan = self.plan(options())
        reader = Reader(self.points(30))
        self.run_source(options(memory_budget_bytes=plan.buffer_bytes - 1), reader, expected=2)
        self.assertEqual(reader.calls, 0)
        self.run_source(options(memory_budget_bytes=plan.buffer_bytes), reader)
        self.assertEqual(reader.calls, 31)

    def test_memory_plan_covers_every_buffer_and_does_not_scale_with_table(self):
        plan = self.plan(options())
        # Six samples, four read rows, three nodes, two leaves, two dimensions.
        expected = (6 * 2 * 4 + 4 * 2 * 4 + 3 * 2 * 4 + 3 * 4 + 6 * 4
                    + 2 * 2 * 4 * 2 + 2 * 4 + 6 * 8 + 6 * 16 + 16 * 8)
        self.assertEqual(plan.buffer_bytes, expected)
        huge = self.plan(options(row_count=2**40))
        self.assertEqual(huge.buffer_bytes, expected)
        self.assertEqual(huge.sample_count, 6)

    def test_shape_and_size_errors_are_rejected_before_source_reads(self):
        cases = [({"row_count": 0}, 1), ({"row_count": 2**63}, 1),
                 ({"dims": 0}, 1), ({"root_count": 3}, 1),
                 ({"leaf_count": 31}, 1), ({"sampling_factor": 0}, 1),
                 ({"batch_size": 0}, 1), ({"metric": 3}, 1),
                 ({"spherical": 2}, 1), ({"seed": -1}, 1),
                 ({"iterations": 0}, 1), ({"restarts": 0}, 1),
                 ({"row_count": 2**40, "leaf_count": 2, "sampling_factor": 2**31 - 1}, 3),
                 ({"row_count": 2**31 - 1, "dims": 2**31 - 1,
                   "leaf_count": 2**31 - 2, "sampling_factor": 1}, 3)]
        for overrides, status in cases:
            with self.subTest(overrides=overrides):
                reader = Reader(self.points(30))
                self.run_source(options(**overrides), reader, expected=status)
                self.assertEqual(reader.calls, 0)


if __name__ == "__main__":
    unittest.main()
