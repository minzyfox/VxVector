"""Step 4: backend selection in the worker, through protocol version 2.

The fake GPU library (tests/fake_gpu.c) makes these tests independent of the
machine's GPU. The tests with the real Metal library run only where it is
built and a Metal device is present.
"""

import ctypes as c
import os
import signal
import struct
import subprocess
import tempfile
import time
import unittest

import numpy as np

from test_data_path import Options, options
from test_worker import Reply

MAGIC_V1 = 0x31524B5758544556
MAGIC_V2 = 0x32524B5758544556
AUTO, CPU, GPU = 0, 1, 2
CPU_BACKEND, GPU_BACKEND = 0, 1
(CPU_REQUESTED, GPU_SELECTED, NOT_INSTALLED, UNAVAILABLE, METRIC, RESTARTS,
 SPHERICAL, ROOTS, SIZE, BUDGET) = range(10)
STATUS_BUDGET, STATUS_GPU = 2, 7
PAGE = 16384

WORKER = os.environ.get("VXVECTOR_WORKER")
FAKE_GPU = os.environ.get("VXVECTOR_FAKE_GPU")
REAL_GPU = os.environ.get("VXVECTOR_REAL_GPU")


class RequestV2(c.Structure):
    _fields_ = [("magic", c.c_uint64), ("options", Options),
                ("backend_mode", c.c_int32), ("reserved", c.c_int32)]


class ReplyV2(c.Structure):
    _fields_ = [("base", Reply), ("backend", c.c_int32), ("reason", c.c_int32),
                ("gpu_bytes", c.c_uint64), ("device", c.c_char * 64),
                ("reason_text", c.c_char * 128)]


def rows_payload(points):
    return b"".join(struct.pack(f"=I{len(p)}f", len(p), *p) for p in points)


def blobs(rows, dims, seed=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, size=(8, dims))
    return (centers[rng.integers(0, 8, size=rows)] + rng.normal(0, 1, size=(rows, dims))).astype(np.float32)


def gpu_bytes(samples, leaves, dims):
    pages = lambda count: -(-count * 4 // PAGE) * PAGE
    return (pages(samples * dims) + 3 * pages(leaves * dims) + 2 * pages(samples)
            + pages(leaves))


class Run:
    """One worker run: the planning reply, then the completion reply and output."""

    def __init__(self, result, version=2):
        self.result = result
        out = result.stdout
        kind = ReplyV2 if version == 2 else Reply
        size = c.sizeof(kind)
        self.ready = kind.from_buffer_copy(out[:size])
        self.done = kind.from_buffer_copy(out[size:2 * size]) if len(out) >= 2 * size else None
        self.output = out[2 * size:]

    def nodes(self):
        base = self.done.base if isinstance(self.done, ReplyV2) else self.done
        count, dims = base.nodes, base.dims
        return np.frombuffer(self.output[:count * dims * 4], dtype=np.float32).reshape(count, dims)


@unittest.skipUnless(WORKER and FAKE_GPU, "set VXVECTOR_WORKER and VXVECTOR_FAKE_GPU")
class BackendSelectionTests(unittest.TestCase):
    def run_worker(self, mode, opts, points=(), library=None, env=None, version=2):
        environment = dict(os.environ)
        environment.pop("FAKE_GPU_PROBE", None)
        environment.pop("FAKE_GPU_TRAIN", None)
        environment["VXVECTOR_GPU_LIBRARY"] = FAKE_GPU if library is None else library
        environment.update(env or {})
        if version == 2:
            request = bytes(RequestV2(MAGIC_V2, opts, mode, 0))
        else:
            request = struct.pack("=Q", MAGIC_V1) + bytes(opts)
        result = subprocess.run([WORKER], input=request + rows_payload(points),
                                capture_output=True, env=environment, timeout=60)
        return Run(result, version)

    def supported(self, **overrides):
        values = dict(row_count=40, leaf_count=4, sampling_factor=100, restarts=1,
                      iterations=5, memory_budget_bytes=64 * 1024 * 1024)
        values.update(overrides)
        return options(**values)

    def points(self, count=40, dims=2):
        return blobs(count, dims).tolist()

    def test_auto_selects_a_ready_gpu_and_trains_on_it(self):
        run = self.run_worker(AUTO, self.supported(), self.points())
        self.assertEqual(run.ready.base.status, 0, run.ready.base.message)
        self.assertEqual((run.ready.backend, run.ready.reason), (GPU_BACKEND, GPU_SELECTED))
        self.assertEqual(run.ready.device, b"Fake GPU")
        self.assertEqual(run.ready.gpu_bytes, gpu_bytes(40, 4, 2))
        self.assertEqual(run.done.base.status, 0, run.done.base.message)
        self.assertEqual(run.result.returncode, 0)
        # The fake GPU's leaves, so the GPU trainer ran.
        self.assertTrue((run.nodes()[1:] == 7.0).all())

    def test_gpu_leaves_get_cpu_roots_and_parents(self):
        run = self.run_worker(AUTO, self.supported(root_count=2), self.points())
        self.assertEqual((run.ready.backend, run.ready.reason), (GPU_BACKEND, GPU_SELECTED))
        self.assertEqual(run.done.base.status, 0, run.done.base.message)
        nodes = run.nodes()
        self.assertEqual(nodes.shape, (1 + 2 + 4, 2))
        # The fake GPU's leaves, and roots that the CPU trained on them.
        self.assertTrue((nodes[3:] == 7.0).all())
        self.assertTrue((nodes[1:3] == 7.0).all())
        self.assertTrue((nodes[0] == 0).all())
        base = run.done.base
        parents = np.frombuffer(run.output[base.nodes * base.dims * 4:], dtype=np.int32)
        self.assertEqual(parents[:3].tolist(), [-1, 0, 0])
        self.assertTrue(set(parents[3:].tolist()) <= {1, 2})

    def test_cpu_mode_ignores_a_ready_gpu(self):
        run = self.run_worker(CPU, self.supported(), self.points())
        self.assertEqual((run.ready.backend, run.ready.reason), (CPU_BACKEND, CPU_REQUESTED))
        self.assertEqual(run.ready.device, b"")
        self.assertEqual(run.ready.gpu_bytes, 0)
        self.assertEqual(run.done.base.status, 0)
        self.assertFalse((run.nodes()[1:] == 7.0).any())

    def test_version_1_requests_still_train_on_the_cpu(self):
        v1 = self.run_worker(None, self.supported(), self.points(), version=1)
        v2 = self.run_worker(CPU, self.supported(), self.points())
        self.assertEqual(v1.done.status, 0, v1.done.message)
        np.testing.assert_array_equal(v1.nodes(), v2.nodes())

    def test_auto_selects_the_cpu_and_says_why(self):
        cases = [
            (dict(), dict(library="/nonexistent/libvxvector_gpu.dylib"), NOT_INSTALLED),
            (dict(), dict(env={"FAKE_GPU_PROBE": "unavailable"}), UNAVAILABLE),
            (dict(metric=2), dict(), METRIC),
            (dict(metric=1), dict(), METRIC),
            (dict(restarts=3), dict(), RESTARTS),
            (dict(spherical=1), dict(), SPHERICAL),
        ]
        for overrides, how, reason in cases:
            with self.subTest(reason=reason, overrides=overrides):
                run = self.run_worker(AUTO, self.supported(**overrides), self.points(), **how)
                self.assertEqual(run.ready.base.status, 0, run.ready.base.message)
                self.assertEqual((run.ready.backend, run.ready.reason), (CPU_BACKEND, reason))
                self.assertTrue(run.ready.reason_text)
                self.assertEqual(run.done.base.status, 0, run.done.base.message)
                self.assertFalse((run.nodes()[1:] == 7.0).any())

    def test_gpu_mode_fails_before_reading_rows_when_the_gpu_cannot_run(self):
        cases = [
            (dict(), dict(library="/nonexistent/libvxvector_gpu.dylib"), NOT_INSTALLED),
            (dict(), dict(env={"FAKE_GPU_PROBE": "unavailable"}), UNAVAILABLE),
            (dict(metric=2), dict(), METRIC),
            (dict(restarts=3), dict(), RESTARTS),
        ]
        for overrides, how, reason in cases:
            with self.subTest(reason=reason):
                # No rows follow the request. The worker must answer without reading any.
                run = self.run_worker(GPU, self.supported(**overrides), (), **how)
                self.assertEqual(run.ready.base.status, STATUS_GPU)
                self.assertEqual(run.ready.reason, reason)
                self.assertIn(b"GPU training was required", run.ready.base.message)
                self.assertIsNone(run.done)
                self.assertNotEqual(run.result.returncode, 0)

    def test_budget_counts_gpu_memory_with_the_cpu_plan(self):
        opts = self.supported()
        plan = self.run_worker(CPU, opts).ready.base.buffer_bytes
        needed = plan + gpu_bytes(40, 4, 2)
        # Enough for the CPU plan, one byte short of the GPU's memory too.
        tight = self.supported(memory_budget_bytes=needed - 1)
        auto = self.run_worker(AUTO, tight, self.points())
        self.assertEqual((auto.ready.backend, auto.ready.reason), (CPU_BACKEND, BUDGET))
        self.assertEqual(auto.done.base.status, 0)
        gpu = self.run_worker(GPU, tight)
        self.assertEqual((gpu.ready.base.status, gpu.ready.reason), (STATUS_GPU, BUDGET))
        exact = self.run_worker(AUTO, self.supported(memory_budget_bytes=needed), self.points())
        self.assertEqual(exact.ready.backend, GPU_BACKEND)

    def test_budget_too_small_for_the_cpu_fails_in_every_mode(self):
        # The planning reply's buffer_bytes is the whole budget the CPU needs.
        needed = self.run_worker(CPU, self.supported()).ready.base.buffer_bytes
        for mode in (AUTO, CPU, GPU):
            with self.subTest(mode=mode):
                run = self.run_worker(mode, self.supported(memory_budget_bytes=needed - 1))
                self.assertEqual(run.ready.base.status, STATUS_BUDGET, run.ready.base.message)
                self.assertIsNone(run.done)

    def test_gpu_failure_after_training_starts_is_reported_not_rerun(self):
        for mode in (AUTO, GPU):
            with self.subTest(mode=mode):
                run = self.run_worker(mode, self.supported(), self.points(),
                                      env={"FAKE_GPU_TRAIN": "fail"})
                self.assertEqual(run.ready.backend, GPU_BACKEND)
                self.assertEqual(run.done.base.status, STATUS_GPU)
                self.assertIn(b"not run again on the CPU", run.done.base.message)
                self.assertEqual(run.output, b"")
                self.assertNotEqual(run.result.returncode, 0)

    def test_invalid_mode_is_rejected(self):
        run = self.run_worker(9, self.supported())
        self.assertEqual(run.ready.base.status, 1)
        self.assertIsNone(run.done)


def metal_ready():
    if not REAL_GPU or not os.path.exists(REAL_GPU):
        return False
    library = c.CDLL(REAL_GPU)
    return library.vxvector_gpu_probe_v1(c.create_string_buffer(64), 64) == 0


@unittest.skipUnless(WORKER and metal_ready(), "needs VXVECTOR_REAL_GPU and a Metal device")
class RealGpuTests(unittest.TestCase):
    def run_real(self, mode, opts, points):
        environment = dict(os.environ, VXVECTOR_GPU_LIBRARY=REAL_GPU)
        result = subprocess.run([WORKER], input=bytes(RequestV2(MAGIC_V2, opts, mode, 0))
                                + rows_payload(points), capture_output=True,
                                env=environment, timeout=300)
        return Run(result)

    def test_gpu_and_cpu_workers_give_the_same_bits(self):
        # The second case samples 512 of 3,000 rows, as the data path does. The
        # third has roots, which the CPU trains on the GPU's leaves.
        for rows, dims, leaves, factor, roots in ((2000, 16, 8, 1000, 0), (3000, 24, 16, 32, 0),
                                                   (4000, 12, 32, 64, 4)):
            with self.subTest(rows=rows, roots=roots):
                points = blobs(rows, dims).tolist()
                opts = options(row_count=rows, dims=dims, leaf_count=leaves, root_count=roots,
                               sampling_factor=factor, restarts=1, iterations=8,
                               batch_size=256, memory_budget_bytes=256 * 1024 * 1024)
                gpu = self.run_real(AUTO, opts, points)
                cpu = self.run_real(CPU, opts, points)
                self.assertEqual(gpu.ready.backend, GPU_BACKEND, gpu.ready.reason_text)
                self.assertTrue(gpu.ready.device)
                self.assertEqual(gpu.done.base.status, 0, gpu.done.base.message)
                self.assertEqual(cpu.done.base.status, 0, cpu.done.base.message)
                self.assertEqual(gpu.done.base.samples, min(rows, leaves * factor))
                np.testing.assert_array_equal(gpu.nodes().view(np.uint32),
                                              cpu.nodes().view(np.uint32))
                self.assertEqual(gpu.output, cpu.output)

    def test_a_killed_gpu_worker_leaves_the_gpu_usable(self):
        rows, dims, leaves = 200000, 64, 256
        points = blobs(rows, dims)
        opts = options(row_count=rows, dims=dims, leaf_count=leaves, sampling_factor=800,
                       restarts=1, iterations=50, batch_size=4096,
                       memory_budget_bytes=2 * 1024 * 1024 * 1024)
        payload = bytes(RequestV2(MAGIC_V2, opts, GPU, 0)) + b"".join(
            struct.pack("=I", dims) + row.tobytes() for row in points)
        # The Metal log goes to a file, so the worker never blocks on a full pipe.
        environment = dict(os.environ, VXVECTOR_GPU_LIBRARY=REAL_GPU, VX_DISPATCH_VERBOSE="1")
        with tempfile.TemporaryFile() as log:
            worker = subprocess.Popen([WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=log, env=environment)
            worker.stdin.write(payload)
            worker.stdin.close()
            ready = ReplyV2.from_buffer_copy(worker.stdout.read(c.sizeof(ReplyV2)))
            self.assertEqual((ready.base.status, ready.backend), (0, GPU_BACKEND))
            time.sleep(1.0)
            self.assertIsNone(worker.poll(), "training ended before the kill; make it longer")
            worker.send_signal(signal.SIGKILL)
            self.assertEqual(worker.wait(timeout=30), -signal.SIGKILL)
            worker.stdout.close()
            log.seek(0)
            text = log.read().decode(errors="replace")
        # The kill came during GPU training: kernels had run on Metal.
        self.assertGreater(text.count("[Vx Metal] vx_npu_kernel_"), 10)
        self.assertNotIn("Falling back", text)
        # A new GPU run after the kill trains normally.
        small = options(row_count=500, dims=8, leaf_count=4, sampling_factor=1000,
                        restarts=1, iterations=3, memory_budget_bytes=64 * 1024 * 1024)
        after = self.run_real(GPU, small, blobs(500, 8).tolist())
        self.assertEqual(after.done.base.status, 0, after.done.base.message)


if __name__ == "__main__":
    unittest.main()
