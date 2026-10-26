"""Run the CPU worker as a separate process and check its binary protocol."""

import ctypes as c
import os
import struct
import subprocess
import unittest

from test_data_path import Options, options


class Request(c.Structure):
    _fields_ = [("magic", c.c_uint64), ("options", Options)]


class Reply(c.Structure):
    _fields_ = [("status", c.c_int32), ("dims", c.c_int32)] + [
        (name, c.c_uint64) for name in ("nodes", "samples", "buffer_bytes")] + [
        (name, c.c_double) for name in ("read_seconds", "sample_seconds", "clustering_seconds")] + [
        ("message", c.c_char * 256)]


WORKER = os.environ.get("VEXTOR_WORKER")


@unittest.skipUnless(WORKER, "set VEXTOR_WORKER to the native CPU worker executable")
class WorkerTests(unittest.TestCase):
    def run_worker(self, payload):
        result = subprocess.run([WORKER], input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10)
        self.assertGreaterEqual(len(result.stdout), c.sizeof(Reply), result.stderr)
        return result, Reply.from_buffer_copy(result.stdout[:c.sizeof(Reply)])

    def test_real_vx_worker_produces_expected_centroids_and_parents(self):
        request = Request(0x31524B5758544556,
                          options(row_count=4, sampling_factor=100, dims=2, restarts=3))
        payload = bytes(request) + b"".join(struct.pack("=Iff", 2, *row)
                                           for row in ((0, 0), (0, 1), (10, 10), (10, 11)))
        result, ready = self.run_worker(payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(ready.status, 0)
        size = c.sizeof(Reply)
        reply = Reply.from_buffer_copy(result.stdout[size:2 * size])
        self.assertEqual(reply.status, 0, reply.message)
        self.assertEqual((reply.nodes, reply.samples, reply.dims), (3, 4, 2))
        nodes = struct.unpack("=6f", result.stdout[2 * size:2 * size + 24])
        self.assertEqual(nodes[:2], (0, 0))
        self.assertEqual(sorted((nodes[2:4], nodes[4:6])), [(0, 0.5), (10, 10.5)])
        self.assertEqual(struct.unpack("=3i", result.stdout[2 * size + 24:]), (-1, 0, 0))

    def test_invalid_magic_and_short_request_are_rejected(self):
        for payload in (b"bad", bytes(Request(0, options()))):
            result, reply = self.run_worker(payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(reply.status, 1)

    def test_budget_is_rejected_before_any_rows_are_read(self):
        result, reply = self.run_worker(bytes(Request(0x31524B5758544556,
                                                      options(memory_budget_bytes=1))))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(reply.status, 2)

    def test_short_extra_and_nonfinite_rows_return_structured_errors(self):
        for rows in (((0, 1),), ((0, 1), (2, 3), (4, 5)), ((float("nan"), 1), (2, 3))):
            request = Request(0x31524B5758544556, options(row_count=2, leaf_count=1))
            result, ready = self.run_worker(bytes(request) + b"".join(
                struct.pack("=Iff", 2, *row) for row in rows))
            self.assertEqual(ready.status, 0)
            size = c.sizeof(Reply)
            reply = Reply.from_buffer_copy(result.stdout[size:2 * size])
            self.assertNotEqual(reply.status, 0)
            self.assertTrue(reply.message)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
