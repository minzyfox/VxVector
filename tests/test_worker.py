"""Run the CPU worker as a separate process and check its binary protocol."""

import ctypes as c
import os
import socket
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


class RequestV2(c.Structure):
    _fields_ = [("magic", c.c_uint64), ("options", Options),
                ("backend_mode", c.c_int32), ("reserved", c.c_int32)]


class ReplyV2(c.Structure):
    _fields_ = [("base", Reply), ("backend", c.c_int32), ("reason", c.c_int32),
                ("gpu_bytes", c.c_uint64), ("device", c.c_char * 64),
                ("reason_text", c.c_char * 128),
                ("gpu_allocation_seconds", c.c_double), ("gpu_copy_in_seconds", c.c_double),
                ("gpu_training_seconds", c.c_double), ("gpu_copy_out_seconds", c.c_double),
                ("gpu_version", c.c_char * 64)]


WORKER = os.environ.get("VXVECTOR_WORKER")
FAKE_GPU = os.environ.get("VXVECTOR_FAKE_GPU")


@unittest.skipUnless(WORKER, "set VXVECTOR_WORKER to the native CPU worker executable")
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

    def test_end_of_rows_is_seen_over_a_socket_after_shutdown(self):
        # pg_vxvector talks to the worker over a socketpair: it reads the plan
        # reply, sends each row as two writes, then shuts down its write side.
        # On macOS 27.2.1 the worker's final blocking read could miss that
        # shutdown in about one run in a hundred and sleep until the backend timed
        # out. 300 runs catch that about 95 percent of the time.
        rows = 1000
        request = bytes(Request(0x31524B5758544556,
                                options(row_count=rows, dims=3, leaf_count=4, sampling_factor=40)))
        size = c.sizeof(Reply)
        for run in range(300):
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            worker = subprocess.Popen([WORKER], stdin=child, stdout=child, stderr=subprocess.DEVNULL)
            child.close()
            try:
                parent.settimeout(5)
                parent.sendall(request)
                ready = Reply.from_buffer_copy(self.receive(parent, size))
                self.assertEqual(ready.status, 0, ready.message)
                for i in range(rows):
                    parent.sendall(struct.pack("=I", 3))
                    parent.sendall(struct.pack("=3f", i % 4 * 10, i % 7, i % 11))
                parent.shutdown(socket.SHUT_WR)
                try:
                    reply = Reply.from_buffer_copy(self.receive(parent, size))
                except socket.timeout:
                    self.fail(f"run {run}: the worker did not see the end of input")
                self.assertEqual(reply.status, 0, reply.message)
            finally:
                parent.close()
                if worker.poll() is None:
                    worker.kill()
                worker.wait()

    @staticmethod
    def receive(connection, size):
        data = b""
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                raise EOFError("the worker closed the connection")
            data += chunk
        return data

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

    def test_v2_auto_uses_ready_gpu_and_reports_backend(self):
        if not FAKE_GPU:
            self.skipTest("set VXVECTOR_FAKE_GPU to the fake GPU library")
        request = RequestV2(0x32524B5758544556,
                            options(row_count=4, sampling_factor=100, dims=2,
                                    leaf_count=2, iterations=1, restarts=1),
                            0, 0)
        payload = bytes(request) + b"".join(struct.pack("=Iff", 2, *row)
                                           for row in ((0, 0), (0, 1), (10, 10), (10, 11)))
        environment = os.environ.copy()
        environment["VXVECTOR_GPU_LIBRARY"] = FAKE_GPU
        result = subprocess.run([WORKER], input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(len(result.stdout), 2 * c.sizeof(ReplyV2))
        ready = ReplyV2.from_buffer_copy(result.stdout[:c.sizeof(ReplyV2)])
        done = ReplyV2.from_buffer_copy(
            result.stdout[c.sizeof(ReplyV2):2 * c.sizeof(ReplyV2)])
        self.assertEqual(ready.base.status, 0, ready.base.message)
        self.assertEqual((ready.backend, ready.reason), (1, 1))
        self.assertEqual(ready.device.split(b"\0", 1)[0], b"Fake GPU")
        self.assertEqual(done.base.status, 0, done.base.message)
        self.assertEqual(done.gpu_version.split(b"\0", 1)[0], b"Fake GPU 1")
        self.assertEqual(done.gpu_training_seconds, 0.25)
        self.assertGreater(done.gpu_bytes, 0)
        self.assertGreaterEqual(done.base.buffer_bytes, done.gpu_bytes)

    def test_v2_auto_falls_back_to_cpu_when_gpu_library_is_missing(self):
        request = RequestV2(0x32524B5758544556,
                            options(row_count=4, sampling_factor=100, dims=2,
                                    leaf_count=1, iterations=1, restarts=1),
                            0, 0)
        payload = bytes(request) + b"".join(struct.pack("=Iff", 2, *row)
                                           for row in ((0, 0), (0, 1), (10, 10), (10, 11)))
        environment = os.environ.copy()
        environment["VXVECTOR_GPU_LIBRARY"] = "/no/such/vxvector-gpu-library"
        result = subprocess.run([WORKER], input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(len(result.stdout), 2 * c.sizeof(ReplyV2))
        ready = ReplyV2.from_buffer_copy(result.stdout[:c.sizeof(ReplyV2)])
        done = ReplyV2.from_buffer_copy(
            result.stdout[c.sizeof(ReplyV2):2 * c.sizeof(ReplyV2)])
        self.assertEqual((ready.backend, ready.reason), (0, 2))
        self.assertEqual(ready.reason_text.split(b"\0", 1)[0], b"GPU library not installed")
        self.assertEqual(done.base.status, 0, done.base.message)


if __name__ == "__main__":
    unittest.main()
