# Step 6: native macOS PostgreSQL with the Apple GPU — November 15, 2026

This record validates Step 6 of the [active roadmap](pg_vxvector.md) on native macOS.
SQL samples the source table, the worker trains on Metal, and VectorChord builds and serves the index.

## Environment

| Item | Value |
| --- | --- |
| Machine | Apple M2 Pro |
| OS | macOS 27.2.1 (26B131), xnu-13432.41.4 |
| PostgreSQL | Homebrew `postgresql@17` 17.12, arm64 |
| pgvector | 0.8.2 |
| VectorChord | 1.1.1, built from source with `strip = "none"` (see below) |
| Vx | `423a10db`, as in [vx-revision.txt](../vx-revision.txt) |
| Rust, linker | rustc 1.94.0, ld-27106.1 |

All checks ran on a scratch cluster created with `initdb`, with `shared_preload_libraries = 'vchord'`.
`vxvector.worker_path` pointed at `build/vxvector-worker`, and the Metal library was next to the worker.
The extensions were installed into the Homebrew `pkglibdir` and `sharedir`.

| File | SHA-256 |
| --- | --- |
| `vchord.dylib` | `8aa265b18fd1e266deb18fce53b6cc26dce600fd2bbd5d09e7962dc4f2092506` |
| `pg_vxvector.dylib` | `e750b382047ebc5d49e6a5a49912d5aa578afc60225daea5a5f7a6671a4f51df` |
| `vector.dylib` | `59e6e3a95827c6f8ab5a9cf2edd12965c357075bdd9e21b2bc0215b1dea5a4c4` |
| `build/vxvector-worker` | `d24058aae416918a9928d12d70d3f77fb73aafa079e544999f76c9f8b48a5f42` |
| `build/libvxvector_kmeans.dylib` | `f6c5debaf72b8da6b0da0441a0fbb59e15c40b60c0267ff2ae6c74be0c6d33d1` |
| `build/gpu/libvxvector_gpu.dylib` | `0d30246c4ad725806ed02fefea229ad0827d4ccd8104a0e43b67e1fb5936765c` |
| `build/kmeans.o` | `6368bc5f275cff6aed804d749938b07e92454c68373773f8f36e0117c9f1dfd0` |

## Blocker 1: stripped bundles do not load

dyld rejected VectorChord's `vchord.dylib` with `mis-aligned LINKEDIT string pool`.
The cause is the strip step, not link-time optimization:

| VectorChord release profile override | String table offset | dyld |
| --- | --- | --- |
| `lto = "off"` | 6642220 (4 mod 8) | Rejects |
| `lto = "off"`, `strip = "none"` | 6827832 (8-aligned) | Loads |
| Stock `lto = "fat"`, `strip = "none"` | 7145248 (8-aligned) | Loads, and `CREATE EXTENSION vchord` succeeds |

Build VectorChord natively with stripping disabled:

```sh
arch -arm64 env PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config \
  CARGO_PROFILE_RELEASE_STRIP=none cargo run -p xtask -- build
```

`pg_vxvector` failed the same way, at string table offset 861388 (4 mod 8).
Its dev profile has `debug = 0`, and Cargo then strips debug info by default.
[build_macos_extension.sh](../scripts/build_macos_extension.sh) now sets `CARGO_PROFILE_DEV_STRIP=none`, which gives offset 913232 (8-aligned).

The Visual Studio Code terminal on this machine runs as x86_64 under Rosetta, so `cc`, `otool`, and the builds need `arch -arm64`.

## Blocker 2: the worker can miss the end of its input

About one SQL call in seventy hung until `worker_timeout_ms`. A standalone harness also hung with the CPU backend.
The backend had sent every row, called `shutdown(SHUT_WR)`, and was waiting for the reply.
The worker was blocked in `read()` for the end-of-input check after the last row.
Its socket had received all 16,072 request and row bytes, had an empty receive queue, and was flagged `SS_CANTRCVMORE` (`netstat` state `0x22`).
The read woke only when the backend's timeout closed the socket.

A standalone C program reproduces this without VxVector code: 5 hangs in 500 runs.
A child process writes a 616-byte reply on a socketpair and then reads 1000 records with blocking `read()`.
The parent shuts down its write side after the last record.
Without the child's reply write, the program never hangs (0 in 1000 runs). Pipes never hang either.
This looks like a lost wakeup in the macOS 27.2.1 kernel.

[worker.c](../src/worker.c) now counts rows. After `row_count` rows, it waits for the final read with `poll()` and a 100 ms timeout, and each `poll` reads the socket state again.
This keeps the check for surplus rows, leaves the wire protocol unchanged, and does not slow the earlier row reads.
[test_worker.py](../tests/test_worker.py) drives the worker over a socketpair 300 times, as the bridge does.
The test failed 3 of 3 times with the previous worker and passed 3 of 3 times with the fix.

| Run | Before the fix | After the fix |
| --- | --- | --- |
| Socketpair harness, GPU, row-by-row sends | 18 hangs in 200 | 0 in 400 |
| Socketpair harness, CPU, row-by-row sends | 2 hangs in 200 | 0 in 400 |
| SQL calls, mostly in `gpu` mode | 9 timeouts in about 610 | 0 in 240 |

## Results

The native CPU suite, `scripts/check_cpu.sh`, passes all 57 tests, including the new socket test, the correctness checks, and the data-path checks.

| Check | Result |
| --- | --- |
| README walkthrough in default `auto` mode | `backend: gpu`, device `Apple M2 Pro`. Five nodes, and the `vchordrq` index serves an index scan |
| `cpu`, `gpu`, and `auto` centroids for the same request | Identical |
| `gpu` mode without a GPU library | Error: `GPU training was required, but the GPU cannot run this request: GPU library not installed` |
| `auto` without a GPU library, with no SQL changes | `backend: cpu`, reason `GPU library not installed` |
| `worker_path = /usr/bin/false` | Error. No outputs remain |
| `worker_timeout_ms = 1` on 400,000 rows | `VxVector worker timed out`. No outputs remain |
| `statement_timeout = 150ms` on 400,000 rows | Cancelled. No outputs remain |
| A successful build, then `ROLLBACK` | The centroid table and the index are gone |
| A build after each failure | Succeeds on the GPU in the same session |
| `scripts/check_recall.py`: synthetic exact-scan gate | Passed. 24 index builds, 120 queries, minimum recall@10 of 1.0 |
| `scripts/check_real_recall.py`: SIFT-128 and GloVe-25 against VectorChord's own trainer | Passed. 24 builds, 3072 queries, 24 setting comparisons. All 12 VxVector builds trained on the GPU |

The recall scripts call `docker exec … psql`. For this run, a shim on `PATH` ran the same `psql` arguments against the native cluster.

## Limits

This record covers one machine and a scratch cluster.
Install packaging, upgrades, and production-scale performance remain unvalidated.
The macOS socket issue and VectorChord's strip alignment have not yet been reported upstream.
