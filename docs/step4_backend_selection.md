# Step 4: backend selection and CPU fallback (November 14, 2026)

This record completes Step 4 of the [active roadmap](pg_vxvector.md).
The standalone worker now chooses the CPU or the Apple GPU before it trains, reports its choice and the reason, and uses the CPU when the GPU cannot run the request.
PostgreSQL does not use it yet. The extension still sends version 1 requests, which train on the CPU as before. Step 6 moves the extension to version 2.

The [metadata file](evidence/step4_metadata_2026-11-14.json) lists every hash. Vx is `423a10db`, the current pin. The device is an Apple M2 Pro running macOS 27.2.1.

## Modes

The modes are those of the roadmap's backend selection contract:

| Mode | Behavior |
| --- | --- |
| `auto` (0) | The GPU when it is ready and can run the request. Otherwise the CPU, with the reason. |
| `cpu` (1) | The CPU, even when a GPU is ready. |
| `gpu` (2) | The GPU, or an error with the reason, before any row is read. |

The worker selects after it plans, and before it reads a row or allocates GPU memory. The GPU is selected only if every check below passes, in this order. The first check that fails gives the reason.

| Check | Reason when it fails |
| --- | --- |
| The GPU library loads | `GPU library not installed` |
| A Metal device is present and Metal is on | `No Metal device, or Metal is turned off` |
| The metric is L2 | `The GPU supports only the L2 metric` |
| One restart | `The GPU supports only one restart` |
| No spherical centroids | `The GPU does not support spherical centroids` |
| No root centroids | `The GPU does not support root centroids` |
| `samples * dims < 2^31` and `leaves + iterations < 2^31` | `The sample is too large for the GPU kernels` |
| The CPU plan and the GPU buffers together fit `memory_budget_bytes` | `CPU and GPU buffers together exceed the memory budget` |

The GPU buffers count against the budget because GPU memory is the same memory on Apple silicon. Each buffer counts as whole 16 KiB pages, since `vx_metal_alloc` allocates whole pages.

**No CPU rerun after a GPU failure.** If the GPU fails after training starts, the worker reports `GPU training failed after it started; it was not run again on the CPU` and returns no centroids. The contract forbids a silent CPU rerun, because the caller asked for, or was told about, the GPU.

## Design

| Part | Where | Why there |
| --- | --- | --- |
| Selection: the checks, reasons, and GPU memory size | `vxvector_select_backend_reason_v1` in [src/kmeans.vx](../src/kmeans.vx) | Application logic belongs in Vx |
| The training call | `vxvector_run_reason_v2` in [src/kmeans.vx](../src/kmeans.vx) | One call site. It calls the CPU kernel directly, or a GPU trainer that C passes in with the same arguments and output layout |
| Messages and the C API: `vxvector_select_backend_v1`, `vxvector_run_v2` | [src/data_path.c](../src/data_path.c), [include/vxvector_data_path.h](../include/vxvector_data_path.h) | As in Step 2: C keeps the message text |
| Loading the GPU library, the probe, and protocol version 2 | [src/worker.c](../src/worker.c), [include/vxvector_protocol.h](../include/vxvector_protocol.h) | Process and library handling stays in C |
| The probe, the device name, and the trainer adapter | `vxvector_gpu_probe_v1` and `vxvector_gpu_kmeans_v1` in [src/gpu_l2.c](../src/gpu_l2.c), [src/gpu_device.m](../src/gpu_device.m) | Metal's device API is Objective-C |

**The GPU library is optional when the worker runs.** The worker loads it with `dlopen`. It looks for `libvxvector_gpu.dylib` next to the worker, or at the path in `VXVECTOR_GPU_LIBRARY`. A CPU-only install, such as the Linux image, has no GPU library, needs nothing new, and reports `GPU library not installed` in `auto`.

**Protocol version 2.** The request adds `backend_mode`. Each reply adds the backend, the reason code and text, the GPU memory, and the device name. The planning reply's `buffer_bytes` includes the GPU memory.
A version 1 request trains on the CPU and gets version 1 replies, so the PostgreSQL extension works unchanged.

The CPU data path API keeps `vxvector_run_cpu_v1`, which now calls `vxvector_run_v2` with the CPU. The selection adds no trusted clauses: the audit still lists 46 ([log](evidence/step4_audit_2026-11-14.txt)).

## Results

[Native suite](evidence/step4_cpu_2026-11-14.txt): 48 tests passed. These include the 37 earlier tests, 9 selection tests with a fake GPU library, and 2 with the real Metal library.
The CPU correctness checks, the data path under ASan and UBSan, and the [codegen and audit checks](evidence/step4_codegen_2026-11-14.jsonl) passed too. The GPU kernel check passed its 25 cases.

### Selection, independent of the machine's GPU

[tests/fake_gpu.c](../tests/fake_gpu.c) stands in for the GPU library. Environment variables make it ready or unavailable, and make its training succeed or fail. On success it writes leaves of 7.0, which shows which backend trained. [tests/test_backend.py](../tests/test_backend.py) checks:

| Test | Result |
| --- | --- |
| `auto` with a ready GPU and a supported request | GPU, the device name, the GPU memory that the page formula predicts, and the fake GPU's leaves |
| `cpu` with a ready GPU | CPU, `CPU requested`, no device, no GPU memory, and CPU leaves |
| Version 1 request | CPU, with the same centroids as `cpu` in version 2 |
| `auto` with no library, no device, metric IP or cosine, three restarts, spherical, or roots | CPU, with the matching reason, and CPU leaves |
| `gpu` with no library, no device, metric cosine, or three restarts | Error 7 in the planning reply, with the reason, and no rows read |
| Budget one byte short of CPU plus GPU memory | `auto` gives CPU with the budget reason. `gpu` gives an error. At the exact size, `auto` gives GPU |
| Budget one byte short of the CPU plan | Error 2 in every mode |
| GPU training fails after it starts, in `auto` and in `gpu` | Error 7, `not run again on the CPU`, no centroids, and a non-zero exit |
| Mode 9 | Error 1 |

### With the real Metal library

| Test | Result |
| --- | --- |
| `auto` against `cpu`, on 2,000 rows using every row, and on 3,000 rows sampled down to 512 | GPU with a device name, and the same bits as the CPU |
| A GPU worker of 200,000 rows, killed after 1 second | Metal had run more than 10 kernels when the kill came. The worker died of `SIGKILL`. A GPU run after it trained normally. The system frees a killed process's Metal memory |

### SIFT-128 through the worker

[Log](evidence/step4_worker_sift_2026-11-14.txt). 100,000 SIFT-128 rows, 10 iterations, one restart:

| Leaves | Samples | `auto` | `cpu` | Same bits |
| --- | --- | --- | --- | --- |
| 256 | 32,768 | GPU (`Apple M2 Pro`), 0.29 s training, 0.68 s total | 41.5 s training | yes |
| 1,024 | 65,536 | GPU, 2.01 s training, 2.39 s total | 329.1 s training | yes |

The CPU worker uses the project build, which Vx compiles at `-O0`. Against an `-O3` CPU build, the [Step 3 measurement](step3_metal_l2.md#performance) is 24x to 28x, not more than 100x.

### Linux and PostgreSQL

The Linux worker, extension, and image were rebuilt with the new sources. The image, extension, snapshot, and synthetic recall checks passed ([log](evidence/step4_linux_2026-11-14.txt)). These checks exercise version 1, so they show that the extension is unaffected. They do not test the GPU.

## Not covered

- **PostgreSQL selection.** The extension sends version 1 and reports the CPU. SQL options for the mode, and the backend in the result, are Step 6.
- **A GPU that fails during a real run.** Metal faults are not injected. The fake library tests the reporting, and the [Step 3 record](step3_metal_l2.md#size-limit-macos-can-stop-long-launches) covers the abort risk that remains.
- **Cancellation from PostgreSQL** of a GPU worker. The kill test covers the worker. The extension sends the same `SIGKILL` that it sends a CPU worker.
- **Other GPUs.** Only Metal on an M2 Pro has been tested. The probe and the library loading would serve another GPU library with the same two functions.
- **Allocation failure of the GPU buffers.** It returns status 2, which is reported as a GPU failure, but it is not tested.
