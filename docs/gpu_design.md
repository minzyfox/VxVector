# GPU training design

How VxVector trains on a GPU, and the rules its results must meet. The [roadmap](pg_vxvector.md) says what is done and what comes next.
The validation runs behind these rules are in the git history of `docs/` (Steps 3 to 6, November 2026).

## Layout

| Part | File | Runs on |
| --- | --- | --- |
| Kernels and the training loop | [src/gpu_kmeans.vx](../src/gpu_kmeans.vx) | Every GPU. It names no GPU runtime |
| Worker API, seeding, timing, status codes, spherical normalization | [src/gpu_l2.c](../src/gpu_l2.c) | Every GPU |
| Memory, copies, wait, device name | [src/gpu_backend.h](../src/gpu_backend.h): [gpu_backend_metal.m](../src/gpu_backend_metal.m), [gpu_backend_cuda.c](../src/gpu_backend_cuda.c) | One file per GPU |
| Backend selection and the training call | `vxvector_select_backend_reason_v1` and `vxvector_run_reason_v2` in [src/kmeans.vx](../src/kmeans.vx) | Worker, before any row is read |
| Loading the GPU library | [src/worker.c](../src/worker.c) | Worker |

The worker loads the GPU library with `dlopen`, from beside the worker or from `VXVECTOR_GPU_LIBRARY`. A CPU-only install needs no GPU library and reports `GPU library not installed` in `auto`.

C allocates every tensor through the backend, and the Vx runtime uses it in place: Metal wraps shared memory without a copy, and the CUDA runtime recognizes device pointers. The samples are copied in once, and every launch runs on the same buffers.

## Backend selection

The modes are those of the roadmap's backend selection contract: `auto`, `cpu`, and `gpu`.
The worker selects after it plans, and before it reads a row or allocates GPU memory. The first failed check gives the reason:

| Check | Reason when it fails |
| --- | --- |
| The GPU library loads | `GPU library not installed` |
| A device is present, and Metal is on | `No Metal device, or Metal is turned off` |
| `samples * dims < 2^31` and `leaves + iterations < 2^31` | `The sample is too large for the GPU kernels` |
| The CPU plan and the GPU buffers together fit `memory_budget_bytes` | `CPU and GPU buffers together exceed the memory budget` |

On Apple silicon GPU memory is host memory, so GPU buffers count against the budget, in whole 16 KiB pages.

**No silent CPU fallback.** The Vx runtime runs a kernel's CPU copy when Metal is off, and Vx built without the CUDA toolkit runs "GPU" kernels on the CPU. Neither tells the caller. VxVector allocates GPU memory first, which fails without a device, and [build_gpu_cuda.sh](../scripts/build_gpu_cuda.sh) refuses any dispatch library but `libvx_cuda_dispatch.so`.

**No CPU rerun after a GPU failure.** A GPU that fails after training starts is reported, and no centroids are returned. The caller asked for, or was told about, the GPU.

## Numeric policy

The GPU follows `run_restart` in [src/kmeans.vx](../src/kmeans.vx) on samples that the data path has already chosen, in sample order.

| Item | Policy | Same as the CPU? |
| --- | --- | --- |
| Seeds | Center `c` of restart `r` starts at sample `(c * n / k + r * 104729) % n` | Yes |
| Distance | L2, inner product, or cosine, added in **float32** in dimension order | No. The CPU adds in float64. Apple GPUs have no float64 |
| Cosine | `1 - dot / (sqrt(left) * sqrt(right))`, 0 for two zero vectors, 1 when one is zero | Same rules. `sqrt(left * right)` would overflow float32 |
| Ties | The lower center index wins, with a strict `<` | Yes |
| Centroid sums | float32, in sample order. One block owns a center, one lane a dimension | Yes, the same additions in the same order |
| Empty cluster | Center `c` takes sample `(c + iteration) % n` | Yes |
| Sum overflow | A non-finite sum sets a flag, the run finishes, and the status is 1 | Same status, no centroids |
| Spherical centers | Normalized on the host in float64 after seeding and after each update, by `vxvector_gpu_normalize_rows`, built with `-ffp-contract=off` | Yes. The build checks for fused multiply-adds |
| Restarts | Each restart on the GPU. The data path keeps the first, then any with a strictly lower float64 objective | Yes, where labels are the float64 nearest centers |
| Roots of a two-level tree | Trained on the CPU from the GPU's leaves | Yes |
| Subnormals | The GPU flushes float32 subnormals to zero | No. Not tested |

**No atomics.** Metal has no float atomics. Each sum has one owner, and Vx's `across` checker proves that no two lanes write the same element.

**Spherical normalization is in C.** A Vx function cannot read GPU memory, and `vxc --action emit-obj` resolves `extern` symbols when it compiles. So Vx gets the C function as a function pointer. vxc does not wait for the GPU before an indirect call ([issue draft 8](vx_issue_drafts_2026-11-14.md)), so the C function waits itself, then copies the k centers to the host and back. On Metal the copies are `memcpy` of shared memory.

## The tolerance

float32 distances can choose another center than float64 at a near-tie, and the runs then diverge. Nearly parallel vectors give cosine distances about 1e-8 apart, below float32's resolution near 1.

A GPU result passes if it has the CPU's bits. Otherwise it must meet both:

1. Every GPU label is the float64 nearest center of the GPU's own centers, except where the float64 gap is below 1e-5 of the distance, or 1e-6 for cosine.
2. The float64 objective of the GPU's centers is within 1e-3 of the CPU's. The largest difference seen was 3.5e-4.

Only runs known to meet near-ties may use it: `cosine_near_ties`, `sift_8192/spherical`, and cosine in the worker test. Every other run must be exact, because exact runs are what catch mutations. A matching objective alone is never enough: equal-cost centroids in another order hide a wrong tie rule.

## Launch size

On a Mac the GPU also draws the screen. macOS can stop a launch that holds it too long, and the Vx runtime then aborts the process. Launches of about 75 ms failed in 5 of 7 runs. The same work in eight launches failed in none: macOS judges each launch, not the batch.

So every launch does at most `launch_work` multiply-adds: 2^26 on Metal, about 2 ms on an M2 Pro, and no limit on CUDA.
Assignment covers up to 32,768 rows and as many centers as fit, at least eight. The update adds a range of rows into the sums. Each distance and sum adds the same numbers in the same order as one launch would, so the split changes no bits. The checks run every case at the default and at a tiny budget, and both must match.

Short launches make an abort unlikely, not impossible. [Issue drafts](vx_issue_drafts_2026-11-14.md) 1 and 2 ask the runtime to retry and to split launches itself.

## Memory limits

The kernels take tensors of shape `[?, ?]`, and Vx checks capacity only for shapes it knows when it compiles (warning W1029). Bounded dynamic shapes, [vx-lang/Vx#245](https://github.com/vx-lang/Vx/issues/245), would let it check them. Until then the limits are the C checks, the backend's allocator, and the memory budget.

## Open questions

- Does the Vx CUDA runtime run launches in order, each finishing before the next reads its results, as Metal does?
- A Vx runtime abort kills the worker. Check that every such crash reaches SQL as a failed GPU run.
- The unavailable-device reason, `No Metal device, or Metal is turned off`, names Metal on CUDA builds too.
- A float64 normalize kernel in a CUDA-only Vx file would remove the host copy for spherical centers.
- Tune the `across lanes` widths for the A100, and stage centers in threadgroup memory on Metal.
