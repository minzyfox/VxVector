# Step 3: L2 clustering on Apple Metal (November 14, 2026)

This record completes Step 3 of the [active roadmap](pg_vxvector.md).
One restart of L2 Lloyd training runs on the Apple GPU through Metal. Every input below gives the same bits as the CPU kernel and an independent reference.
It is a standalone macOS result. Nothing uses it yet: the worker, the data path, and PostgreSQL are unchanged. Step 4 wires in backend selection.

The first version ran each assignment as one long launch, and macOS stopped some of them. The second version, described here, splits the work into launches of at most about 2 ms. It also gives each lane eight centers at a time. It still matches the CPU bit for bit, and it trains 24 to 28 times faster than a single-threaded `-O3` CPU build. See [Size limit](#size-limit-macos-can-stop-long-launches) and [Performance](#performance).

## Revisions

| Item | Value |
| --- | --- |
| Vx | `423a10dbbd91230df290494f9705a746dc444baf`, the current pin |
| Compiler | `vxc 0.0.4 (423a10db)`, debug profile, SHA-256 `18889bbc…` |
| Target | `--host default --machine fleet/m4-uma.vx`, the only Apple machine file in Vx |
| Device | Apple M2 Pro, macOS 27.2.1 |
| Metal runtime | `libvx_npu_dispatch.dylib` of that compiler. Shaders compile with `MTLMathModeSafe` |

The [metadata file](evidence/step3_metadata_2026-11-14.json) lists every hash, including the sources, the library, and the generated Metal source.
The kernels target the M4 machine file and run on an M2 Pro. The generated kernels use no M4-only features, and the results below are from the M2 Pro.

## What was built

| File | Role |
| --- | --- |
| [src/gpu_kmeans.vx](../src/gpu_kmeans.vx) | Five Apple GPU kernels (seed, assign, zero, accumulate, finish) and the Lloyd loop that launches them in short pieces |
| [src/gpu_l2.c](../src/gpu_l2.c) and [include/vxvector_gpu.h](../include/vxvector_gpu.h) | `vxvector_gpu_l2_v1`: argument checks, GPU memory, the tensor ABI, the overflow status, and `launch_work`, the bound on each launch |
| [scripts/build_gpu.sh](../scripts/build_gpu.sh) | Builds `build/gpu/libvxvector_gpu.dylib`. Run it under `arch -arm64` |
| [scripts/check_gpu_l2.py](../scripts/check_gpu_l2.py) | The checks below |

The C file holds what Vx cannot do from a function that C calls: allocate GPU memory with `vx_metal_alloc` and pass tensors in the MLIR descriptor ABI. This is the same kind of exception as the allocation in `src/data_path.c`.

## GPU policy

The GPU follows `run_restart` in [src/kmeans.vx](../src/kmeans.vx) with restarts=1, L2, no spherical normalization, and no roots. It trains on samples that the caller has already chosen, in sample order.

| Item | Policy | Same as the CPU? |
| --- | --- | --- |
| Seeds | Center `c` starts at sample `c * n / k` | Yes. This is `spaced_position` for restart 0 |
| Distance | Squared L2, added in **float32** in dimension order | No. The CPU adds in float64. Apple GPUs have no float64 |
| Ties | The lower center index wins, using a strict `<` | Yes |
| Centroid sums | float32, added in sample order. One block owns a center, and one lane owns a dimension | Yes, the same additions in the same order |
| Mean | `sum / count` in float32. Metal's safe math mode divides in IEEE | Yes |
| Empty cluster | Center `c` takes sample `(c + iteration) % n` | Yes |
| Sum overflow | A sum that is not finite sets a flag. The run finishes, and the function returns status 1 | The CPU stops at once with status 1. The status is the same, and no centroids are returned |
| Non-finite input | Not accepted. Inputs must be finite, as on the CPU | Yes |
| Subnormals | The GPU flushes float32 subnormals to zero | No. Not tested here |
| Iterations | Fixed, at least 1, with no early stop | Yes |

**The one difference is float32 distances.** At a near-tie that float32 cannot separate, the GPU may choose another center than the CPU. The two runs then diverge.
An exact tie (equal distances) goes to the lower index on both, because both distances are equal in float32 and in float64.

**How the gates use this.** The inputs below have no float32 near-ties, so every case must match the CPU kernel and the NumPy reference **bit for bit**: centroids, labels, and objective.
A matching objective alone is not accepted. An early draft accepted it, and a wrong tie rule passed, because equal-cost centroids in another order hide it.
The tolerance for real near-ties belongs in Step 5, which compares across larger and messier inputs.

## Parallelism, races, and memory

- **Launches.** Each launch does at most about `launch_work` multiply-adds, 2^26 by default.
  - Assignment covers up to 32,768 rows, one per lane in blocks of 256, and as many centers as fit, at least 8. Each lane compares eight centers at a time: it reads each sample value once and updates eight distances. `labels` and `best` carry the nearest center and its distance from one range of centers to the next.
  - The update has three kernels. `zero` clears the sums, `accumulate` adds a range of rows, and `finish` divides or replaces. Each of these runs `k` blocks with lanes over dimensions.
  - Every sum and distance adds the same numbers in the same order as a single launch would, so the split changes no bits. The checks run each case at the default and at a tiny budget of up to 26,017 launches, and both must match.
- **No races, checked by the compiler.** Vx's `across` checker rejects a lane loop when two passes may write the same element. It rejected the first draft, which had one overflow flag per center (`E6032`). The flag is now per element. No kernel uses atomics. Metal has no float atomics, so sums go through ownership instead.
- **Resident memory.** C allocates the seven tensors with `vx_metal_alloc`: samples, centers, labels, best distances, sums, counts, and overflow flags. Metal wraps them in place (`newBufferWithBytesNoCopy`). The samples are copied in once. Every launch then runs on the same GPU buffers. The log shows seven buffers made per run and no copied tensors.
- **Completion.** The Vx function waits for the GPU before it returns, so C reads the results after the work ends. `vx_metal_free` waits for any work on a buffer before it releases it. The log shows the wait after each launch and each free.
- **No silent CPU fallback.** When Metal is off, the Vx runtime would run each kernel's CPU copy without saying so. `vxvector_gpu_l2_v1` refuses instead: `vx_metal_alloc` returns NULL, and the function returns `VXVECTOR_GPU_UNAVAILABLE`. A check runs with `VX_METAL=0` and requires that status.

## Results

[scripts/check_gpu_l2.py](../scripts/check_gpu_l2.py) runs each case in a worker process with `VX_DISPATCH_VERBOSE=1`. It compares the GPU with three things:
- the CPU kernel `vxvector_kmeans_v2` on the same rows, in order, with restarts=1;
- an independent NumPy implementation of the policy, with float64 distances and sequential float32 sums;
- float64 nearest-center labels and objectives.

It also requires the number of Metal launches that the launch plan predicts, no copied tensor, and no CPU fallback.
Each case runs twice: at the default `launch_work`, and split into many short launches (4,096 multiply-adds, or 2^20 for SIFT). Both runs must give the same bits.
[Results](evidence/step3_gpu_l2_2026-11-14.jsonl), [Metal dispatch summary](evidence/step3_metal_dispatch_2026-11-14.txt), and [generated Metal source](evidence/step3_gpu_kmeans_2026-11-14.metal).

| Case | n | dims | k | Iterations | Launches: default, split | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `blobs` | 4,096 | 32 | 16 | 10 | 42, 10,773 | Same bits as the CPU and the reference, both runs |
| `wide_rows` (dims above 256 lanes) | 1,000 | 300 | 10 | 8 | 34, 26,017 | Same bits |
| `one_past_a_block` | 257 | 3 | 7 | 6 | 26, 39 | Same bits |
| `uniform` | 2,048 | 17 | 33 | 12 | 50, 8,026 | Same bits |
| `empty_clusters` (exact ties) | 64 | 4 | 4 | 5 | 22, 22 | Same bits |
| `empty_replacement` | 64 | 4 | 4 | 5 | 22, 22 | Same bits |
| `group_ties` (exact ties inside groups of eight) | 256 | 4 | 16 | 5 | 22, 55 | Same bits |
| `one_center` | 100 | 8 | 1 | 4 | 18, 18 | Same bits |
| `every_sample_a_center` | 9 | 5 | 9 | 3 | 14, 14 | Same bits |
| **`sift_8192`, real SIFT-128 vectors** | 8,192 | 128 | 64 | 10 | 42, 1,365 | **Same bits** |
| `no_iterations` | 300 | 6 | 5 | 0 | none | Rejected, as on the CPU |
| `sum_overflow` (3e38 values) | 4 | 2 | 1 | 2 | 10, 10 | Status 1, as on the CPU |
| `metal_disabled` (`VX_METAL=0`) | 64 | 4 | 4 | 2 | none | `VXVECTOR_GPU_UNAVAILABLE`, and no CPU fallback |

In every training case, every label is the float64 nearest center.

### Mutations

Each mutation of [src/gpu_kmeans.vx](../src/gpu_kmeans.vx) must fail the check ([log](evidence/step3_mutations_2026-11-14.jsonl)):

| Mutation | Caught by |
| --- | --- |
| A tie inside a group of eight goes to the higher index (`<=`) | `group_ties`, both runs |
| A tie in the remainder loop goes to the higher index | `empty_clusters`, `empty_replacement`, both runs |
| An empty cluster takes sample `c + iteration + 1` | `empty_replacement`, both runs |
| Seeds move by one sample | 14 runs |
| `accumulate` skips the last row of its range | 18 runs |
| Assignment drops the carried nearest center for `first_center` up to 8 | 4 split runs |
| The overflow flag is never set | `sum_overflow`, both runs |

The first draft of the check missed two mutations. It allowed any difference with a matching objective, and every replacement row in `empty_clusters` was equal. The check now requires exact bits, and `empty_replacement` has distinct replacement rows.
The groups of eight added a third miss: no tie fell inside a group until `group_ties` was added.

## Vx findings

1. **`f32::is_finite` has no `ensures writes_only()`,** so an `across` loop cannot call it (`E6032`). The kernel uses its own `finite_f32`, which tests `x - x == 0.0`. An issue is not filed yet.
2. **No stable path to the Metal runtime.** `libvx_npu_dispatch.dylib` exists only under `target/debug/build/vxc-<hash>/out/`, and several hashes remain after rebuilds. `scripts/build_gpu.sh` reads the path that `vxc` itself embeds.
3. **Silent CPU fallback.** With Metal off, a kernel runs its CPU copy and reports nothing to the caller. Allocation through `vx_metal_alloc` is the only check available to a library.
4. **A device helper's symbol is global.** The first draft named its helper `finite`, which became `_finite` and collides with the C library. Vx emits private helpers as global symbols.

## Size limit: macOS can stop long launches

This section was added after a timing run on November 14, 2026.

The display also uses the GPU. macOS can stop a launch that holds the GPU too long, with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. The Vx runtime then prints `FATAL` and calls `abort()`, so the process that called the library dies.
It happens at random, more often as launches get longer. Each assignment launch handles every sample, and each lane loops over all `k * dims` center values:

| SIFT-128 input | Longest assignment launch | Outcome |
| --- | --- | --- |
| 8,192 rows, k=64 | 10 ms | passed |
| 16,384 rows, k=192 | 34 ms | passed |
| 32,768 rows, k=256, in three runs | 81 to 83 ms | passed twice, stopped once |
| 65,536 rows, k=256 | 156 ms | passed once |

A standalone Vx program with launches of about 75 ms failed in 5 of 7 runs. The same work split into eight launches failed in none of 5, although the batch still took 74 ms. So macOS judges each launch, not the batch.
**The fix.** The second version splits every launch so that it does at most `launch_work` multiply-adds, as [Parallelism](#parallelism-races-and-memory) describes. With the default of 2^26, no launch took more than 3 ms. Sixteen stress runs at the sizes that failed, and larger ones, all passed ([log](evidence/step3_performance_2026-11-14.txt) and earlier runs):

| SIFT-128 input, 10 iterations | Runs | Launches per run | Longest kernel | Outcome |
| --- | --- | --- | --- | --- |
| 32,768 rows, k=256 | 5 | 357 | 2.4 ms | all passed |
| 65,536 rows, k=1,024 | 13 | 2,709 | 2.0 ms | all passed |
| 262,144 rows, k=1,024 | 1 | 10,773 | 1.9 ms | passed |

Short launches make an abort much less likely, not impossible. macOS's rule is not documented, and the Vx runtime still aborts if it stops one. The [Vx issue drafts](vx_issue_drafts_2026-11-14.md), numbers 1 and 2, ask the runtime to retry and to split launches.

## Performance

[Log](evidence/step3_performance_2026-11-14.txt). SIFT-128 vectors, 10 iterations, one restart. The CPU kernel is built with `vxc -O3` outside the project build, which uses `-O0`, Vx's default. Both builds give the same bits, and so does the GPU.

| Input | GPU | CPU `-O3` | Speed-up | CPU `-O0` |
| --- | --- | --- | --- | --- |
| 8,192 rows, k=64 | 0.03 to 0.06 s | 0.44 s | 8x to 14x | 2.70 s |
| 32,768 rows, k=256 | 0.29 s | 6.94 s | 24x | not run |
| 65,536 rows, k=1,024 | 1.99 s | 54.83 s | 28x | not run |

Limits of this comparison:
- The CPU kernel uses one thread, and the M2 Pro has 10 CPU cores. A parallel CPU kernel would narrow the gap, perhaps to a few times. Step 7 has the optimized CPU baseline.
- The CPU adds distances in float64, and the GPU in float32.
- The GPU still waits after every launch. At 65,536 rows and k=1,024, it was busy for 1.5 of 2.07 seconds.
- 65,536 rows and k=1,024 add about 9.5 × 10^10 distance terms (11 assignments × n × k × dims) in 1.99 s, about 47 billion a second. The M2 Pro GPU can do roughly 3.4 trillion fused multiply-adds a second. Centers are not staged in threadgroup memory yet.

An earlier figure of 36x compared the first GPU version with the `-O0` CPU build. It was not a fair comparison.

## Memory is not checked at compile time

The kernels take tensors of shape `[?, ?]`. Vx checks capacity only for shapes that it knows when it compiles, so every parameter gets warning W1029 (`capacity of 'HBM' not verified`), and no machine file changes that.
A 20 GiB tensor of fixed shape is rejected against `machines/m2-pro-16g.vx` (E6009) and accepted against `fleet/m4-uma.vx`. The checks would need bounded dynamic shapes, [vx-lang/Vx#245](https://github.com/vx-lang/Vx/issues/245), which is open.
The only limits now are the C checks (`n * dims < 2^31`) and `vx_metal_alloc`, which fails above Metal's largest buffer. Step 4 needs a memory budget for the GPU, as the CPU plan has.

## Not covered

- **Performance tuning.** The figures above are first measurements on one machine, with no tuning of `launch_work`. The `accumulate` kernel reads every label once per center and dimension, O(k × n × dims) label reads. Step 7 covers tuning.
- **float32 near-ties.** No input exercises a near-tie, so a GPU and CPU divergence is not tested. Step 5 must define its tolerance.
- **Other metrics, normalization, restarts, two-level trees, and sampling on the GPU.** These are Step 5.
- **Limits.** `n * dims` and `k + iterations` must stay below 2^31, because the kernels index with `i32`. Allocation-failure behavior is not tested.
- **Integration.** Nothing calls the GPU library yet. Backend selection is Step 4.
- **Contracts.** `src/gpu_kmeans.vx` has no `requires` or `ensures`. Its index safety rests on the C checks and on the tensor bounds that Vx knows.
