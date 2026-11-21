# VxVector on NVIDIA: test plan

2026-11-20. Shared copy: [VxVector on NVIDIA: test plan](https://claude.ai/artifact/2uN4SEdVsxvecesnqGi6HW).

## What to test (short version)

**Use the H200.** Vx has a machine file for it (`fleet/h200.vx`), and it's the closest match to what pgpu users run. An RTX 6000 Ada or RTX PRO 6000 (Blackwell) is a fine second card, but Vx has no machine file for it: build with the A100 file and rely on PTX JIT. The older Quadro RTX 6000 (Turing) won't work.

1. **Does it run, and does it match the CPU?** Build the CUDA library and run the GPU-vs-CPU check. L2 and inner product must be bit-identical. Cosine must be within 1e-3 of the CPU objective.
2. **Does it help a real index build?** On 1M-row datasets (SIFT-128, GIST-960, a 768-dim and a 1536-dim embedding set), time `CREATE INDEX` three ways:
    - VectorChord's own trainer, all cores
    - VxVector on CPU
    - VxVector on the GPU

   Report total time, time spent training, and recall@10/QPS at a few probe settings.
3. **How do we compare with pgpu?** Run the same builds with pgpu on the same GPU, and compare time, recall and centroid quality.

Stop after step 1 if it fails. The rest of this doc says how to do each step and what to record.

## Purpose

This run answers three questions, in order. Each phase gates the next: stop and report if one fails.

1. **Does the CUDA path run and match the CPU?** It compiles today but has never run on an NVIDIA GPU.
2. **Does faster training make index builds meaningfully faster?** We need the share of `CREATE INDEX` time spent in training, at realistic scale, against VectorChord's own trainer.
3. **Where do we stand against pgpu/cuVS** on the same GPU, at matched recall?

Bring back three things: the filled results tables (last section), the raw JSON and logs, and a short list of anything that failed or needed a workaround. A clear negative result is as useful as a positive one. Don't tune things to make numbers look good.

Repo: [minzyfox/vextor](https://github.com/minzyfox/vextor), branch `main`. Vx is pinned to the commit in `vx-revision.txt` (`889a1fbb`).

## Machine and environment

Record all of this in `results/env.txt` before building anything. Every number we publish needs it.

- GPU model, memory, and compute capability: `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv`
- CUDA toolkit version (`nvcc --version`), driver version, and whether a display is attached to the GPU (the display watchdog can kill long kernels)
- CPU model, physical cores, RAM (`lscpu`, `free -g`), OS and kernel (`uname -a`)
- Git SHAs of vextor, Vx, VectorChord, pgvector, and pgpu
- PostgreSQL version and the non-default `postgresql.conf` settings used

**Check the GPU architecture first.** The CUDA build defaults to the Vx machine file `fleet/a100-40.vx`, which emits sm_80 PTX. Vx also ships `a100-80`, `h100-pcie`, `h100-sxm`, `h200` and `b200`. Set `VXVECTOR_GPU_MACHINE` to the matching file. sm_80 PTX should JIT on newer cards (Ampere consumer, Ada, Hopper) but will not run on Turing (T4, RTX 20xx) or older. If no machine file matches, stop and report; don't edit one to make it work.

## Setup

Everything runs natively on the Linux host. The Docker PostgreSQL image has no GPU library, so it can only serve as the CPU reference. Native Linux PostgreSQL with the GPU has never been set up before. Keep every command you run in `scripts/setup_linux_cuda.sh` so we can commit it.

1. **Check out both repos** as in the README's "Get VxVector and the pinned Vx source" section. `VX_ROOT` must be a clean checkout at `889a1fbb`.
2. **Build Vx with the CUDA toolkit installed.** Follow "Prepare the native CPU tests" (LLVM/MLIR 22, Rust, Z3, `VX_VERSION_WITH_COMMIT=1`). Vx builds `libvx_cuda_dispatch.so` only when it finds the toolkit at build time. Without it, Vx quietly falls back to a CPU dispatch library.
3. **Build the CPU library and run the CPU suite:** `./scripts/build.sh`, create `build/correctness-venv`, then `bash scripts/check_cpu.sh`. It must pass before any GPU work.
4. **Build the CUDA library:**

   ```sh
   CUDA_HOME=/usr/local/cuda VXVECTOR_GPU_MACHINE=$VX_ROOT/fleet/<your-gpu>.vx ./scripts/build_gpu_cuda.sh
   ```

   This writes `build/linux/libvxvector_gpu.so`. It refuses a compiler built without the CUDA dispatch library. That refusal is correct: fix the Vx build, not the script.
5. **Point the worker at it.** Copy the `.so` beside `build/vxvector-worker`, or set `VXVECTOR_GPU_LIBRARY` to its absolute path.
6. **PostgreSQL 17 on the host:** pgvector 0.8.2 and VectorChord 1.1.1 (the versions in the image), with `shared_preload_libraries = 'vchord'`. Build `pg_vxvector` against the same `pg_config`, adapting `scripts/build_macos_extension.sh` (`.so` instead of `.dylib`). Install it, set `vxvector.worker_path` to the absolute path of `build/vxvector-worker`, and restart.
7. **pgpu**, only for Phase 3: install from [EnterpriseDB/pgpu](https://github.com/EnterpriseDB/pgpu) against the same PostgreSQL. Record its commit and its cuVS version.
8. **Datasets.** Download SIFT-128 and GloVe-25 as in the README and check them against `tests/datasets.sha256`. For scale, see Phase 2.

## Phase 1: CUDA correctness

The goal is for the CUDA library to match the CPU kernel under the same rules the Metal library passed. Plan on half a day if the build works.

1. **Smoke test.** Run `scripts/test_gpu_lloyd_cuda.sh`, the older Driver API prototype. It shows the toolkit, driver and GPU work together.
2. **Port the GPU check to CUDA.** `scripts/check_gpu_l2.py` is the 97-run gate used for Metal, but it hard-codes `.dylib` paths and parses the Metal dispatch log. Make it platform-aware: load `build/libvxvector_kmeans.so` and `build/linux/libvxvector_gpu.so`. Find out what `VX_DISPATCH_VERBOSE=1` prints with the CUDA dispatch library. If it prints nothing useful, use `nsys profile` instead to prove every kernel ran on the GPU.
3. **Run it with SIFT downloaded**, so the SIFT cases aren't skipped.
4. **Run `bash scripts/check_cpu.sh` again.** With the real GPU library present, the three real-GPU worker tests should run instead of being skipped.
5. **Worker behavior:**
   - `auto` selects CUDA and reports the device.
   - `gpu` with `CUDA_VISIBLE_DEVICES=""` fails cleanly and reports why.
   - `cpu` never touches the GPU.
   - Killing the worker mid-training leaves PostgreSQL running.
6. **End-to-end SQL.** Run the README walkthrough, then `check_real_recall.py` on SIFT-128 and GloVe-25 with `vxvector.backend_mode = 'gpu'`. The returned JSON must say `backend: gpu`.

**Pass criteria** (from `docs/gpu_design.md`):

- L2 and inner product: centroids bit-identical to the CPU in every case.
- Cosine and spherical centroids: the objective is within 1e-3 of the CPU's. Report the largest gap seen (on Metal it was 3.5e-4).
- Every kernel shows up as having run on the GPU.
- The recall gates pass.

**If anything fails, stop.** Record the case, the inputs and the first mismatching index. Don't loosen tolerances or change kernels to get a pass.

## Phase 2: scale baseline and time breakdown

This phase finds out how much of `CREATE INDEX` time goes to training, and how we compare with VectorChord's own multi-threaded trainer. It decides what we work on next, so it matters most.

**Datasets.** Start with the smallest and move down the list as time allows. Record each one's source URL and sha256.

| Dataset | Rows | Dims | Metric | Lists to try |
| --- | --- | --- | --- | --- |
| SIFT-128 (ann-benchmarks) | 1M | 128 | L2 | `[1000]`, `[4000]` |
| GIST-960 (ann-benchmarks) | 1M | 960 | L2 | `[1000]`, `[4000]` |
| Cohere 768 (VectorDBBench) | 1M | 768 | cos | `[1000]`, `[4000]` |
| OpenAI 1536 (VectorDBBench or dbpedia-openai) | 1M | 1536 | cos | `[1000]`, `[4000]` |
| Cohere 768 (VectorDBBench) | 10M | 768 | cos | `[10000]`, `[100, 10000]` |

**Fixed training settings** for every configuration: `sampling_factor = 256`, `kmeans_iterations = 10`, `kmeans_nredo = 1`, one seed. Raise `memory_budget_bytes` above the sample size (lists × 256 × dims × 4 bytes). Raise `worker_timeout_ms` and `statement_timeout` too.

**Three ways to build each index:**

- **A. VectorChord internal:** `CREATE INDEX ... USING vchordrq` with `[build.internal]` set to the same lists, sampling factor and iterations, and `build_threads` = physical cores. Check the option names against the VectorChord 1.1.1 docs.
- **B. VxVector on CPU:** `vxvector.backend_mode = 'cpu'`. Today this is one thread. Cap each run at 2 hours and record "> 2 h" if it hits the cap.
- **C. VxVector on CUDA:** `vxvector.backend_mode = 'gpu'`.

**What to measure for each build:**

- Total `CREATE INDEX` wall time.
- For B and C, the stage timings in the returned JSON: read/validate, sampling, training, GPU allocation/copy, and index build after centroids. Also run once with `skip_index_build => true` to time training alone.
- For A, sample `pg_stat_progress_create_index` every second to split training from the rest, if VectorChord reports phases there. If it doesn't, say so.
- Peak RSS of the backend and worker. For C, peak GPU memory from `nvidia-smi --query-gpu=memory.used --format=csv -lms 200`.

**Index quality** for each built index:

- Use the dataset's held-out queries (1,000 if available) and exact ground truth.
- Measure recall@10, single-client QPS, and p50/p99 latency.
- Sweep `vchordrq.probes` at about 1%, 2%, 5% and 10% of lists. Two-level indexes need two probe values.
- Keep `vchordrq.epsilon` at its default.

A and C should give about the same recall. If they don't, that's a finding in itself: report it, and don't tune it away.

**The answer we need from this phase:** for each dataset and lists setting, the share of A's build time spent in training, and C's end-to-end speedup over A.

## Phase 3: pgpu/cuVS head-to-head

This phase shows whether we're at parity with pgpu on its home hardware. We don't need to win here; parity plus portability is the story. Run it only after Phase 2 is done for at least the 1M datasets.

- **D. pgpu:** `pgpu.create_vector_index_on_gpu(...)` with the same table, lists, `sampling_factor`, `kmeans_iterations`, `kmeans_nredo`, distance operator and spherical setting as C. Our SQL function copies its signature, so the arguments should line up one to one.
- Measure the same things as in Phase 2: total wall time, training versus index build, peak GPU and host memory, then recall@10 and QPS on the same probe sweep.
- **Compare the centroid tables.** Same schema? Same root/parent layout? Compute the k-means objective of each centroid set on the same sample with `scripts/objective_comparison.py` or a small NumPy script, so we compare centroid quality, not just speed.
- **Settle an open question from our docs:** does pgpu only train centroids and then call VectorChord's `build.external` (as its README suggests), or does more of the build run on the GPU (as EDB's docs suggest)? Answer from its source and from a profile (`nsys`), and write down which.
- **Optional:** a direct cuVS k-means harness on the same sampled array, plus the same sample through `vxvector_kmeans_v2` and the CUDA library. This compares kernels alone, without database reads and writes.

Record pgpu's settings that differ from ours: initialization, sampling, empty-cluster handling and stopping rule. Don't assume the same seed means the same start.

## Fair-measurement rules

- Same machine, same PostgreSQL instance, same table and same snapshot for A, B, C and D. Nothing else heavy runs during a measurement.
- Run each configuration 3 times and report the median. Note the spread if the slowest run is more than 10% slower than the fastest.
- Before timed runs, warm the table into the OS cache with one untimed `SELECT count(*)`. Drop the index and the centroid table between runs.
- Same probes and epsilon when comparing recall. Same query set for every index. Never compare QPS at different recall levels.
- Report the settings that differ between systems instead of hiding them: threads, sampling, initialization, iterations.
- Every number is labelled with dataset, rows, dims, lists, backend, and the git SHAs from the environment file.
- Keep raw outputs, including the full JSON from `create_vector_index` and logs, under `results/<dataset>/<config>/run<N>/`.

## Results template

Fill in one row per run configuration, using medians of 3 runs. Add rows as needed.

**Phase 1: correctness**

| Check | Result | Notes |
| --- | --- | --- |
| CUDA smoke test |  |  |
| check_gpu_l2 (CUDA): L2/IP bit-identical |  |  |
| check_gpu_l2 (CUDA): largest cos/spherical objective gap |  |  |
| Every kernel seen on the GPU |  |  |
| check_cpu.sh with real GPU |  |  |
| Worker auto / gpu / cpu / kill |  |  |
| check_real_recall (gpu) SIFT, GloVe |  |  |

**Phase 2 and 3: builds**

| Dataset | Lists | Config | Total build (s) | Training (s) | Index after centroids (s) | Peak host RAM (GB) | Peak GPU mem (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT-1M | [1000] | A VectorChord internal |  |  |  |  |  |
| SIFT-1M | [1000] | B VxVector CPU |  |  |  |  |  |
| SIFT-1M | [1000] | C VxVector CUDA |  |  |  |  |  |
| SIFT-1M | [1000] | D pgpu |  |  |  |  |  |

**Phase 2 and 3: index quality**

| Dataset | Lists | Config | Probes | Recall@10 | QPS | p50 (ms) | p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT-1M | [1000] | A |  |  |  |  |  |

**Summary:** three to five sentences. What share of build time is training? What is C's end-to-end speedup over A, and how does C compare with D? Did anything surprise you?

## Notes for the agent

**Allowed:**

- New scripts for setup, benchmarking and recall measurement.
- Making `check_gpu_l2.py` and other scripts work on Linux/CUDA (paths, library suffix, dispatch-log parsing).
- Fixing build-script issues that are clearly platform-specific.

Keep each change small and on a branch named `nvidia-validation`.

**Not allowed:**

- Editing `src/gpu_kmeans.vx`, `src/kmeans.vx`, `src/gpu_l2.c`, the tolerances in `docs/gpu_design.md`, or the Vx pin.
- Optimizing anything to improve a benchmark.
- Tuning VectorChord or pgpu differently from what this plan says.

If a change looks necessary, write it down and ask.

**Stop and ask when:**

- `build_gpu_cuda.sh` refuses the dispatch library and rebuilding Vx with the toolkit doesn't fix it.
- No Vx machine file matches the GPU.
- Any Phase 1 correctness check fails.
- The GPU or host runs out of memory at a dataset size.
- A run is likely to take more than about 2 hours.

**Vx compiler bugs:** if you hit one, write a minimal reproduction in the style of `docs/vx_issue_drafts_2026-11-14.md`. Don't work around it silently.

**At the end:** commit the scripts and the filled results tables to `nvidia-validation` and open a PR. Raw results stay out of git unless they are small.
