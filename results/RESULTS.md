# VxVector on NVIDIA: results

Plan: [docs/nvidia_test_plan.md](https://github.com/minzyfox/VxVector/blob/nvidia-test-plan/docs%2Fnvidia_test_plan.md).
Machine and revisions: [env.txt](env.txt). One NVIDIA H200 (sm_90, driver 580.196.07,
CUDA 13.0.88), 8 physical cores (Xeon Platinum 8468), 196 GB RAM; Nebius `gpu-h200-sxm`,
preemptible, eu-west1. vextor `6074e08` plus the scripts on branch `nvidia-validation`;
Vx `889a1fbb`; VectorChord 1.1.1; pgvector 0.8.2; PostgreSQL 17.12. Run on 2026-11-21.

## Summary

Phase 1 passed. The CUDA library, built for `fleet/h200.vx`, matches the CPU kernel bit for
bit for L2 and inner product in every run outside the declared near-tie runs. Cosine and
spherical runs stay within 3.49e-4 of the CPU objective. Every kernel ran on the GPU.

In Phase 2, training is 12-19% of VectorChord's own build time at `[1000]` lists and 38-55%
at `[4000]`. The rest is VectorChord's index build after centroids, which takes the same
time whichever trainer made the centroids. C (VxVector on CUDA) is 1.26-1.37x faster than A
(VectorChord's trainer on 8 cores) end to end at `[4000]` on SIFT, GIST and Cohere. It is
slower at every `[1000]` setting (0.78-0.93x) and on OpenAI at `[4000]` (0.85x).

Two things cap C. First, moving the table to the worker costs 3-23 s, more than the GPU saves
at `[1000]`. Second, the GPU kernel runs at 0.7-0.8 T multiply-adds/s, a few percent of the
H200, while VectorChord's trainer skips most exact distances with RaBitQ bounds. At 1,536
dimensions that puts the two trainers level: 77.7 s on the GPU against 80.1 s on 8 cores.

Recall of A, B and C matches within about 0.01 on SIFT, GIST and OpenAI. The surprise is
Cohere: VxVector's centroids, from B and C alike, give 0.009-0.059 lower recall@10 than A's
at every probe setting. The likely cause is unit-length versus raw-scale centroids under
cosine (finding 4). Phase 3 (pgpu) was not run.

## Phase 1: correctness — passed

The CUDA library was built with `VXVECTOR_GPU_MACHINE=$VX_ROOT/fleet/h200.vx`
(`.target sm_90a`), against Vx's `libvx_cuda_dispatch.so`.

| Check | Result | Notes |
| --- | --- | --- |
| CUDA smoke test | Passed | "CUDA worker-library smoke test passed on NVIDIA H200." Needed `-Wno-error=deprecated-declarations` (workaround 4). `phase1/smoke.log` |
| check_gpu_l2 (CUDA): L2/IP bit-identical | Passed | 97 cases, 0 failures: 96 training runs, none skipped, plus `cuda_disabled`. L2 24/24 and IP 22/22 runs bit-identical to the CPU kernel and the NumPy reference. Spherical: L2 6/8, IP 7/8. The other 3 are the declared near-tie runs (`cosine_near_ties/spherical`, `cosine_near_ties/ip/spherical`, `sift_8192/spherical`). `sift_8192/spherical` diverged exactly as recorded on Metal: 152 labels, objective gap 1.94e-8. `phase1/check_gpu_l2.jsonl`, `phase1/check_gpu_l2_summary.json` |
| check_gpu_l2 (CUDA): largest cos/spherical objective gap | 3.49e-4 | `cosine_near_ties/cos/spherical` (Metal: 3.5e-4). Cosine 20/22 runs bit-identical, cosine/spherical 7/8. Every inexact run is a declared near-tie run, has 0 labels farther than float32 slack from the nearest center, and is within 1e-3. |
| Every kernel seen on the GPU | Passed | Vx log (`VX_DISPATCH_VERBOSE=1`): 277,413 launches "ran on GPU 0 from its own image", the expected count for every run, 0 host fallbacks. Cross-checked with `nsys` (the CUDA trace, not the Vx log): `blobs` 32,571 of 32,571 expected launches, `sift_8192` 4,347 of 4,347. `phase1/nsys_summary.txt` |
| check_cpu.sh with real GPU | Passed | 57 unit tests, 0 skipped (the 3 real-GPU tests ran). 48 Faiss, 96 Lloyd and 6 empty-cluster cases; data-path and sanitizer checks. One cosine near-tie in the worker test, objective gap +1.16e-4 (allowed). `phase1/check_cpu_with_gpu.log` |
| Worker auto / gpu / cpu / kill | Passed | auto: `backend gpu`, "GPU available and supports the request", device "NVIDIA H200", version "CUDA, Vx 889a1fbb". cpu: `backend cpu`; 80 samples at 100 ms during the 10.7 s build saw no CUDA context and no libcuda mapped in the worker. gpu with `CUDA_VISIBLE_DEVICES=""`: a clean `ERROR`, no output table, the server still answers; auto falls back to the CPU. The reason text says "No Metal device" (finding 5). kill -9 of the worker during GPU training: the statement fails ("VxVector worker closed the connection or failed"), the postmaster keeps its PID, no GPU process is left, no output table, and the next GPU build trains normally. `phase1/mode_*.txt`, `phase1/kill_*.txt` |
| check_real_recall (gpu) SIFT, GloVe | Passed | `vxvector.backend_mode = 'gpu'`: 24 builds (12 Vx, all `backend gpu` on the NVIDIA H200, and 12 internal), 3,072 recall queries, 24 setting comparisons; all gates pass. README walkthrough in auto mode: both builds on the GPU, 5 centroid nodes, 5 nearest rows from group-1, an index scan on `vxvector_demo_idx`. `phase1/check_real_recall_gpu.log`, `phase1/walkthrough.txt` |

## Phase 2: index builds at 1M rows

### Method

**Scope.** This was agreed for this run: the Nebius balance allowed about 17 hours of the
instance, and B alone at three runs would have needed about 40.

- The four 1M datasets, at lists `[1000]` and `[4000]`.
- A and C: three runs each, alternated A C, C A, A C, with every round kept.
- B: one run, at `[1000]` only. B at `[4000]` is an estimate from B's measured training rate.
- One C run with `skip_index_build => true`.
- Cohere-10M was not run.

**Datasets.** Ground truth is exact: the top 100 by Faiss `IndexFlat` in float32 over the
loaded rows (L2, or inner product on normalized vectors for cosine). Hashes:
`ann-benchmarks.sha256`, `cohere-768-1m.sha256`, `dbpedia-openai-1m.sha256`; details in
`prepare/`.

| Dataset | Rows | Dims | Metric | Source | Queries | Agreement with the source's top-10 |
| --- | --- | --- | --- | --- | --- | --- |
| sift-1m | 1,000,000 | 128 | L2 | ann-benchmarks `sift-128-euclidean.hdf5` | first 1,000 of `test` | 0.9991 |
| gist-1m | 1,000,000 | 960 | L2 | ann-benchmarks `gist-960-euclidean.hdf5` | 1,000 (`test`) | 0.9993 |
| cohere-1m | 1,000,000 | 768 | cos | VectorDBBench `cohere_medium_1m` (`train`, `test`, `neighbors.parquet`); vector norms 12.3-16.2 | 1,000 (`test`) | 1.0 |
| openai-1m | 999,000 | 1536 | cos | Hugging Face `KShivendu/dbpedia-entities-openai-1M`, column `openai` (26 files, each checked against its LFS sha256); unit vectors | 1,000 rows held out (NumPy `default_rng(42)`), not in the table | no source ground truth |

ann-benchmarks has no `dbpedia-openai-1000k-angular.hdf5` (it returns 404), and
VectorDBBench's OpenAI sets are 500K and 5M, hence the Hugging Face set.

**Training settings.**

- All configs: `sampling_factor = 256`, 10 iterations, one restart, Lloyd k-means,
  `spherical_centroids = false`, `residual_quantization = false`.
- VxVector (B, C): `seed = 42`, `memory_budget_bytes` = 16 GiB, `worker_timeout_ms` = 2 h
  for B (the cap) and 4 h for C.
- A: `[build.internal]` with `build_threads = 8`, the physical cores. VectorChord 1.1.1's
  defaults are already Lloyd, 256, 10 and non-spherical, so A differs from the defaults only
  in `lists` and `build_threads`.

**PostgreSQL settings.** `shared_buffers = 48GB` for every config (workaround 8); every other
setting is at its default. So `max_parallel_maintenance_workers = 2` applies to every index
build, including the index part of B and C.

**What was measured.**

- Build time: wall time of the `CREATE INDEX` or `vxvector.create_vector_index` statement,
  measured from the client.
- A's training time: the `initializing index, by internal build` phase of
  `pg_stat_progress_create_index`, sampled every 200 ms. B's and C's stages come from the
  timings the function returns.
- Memory: peak RSS (`VmHWM`) of the backend, its parallel workers and the VxVector worker,
  and the GPU's memory in use (`nvidia-smi -lms 200`), both sampled every 200 ms.

**Between builds.**

- Before each build: an untimed read of every row, including the TOASTed vectors
  (`sum(vector_dims(features))`; `count(*)` would not read them), and a check that no
  process is using the GPU.
- After each build: the index and centroid table are dropped and a `CHECKPOINT` runs.

**Quality measurement.**

- 1,000 queries from one client (psycopg, prepared statements, Python GC off).
- 100 untimed warm-up queries per probe setting.
- Probes at 1, 2, 5 and 10% of lists, with `vchordrq.epsilon` at its default.
- The plan of the first query must use the index.

Raw data is in `results/<dataset>/<config>/lists<L>/run<N>/`.

### Builds

Medians of 3 runs for A and C; B ran once. "Training" is A's internal-build phase, or
`clustering_seconds` for B and C. Peak host RAM is the sum of the backend's, its parallel
workers' and the VxVector worker's peak RSS, which includes the shared-buffer pages each
process touched. Peak GPU memory is the increase over the idle GPU.

| Dataset | Lists | Config | Total build (s) | Training (s) | Index after centroids (s) | Peak host RAM (GB) | Peak GPU mem (GB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift-1m | [1000] | A VectorChord internal | 18.4 | 2.3 | 16.1 | 2.81 | — | 3 run(s) |
| sift-1m | [1000] | B VxVector CPU | 306.4 | 287.4 | 15.9 | 2.80 | 0.00 | 1 run(s) |
| sift-1m | [1000] | C VxVector CUDA | 19.9 | 0.5 | 16.0 | 2.81 | 0.64 | 3 run(s) |
| sift-1m | [1000] | C VxVector CUDA, skip_index_build | 4.1 | 0.5 | 0.0 | 1.10 | 0.64 | 1 run(s) |
| sift-1m | [4000] | A VectorChord internal | 88.2 | 33.5 | 54.7 | 2.98 | — | 3 run(s) |
| sift-1m | [4000] | C VxVector CUDA | 64.6 | 6.7 | 54.0 | 2.96 | 1.01 | spread 10%; 3 run(s) |
| sift-1m | [4000] | C VxVector CUDA, skip_index_build | 10.1 | 6.7 | 0.0 | 1.50 | 1.01 | 1 run(s) |
| sift-1m | [4000] | B VxVector CPU | not run | ≈ 1.2 h (estimate) | | | | estimated from B's measured rate at [1000], 1.14 G multiply-adds/s |
| gist-1m | [1000] | A VectorChord internal | 44.8 | 7.7 | 37.1 | 19.65 | — | 3 run(s) |
| gist-1m | [1000] | B VxVector CPU | 2420.0 | 2369.7 | 36.8 | 23.57 | 0.00 | 1 run(s) |
| gist-1m | [1000] | C VxVector CUDA | 55.0 | 3.5 | 36.6 | 23.53 | 1.44 | 3 run(s) |
| gist-1m | [1000] | C VxVector CUDA, skip_index_build | 19.0 | 3.5 | 0.0 | 9.72 | 1.44 | 1 run(s) |
| gist-1m | [4000] | A VectorChord internal | 201.6 | 103.1 | 98.5 | 21.57 | — | 3 run(s) |
| gist-1m | [4000] | C VxVector CUDA | 159.8 | 48.4 | 94.9 | 23.85 | 4.15 | 3 run(s) |
| gist-1m | [4000] | C VxVector CUDA, skip_index_build | 64.2 | 48.4 | 0.0 | 12.64 | 4.15 | 1 run(s) |
| gist-1m | [4000] | B VxVector CPU | not run | ≈ 10.3 h (estimate) | | | | estimated from B's measured rate at [1000], 1.04 G multiply-adds/s |
| cohere-1m | [1000] | A VectorChord internal | 42.7 | 8.1 | 34.6 | 19.49 | — | 3 run(s) |
| cohere-1m | [1000] | B VxVector CPU | 2587.6 | 2536.6 | 38.2 | 23.37 | 0.00 | 1 run(s) |
| cohere-1m | [1000] | C VxVector CUDA | 53.4 | 2.8 | 37.2 | 23.36 | 1.26 | 3 run(s) |
| cohere-1m | [1000] | C VxVector CUDA, skip_index_build | 15.8 | 2.8 | 0.0 | 9.54 | 1.26 | 1 run(s) |
| cohere-1m | [4000] | A VectorChord internal | 161.7 | 89.5 | 72.0 | 20.64 | — | 3 run(s) |
| cohere-1m | [4000] | C VxVector CUDA | 127.5 | 38.9 | 74.7 | 23.66 | 3.42 | 3 run(s) |
| cohere-1m | [4000] | C VxVector CUDA, skip_index_build | 53.5 | 38.9 | 0.0 | 11.88 | 3.42 | 1 run(s) |
| cohere-1m | [4000] | B VxVector CPU | not run | ≈ 11.0 h (estimate) | | | | estimated from B's measured rate at [1000], 0.78 G multiply-adds/s |
| openai-1m | [1000] | A VectorChord internal | 73.9 | 9.3 | 64.6 | 37.48 | — | 3 run(s) |
| openai-1m | [1000] | B VxVector CPU | 5311.9 | 5224.2 | 64.5 | 44.98 | 0.00 | 1 run(s) |
| openai-1m | [1000] | C VxVector CUDA | 95.0 | 5.6 | 65.7 | 45.01 | 2.00 | 3 run(s) |
| openai-1m | [1000] | C VxVector CUDA, skip_index_build | 26.9 | 5.6 | 0.0 | 18.63 | 2.00 | 1 run(s) |
| openai-1m | [4000] | A VectorChord internal | 150.7 | 80.1 | 70.6 | 40.09 | — | 3 run(s) |
| openai-1m | [4000] | C VxVector CUDA | 178.2 | 77.7 | 77.1 | 45.49 | 6.31 | 3 run(s) |
| openai-1m | [4000] | C VxVector CUDA, skip_index_build | 100.4 | 77.6 | 0.0 | 23.29 | 6.31 | 1 run(s) |
| openai-1m | [4000] | B VxVector CPU | not run | ≈ 22.7 h (estimate) | | | | estimated from B's measured rate at [1000], 0.75 G multiply-adds/s |

### Where C's time goes

Medians of C's 3 runs, from the timings `vxvector.create_vector_index` returns. "Transfer"
is the backend streaming every row of the table to the worker.

| Dataset | Lists | Read + sample (s) | Transfer to worker (s) | GPU alloc + copies (s) | GPU training (s) | Centroid write (s) | Index after centroids (s) | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift-1m | [1000] | 0.2 | 2.8 | 0.02 | 0.5 | 0.01 | 16.0 | 19.9 |
| sift-1m | [4000] | 0.4 | 3.1 | 0.06 | 6.7 | 0.05 | 54.0 | 64.6 |
| gist-1m | [1000] | 0.6 | 14.2 | 0.11 | 3.5 | 0.03 | 36.6 | 55.0 |
| gist-1m | [4000] | 1.6 | 15.2 | 0.43 | 48.4 | 0.12 | 94.9 | 159.8 |
| cohere-1m | [1000] | 0.5 | 12.7 | 0.09 | 2.8 | 0.03 | 37.2 | 53.4 |
| cohere-1m | [4000] | 1.3 | 13.1 | 0.35 | 38.9 | 0.11 | 74.7 | 127.5 |
| openai-1m | [1000] | 0.8 | 22.7 | 0.19 | 5.6 | 0.05 | 65.7 | 95.0 |
| openai-1m | [4000] | 2.4 | 22.2 | 0.68 | 77.7 | 0.17 | 77.1 | 178.2 |

### Training throughput

"Exact Lloyd multiply-adds" is samples x lists x dims x 10 iterations, the distance work of
plain Lloyd. VxVector does all of it; VectorChord's trainer does not (finding 3). For
scale, the H200's FP32 peak is about 67 TFLOP/s, about 33 T multiply-adds/s.

| Dataset | Lists | Samples | A training (s) | C GPU training (s) | B CPU training (s) | Exact Lloyd multiply-adds | C rate (T/s) | B rate (G/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift-1m | [1000] | 256,000 | 2.3 | 0.5 | 287 | 3.28e+11 | 0.68 | 1.14 |
| sift-1m | [4000] | 1,000,000 | 33.5 | 6.7 | — | 5.12e+12 | 0.77 | — |
| gist-1m | [1000] | 256,000 | 7.7 | 3.5 | 2,370 | 2.46e+12 | 0.70 | 1.04 |
| gist-1m | [4000] | 1,000,000 | 103.1 | 48.4 | — | 3.84e+13 | 0.79 | — |
| cohere-1m | [1000] | 256,000 | 8.1 | 2.8 | 2,537 | 1.97e+12 | 0.70 | 0.78 |
| cohere-1m | [4000] | 1,000,000 | 89.5 | 38.9 | — | 3.07e+13 | 0.79 | — |
| openai-1m | [1000] | 256,000 | 9.3 | 5.6 | 5,224 | 3.93e+12 | 0.70 | 0.75 |
| openai-1m | [4000] | 999,000 | 80.1 | 77.7 | — | 6.14e+13 | 0.79 | — |

### The answer: training share and C's speedup

| Dataset | Lists | A: training share | C speedup over A |
| --- | --- | --- | --- |
| sift-1m | [1000] | 12% | 0.93x |
| sift-1m | [4000] | 38% | 1.37x |
| gist-1m | [1000] | 17% | 0.81x |
| gist-1m | [4000] | 51% | 1.26x |
| cohere-1m | [1000] | 19% | 0.80x |
| cohere-1m | [4000] | 55% | 1.27x |
| openai-1m | [1000] | 13% | 0.78x |
| openai-1m | [4000] | 53% | 0.85x |

### Index quality

Medians over runs; 1,000 queries, single client; `vchordrq.epsilon` at its default.

| Dataset | Lists | Config | Probes | Recall@10 | QPS | p50 (ms) | p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sift-1m | [1000] | A | 10 | 0.8624 | 169 | 5.53 | 12.70 |
| sift-1m | [1000] | A | 20 | 0.9481 | 96 | 9.93 | 23.34 |
| sift-1m | [1000] | A | 50 | 0.9912 | 50 | 19.02 | 44.83 |
| sift-1m | [1000] | A | 100 | 0.9980 | 33 | 28.19 | 76.31 |
| sift-1m | [1000] | B | 10 | 0.8614 | 171 | 5.41 | 13.70 |
| sift-1m | [1000] | B | 20 | 0.9452 | 98 | 9.48 | 24.02 |
| sift-1m | [1000] | B | 50 | 0.9920 | 50 | 18.81 | 47.99 |
| sift-1m | [1000] | B | 100 | 0.9982 | 32 | 28.36 | 77.35 |
| sift-1m | [1000] | C | 10 | 0.8613 | 171 | 5.42 | 13.82 |
| sift-1m | [1000] | C | 20 | 0.9452 | 101 | 9.17 | 23.44 |
| sift-1m | [1000] | C | 50 | 0.9920 | 50 | 18.74 | 47.71 |
| sift-1m | [1000] | C | 100 | 0.9982 | 32 | 28.66 | 76.93 |
| sift-1m | [4000] | A | 40 | 0.9408 | 135 | 6.97 | 15.61 |
| sift-1m | [4000] | A | 80 | 0.9799 | 81 | 11.88 | 26.47 |
| sift-1m | [4000] | A | 200 | 0.9966 | 43 | 22.34 | 53.10 |
| sift-1m | [4000] | A | 400 | 0.9986 | 28 | 33.36 | 84.24 |
| sift-1m | [4000] | C | 40 | 0.9423 | 135 | 6.98 | 15.38 |
| sift-1m | [4000] | C | 80 | 0.9814 | 79 | 12.28 | 27.05 |
| sift-1m | [4000] | C | 200 | 0.9968 | 41 | 23.37 | 53.35 |
| sift-1m | [4000] | C | 400 | 0.9988 | 27 | 33.89 | 85.41 |
| gist-1m | [1000] | A | 10 | 0.7068 | 147 | 5.93 | 19.67 |
| gist-1m | [1000] | A | 20 | 0.8404 | 88 | 9.59 | 33.68 |
| gist-1m | [1000] | A | 50 | 0.9509 | 49 | 17.03 | 62.16 |
| gist-1m | [1000] | A | 100 | 0.9892 | 31 | 25.47 | 96.82 |
| gist-1m | [1000] | B | 10 | 0.7050 | 147 | 6.08 | 19.76 |
| gist-1m | [1000] | B | 20 | 0.8428 | 90 | 9.75 | 33.54 |
| gist-1m | [1000] | B | 50 | 0.9553 | 46 | 18.16 | 69.41 |
| gist-1m | [1000] | B | 100 | 0.9894 | 31 | 25.73 | 100.29 |
| gist-1m | [1000] | C | 10 | 0.7054 | 150 | 5.91 | 18.95 |
| gist-1m | [1000] | C | 20 | 0.8433 | 85 | 10.21 | 35.52 |
| gist-1m | [1000] | C | 50 | 0.9540 | 48 | 17.46 | 65.48 |
| gist-1m | [1000] | C | 100 | 0.9893 | 30 | 26.36 | 100.87 |
| gist-1m | [4000] | A | 40 | 0.8160 | 121 | 7.45 | 22.92 |
| gist-1m | [4000] | A | 80 | 0.9087 | 77 | 11.52 | 37.81 |
| gist-1m | [4000] | A | 200 | 0.9779 | 42 | 19.83 | 71.86 |
| gist-1m | [4000] | A | 400 | 0.9953 | 28 | 28.81 | 108.24 |
| gist-1m | [4000] | C | 40 | 0.8057 | 126 | 7.34 | 22.03 |
| gist-1m | [4000] | C | 80 | 0.9074 | 77 | 11.57 | 37.35 |
| gist-1m | [4000] | C | 200 | 0.9782 | 42 | 19.85 | 69.70 |
| gist-1m | [4000] | C | 400 | 0.9952 | 28 | 28.79 | 106.86 |
| cohere-1m | [1000] | A | 10 | 0.8784 | 193 | 3.71 | 21.28 |
| cohere-1m | [1000] | A | 20 | 0.9355 | 140 | 5.31 | 27.49 |
| cohere-1m | [1000] | A | 50 | 0.9770 | 89 | 8.94 | 36.99 |
| cohere-1m | [1000] | A | 100 | 0.9919 | 60 | 13.96 | 46.91 |
| cohere-1m | [1000] | B | 10 | 0.8207 | 355 | 2.42 | 8.86 |
| cohere-1m | [1000] | B | 20 | 0.8892 | 251 | 3.41 | 12.40 |
| cohere-1m | [1000] | B | 50 | 0.9464 | 150 | 5.74 | 17.36 |
| cohere-1m | [1000] | B | 100 | 0.9751 | 95 | 9.04 | 27.06 |
| cohere-1m | [1000] | C | 10 | 0.8198 | 370 | 2.34 | 8.66 |
| cohere-1m | [1000] | C | 20 | 0.8899 | 268 | 3.17 | 11.62 |
| cohere-1m | [1000] | C | 50 | 0.9472 | 164 | 5.18 | 16.72 |
| cohere-1m | [1000] | C | 100 | 0.9751 | 104 | 8.23 | 24.83 |
| cohere-1m | [4000] | A | 40 | 0.9397 | 182 | 4.13 | 19.75 |
| cohere-1m | [4000] | A | 80 | 0.9668 | 129 | 6.02 | 26.77 |
| cohere-1m | [4000] | A | 200 | 0.9872 | 83 | 9.88 | 37.51 |
| cohere-1m | [4000] | A | 400 | 0.9954 | 57 | 14.95 | 48.23 |
| cohere-1m | [4000] | C | 40 | 0.8974 | 290 | 3.19 | 8.09 |
| cohere-1m | [4000] | C | 80 | 0.9360 | 212 | 4.30 | 10.99 |
| cohere-1m | [4000] | C | 200 | 0.9718 | 126 | 7.19 | 18.36 |
| cohere-1m | [4000] | C | 400 | 0.9865 | 83 | 10.86 | 26.66 |
| openai-1m | [1000] | A | 10 | 0.8978 | 394 | 2.16 | 7.34 |
| openai-1m | [1000] | A | 20 | 0.9387 | 301 | 3.03 | 7.77 |
| openai-1m | [1000] | A | 50 | 0.9735 | 166 | 5.65 | 11.97 |
| openai-1m | [1000] | A | 100 | 0.9870 | 91 | 10.69 | 18.36 |
| openai-1m | [1000] | B | 10 | 0.8982 | 363 | 2.40 | 7.86 |
| openai-1m | [1000] | B | 20 | 0.9417 | 291 | 3.14 | 8.24 |
| openai-1m | [1000] | B | 50 | 0.9741 | 153 | 6.26 | 12.34 |
| openai-1m | [1000] | B | 100 | 0.9881 | 84 | 11.69 | 20.38 |
| openai-1m | [1000] | C | 10 | 0.8980 | 392 | 2.18 | 7.40 |
| openai-1m | [1000] | C | 20 | 0.9416 | 302 | 3.01 | 8.05 |
| openai-1m | [1000] | C | 50 | 0.9740 | 161 | 5.88 | 12.21 |
| openai-1m | [1000] | C | 100 | 0.9880 | 85 | 11.34 | 19.44 |
| openai-1m | [4000] | A | 40 | 0.9293 | 349 | 2.50 | 7.42 |
| openai-1m | [4000] | A | 80 | 0.9584 | 257 | 3.58 | 8.60 |
| openai-1m | [4000] | A | 200 | 0.9829 | 140 | 6.82 | 12.99 |
| openai-1m | [4000] | A | 400 | 0.9929 | 79 | 12.47 | 18.99 |
| openai-1m | [4000] | C | 40 | 0.9254 | 338 | 2.59 | 7.50 |
| openai-1m | [4000] | C | 80 | 0.9571 | 259 | 3.57 | 8.18 |
| openai-1m | [4000] | C | 200 | 0.9835 | 139 | 6.92 | 13.07 |
| openai-1m | [4000] | C | 400 | 0.9927 | 79 | 12.48 | 19.08 |

### Cohere: lower recall with VxVector's centroids

This is the only dataset where A's and C's recall differ by more than about 0.01. B and C
agree to within 0.001, so the cause is VxVector's training, not the GPU.

| Dataset | Lists | Probes | A recall@10 | B recall@10 | C recall@10 | A − C |
| --- | --- | --- | --- | --- | --- | --- |
| cohere-1m | [1000] | 10 | 0.8784 | 0.8207 | 0.8198 | +0.0586 |
| cohere-1m | [1000] | 20 | 0.9355 | 0.8892 | 0.8899 | +0.0456 |
| cohere-1m | [1000] | 50 | 0.9770 | 0.9464 | 0.9472 | +0.0298 |
| cohere-1m | [1000] | 100 | 0.9919 | 0.9751 | 0.9751 | +0.0168 |
| cohere-1m | [4000] | 40 | 0.9397 | — | 0.8974 | +0.0423 |
| cohere-1m | [4000] | 80 | 0.9668 | — | 0.9360 | +0.0308 |
| cohere-1m | [4000] | 200 | 0.9872 | — | 0.9718 | +0.0154 |
| cohere-1m | [4000] | 400 | 0.9954 | — | 0.9865 | +0.0089 |

## Phase 3: not run

Deferred to a later run (decided 2026-11-21). These are staged for it but were never run:
`scripts/setup_pgpu.sh` (cuVS 25.12 from RAPIDS' conda packages, cargo-pgrx 0.16.1, pgpu
`b39cc5e`) and config D in `scripts/phase2_bench.py`. The stopped instance keeps its disk,
with the loaded tables and the PostgreSQL cluster. Points read from pgpu's source at
`b39cc5e`, to check when it runs:

- **Hard-coded column.** `create_vectorchord_index` runs
  `CREATE INDEX ... USING vchordrq (embedding ...)`, whatever `column_name` is.
- **No cosine index.** It builds the operator class as `vector_{distance_operator}_ops`, so
  `'cos'` gives `vector_cos_ops`, which doesn't exist (`vector_cosine_ops` does). No cosine
  index can build. Config D runs pgpu's training and then pgpu's own `CREATE INDEX` with
  these two corrected.
- **`build.pin = 2`.** It adds this to VectorChord's external build; VxVector doesn't.
- **Training.** cuVS `kmeans::fit`, with cuVS's defaults for initialization and tolerance,
  `max_iter`, and `n_init` = `kmeans_nredo`.
- **Sampling.** By default, one contiguous block of `lists x sampling_factor` rows from a
  random offset, not seeded. With `batch_size` below `lists x sampling_factor` (its default
  is 100,000), it clusters batches and then consolidates them with a weighted k-means.
- **Only training runs on the GPU.** After training it inserts the centroids one row at a
  time through SPI, then runs an ordinary `CREATE INDEX` with `[build.external]`. To confirm
  with `nsys`.

## Findings

1. **Moving the rows to the worker costs more than GPU training saves at `[1000]`.**
   `vxvector.create_vector_index` streams the whole table from the backend to the worker.
   That takes 2.8 s on SIFT, 12-15 s on Cohere and GIST, and 22 s on OpenAI. It is longer
   than C's GPU training at every `[1000]` setting, and the main reason C is slower than A
   there.
2. **The CUDA kernel uses a small part of the H200.** It runs at 0.68-0.79 T exact
   multiply-adds/s, roughly 2-3% of FP32 peak. That is about 700x B's single thread, but only
   about 2x VectorChord's 8-thread trainer at `[4000]` (48.4 s against 103.1 s on GIST).
3. **VectorChord's "Lloyd" is not plain Lloyd.** In VectorChord 1.1.1,
   `crates/k_means/src/lib.rs` switches to its RaBitQ variant whenever there are more than
   twice as many samples as centroids, which is every build here. That variant rotates the
   samples and computes the exact distance to a centroid only when a RaBitQ lower bound
   (`rough - 1.9 * err`) could beat the best so far (`crates/k_means/src/index.rs`).
   VxVector computes every distance, which its bit-for-bit match with the CPU kernel
   requires. A therefore does much less arithmetic, and the saving grows with dimension.
   On OpenAI at `[4000]`, A's 80.1 s on 8 cores matches C's 77.7 s on the GPU.
4. **On Cohere, VxVector's centroids give lower recall than VectorChord's.** The gap is
   0.009-0.059 at every probe setting, for B and C alike; on the other datasets it is under
   about 0.01. The likely cause was found by reading the code and has not been tested:
   - For `vector_cosine_ops`, VectorChord normalizes every row (`opclass.rs:54`) and trains
     A's centroids on the unit vectors.
   - It uses external centroids as given: `make_external_build` ignores its opfamily.
   - VxVector's cosine centroids without `spherical_centroids` are means of the raw vectors.
   - Cohere's vectors have norms of 12.3-16.2, so its centroids are far off the unit sphere.
     OpenAI's are unit vectors, which would explain why OpenAI shows no gap.

   GloVe isn't unit-length either, yet showed no gap at the repo's 8,192-row scale, so this
   needs an experiment: C on Cohere with `spherical_centroids => true`, or the centroid norms.
5. **The GPU-unavailable reason names Metal on CUDA.** With no visible device, the
   `backend_reason` text and the `gpu`-mode error say "No Metal device, or Metal is turned
   off". The behavior is right; the text, at `src/data_path.c:69`, is Metal-only. Not
   changed here, since it isn't a script.
6. **Peak host RAM counts shared buffers.** `VmHWM` includes the shared-buffer pages each
   process touched (up to the 48 GB setting), so the RAM column overstates private memory.
   The VxVector worker's own peak RSS, from the same 200 ms sampling, was 0.15-1.6 GB for B
   at `[1000]`, and 0.4-1.9 GB for C at `[1000]` and 0.8-6.5 GB at `[4000]`. That is about
   the sample size: up to 1M x 1536 float32, 6.1 GB, on OpenAI.

## Failures and workarounds

1. **Vx's provisioning script stops on the Nebius image.** The image pins `ninja-build` to
   priority -1 (`/etc/apt/preferences.d/01-ubuntu-conflicts-lock`), so Vx's
   `scripts/provision/setup_linux.sh` fails. `scripts/setup_linux_cuda.sh` runs the same
   commands without `ninja-build`, which only Vx's from-source LLVM/Enzyme scripts use. The
   image's apt locks are untouched.
2. **The sanitizer checks need LLVM 22's compiler-rt** (`libclang-rt-22-dev`), which Vx's
   provisioning doesn't install. Its apt package depends on `libc6-i386`, which the image's
   `libc6*` pin refuses. The setup script instead unpacks the same package from apt.llvm.org
   into `/usr/lib/llvm-22`, without replacing any file.
3. **`pg_vxvector` doesn't build with LLVM 22's libclang.** The bindgen 0.71 in pgrx 0.16.1
   emits PostgreSQL 17.12's `HeapTupleHeaderData`, `Node` and `RelationData` without fields:
   14 errors such as "no field `t_infomask`", with or without rustfmt. It is built with
   Ubuntu's libclang 18 instead. (The Docker build uses libclang 14.)
4. **The Driver API prototype doesn't compile with CUDA 13.** `src/gpu_cuda.cpp` calls
   `cuDeviceComputeCapability`, which CUDA 13 deprecates, and the build uses `-Werror`.
   `scripts/build_gpu_lloyd_cuda.sh` now passes `-Wno-error=deprecated-declarations`; the
   source is unchanged.
5. **`scripts/build_gpu_cuda.sh` was committed without its executable bit**, so the plan's
   `./scripts/build_gpu_cuda.sh` fails with "Permission denied". Fixed in git (mode 100755).
6. **Scripts and tests that assumed macOS/Metal.** These are now platform-aware through the
   new `scripts/gpu_platform.py`, which holds the library names, the GPU library path, the
   backend name, and the Vx log lines for launches and host fallbacks:
   - `scripts/check_gpu_l2.py`. It also now takes the work per launch of spherical runs from
     the library (`vxvector_gpu_backend_launch_work`): 1 << 62 on CUDA, which never splits a
     launch, and 1 << 26 on Metal, unchanged.
   - `tests/test_backend.py`, where a `.dylib` path made one CPU unit test fail on Linux.
   - `scripts/check_cpu.sh`, for the real GPU library's path.
7. **The database scripts ran only in Docker.** `scripts/check_real_recall.py` now has
   `--native` (local `psql`) and `--backend-mode`, which also requires every Vx build to
   report that backend. PostgreSQL runs as the `ubuntu` user (`~/pgdata`, socket in
   `/tmp`), so the server and the worker it starts can read the checkouts without opening up
   the home directory.
8. **`shared_buffers` was raised from 128 MB to 48 GB for Phase 2.** At the default 128 MB, a
   SIFT query at 10 probes missed the buffer cache on 5,000-8,500 of its 9,000-13,000
   buffers, and single-client QPS was about 70 at a 14 ms median. 48 GB is 25% of RAM, the
   usual guideline. The first SIFT runs, made at 128 MB, are kept in
   `discarded-shared-buffers-128MB/` and not used. Phase 1 ran at the defaults.
