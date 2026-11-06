# Vx main update: November 5, 2026

> Historical validation record. Names and runtime identifiers below reflect the installation before the VxVector rename. See the [current setup](../README.md).

## Revision and scope

Vextor now pins `ded616150af9a3372ad7e6badbc68ebb824c5c8b`.
The supplied local Vx checkout, `main`, and `origin/main` identify this commit.
The previous pin was `7cc3d2d95fb61f0b80d3b0bd1d8ff38b833f1c47`.
No fetch or branch change was needed. Local untracked files stayed intact.

The [upstream changes](https://github.com/vx-lang/Vx/compare/7cc3d2d95fb61f0b80d3b0bd1d8ff38b833f1c47...ded616150af9a3372ad7e6badbc68ebb824c5c8b)
include implemented features and proposed designs. The distinction matters for the TODOs.
This update changes the pin and roadmap, without changing the clustering algorithm or public ABI.
Earlier database, objective, and compiler evidence retains its original revision.
It is historical evidence, not validation of this new pin.

The first native build retained an old cached version suffix.
Cleaning only the `vx-build-commit` package and rebuilding refreshed the suffix.
The final native compiler reports `vxc 0.0.3 (ded61615)`.

## Capabilities that landed

| Capability | Evidence in this update | Consequence for Vextor |
| --- | --- | --- |
| Safe pointer requirements, pointer postconditions, and `!frees` tracking | Existing pointer fixtures compile through flat code generation. The Vec execution fixture checks its results. | Extend buffer contracts using current facilities before requesting a new pointer-fact system. Foreign allocation obligations remain trusted. |
| Trusted-fact audit | The audit fixture passes FileCheck. Auditing the unchanged k-means kernel lists 28 trusted clauses at unsafe calls. | The audit tool is available. Reducing the trusted boundaries remains open. |
| `on Topology::Any` | The upstream fixture executes and produces its expected host/device results through flat code generation. | Helpers can use the caller's topology. Validate the actual clustering helpers before adoption. |
| Runtime tensor dimensions and tensor-valued `tasks` | Existing fixtures execute through flat code generation and pass FileCheck. The task fixture also passes with one shared-runtime worker. | Runtime-sized tiles and batched work can use these implementations. Memory budgets and application concurrency still need validation. |
| CPU block-parallel `across` | The upstream four-worker fixture produces `total=19200`. | Evaluate current CPU block workers before proposing another scheduler. Centroid reductions still need race-free partial buffers and merging. |
| Reusable async-copy tokens | The existing A100 PTX fixture passes FileCheck, including the tiled pipeline. | A compile-time token-reassignment gap is resolved. Native CUDA overlap and lifetime behavior still need measurement. |
| Apple GPU kernel backend | MSL generation passes FileCheck. The native GPU fixture passes output and Metal-dispatch checks. Its four-worker CPU copy produces identical output. | Optional local GPU experiments are possible. This does not complete Vextor GPU integration or NVIDIA validation. |
| Apple GPU batching and completion | The new runtime implements command batching, memory-dependent waits, and waits at public library returns. Source inspection supports these claims. | Evaluate completion and C-boundary behavior for Vextor's own buffers. Portable overlap remains open. |

Relevant upstream changes include [#2769](https://github.com/vx-lang/Vx/pull/2769),
[#2768](https://github.com/vx-lang/Vx/pull/2768),
[#2780](https://github.com/vx-lang/Vx/pull/2780),
[#2786](https://github.com/vx-lang/Vx/pull/2786),
[#2731](https://github.com/vx-lang/Vx/pull/2731),
[#2749](https://github.com/vx-lang/Vx/pull/2749),
[#2761](https://github.com/vx-lang/Vx/pull/2761),
[#2775](https://github.com/vx-lang/Vx/pull/2775),
[#2762](https://github.com/vx-lang/Vx/pull/2762),
[#2750](https://github.com/vx-lang/Vx/pull/2750),
[#2812](https://github.com/vx-lang/Vx/pull/2812), and
[#2841](https://github.com/vx-lang/Vx/pull/2841).

## Gaps that remain

Both M4 reduced proof cases still fail. `nearest_loop.vx` reports E8009 for invariant
preservation and E8001 for its result contract. `variable_offset.vx` reports E8001.
The loop-invariant implementation in [#2842](https://github.com/vx-lang/Vx/pull/2842)
does not establish these application proofs.

Pointer clauses in loop invariants remain a
[proposal](https://github.com/vx-lang/Vx/blob/ded616150af9a3372ad7e6badbc68ebb824c5c8b/docs/discussions/raw_pointer_facts_in_loops.md).
Type invariants also remain a
[proposal](https://github.com/vx-lang/Vx/blob/ded616150af9a3372ad7e6badbc68ebb824c5c8b/docs/discussions/type_invariants.md).
Do not treat their examples as supported syntax.
The audit lists trusted clauses and assumptions, not every unsafe pointer access.
Its count is not a complete measure of memory-safety obligations.
The new Vec pointer postconditions still rely on explicit trusted assumptions inside the library.
They do not establish fallible allocation or complete type-invariant verification.

The `std::tensor`, `std::simd`, and `std::collective` API files did not change in this range.
The following candidates remain open:

- Reusable caller-budgeted matrix packing workspace and recoverable matrix-allocation failures.
- Efficient borrowed-buffer SIMD with widening, reductions, and tails.
- Distance-producer/argmin fusion and budgeted grouping or segmented reductions.
- Allocation, register, spill, and cross-backend resource diagnostics.
- Layout-aware composition, distributed collective requirements, and Vextor-specific lifetime checks.

The Apple GPU backend rejects f64 device types, atomics, dynamic shared memory, and vector operations.
Vextor's distance intermediates remain f64. Moving the current kernel unchanged is not established.
The backend also flushes f32 subnormals in both GPU execution and its CPU copy.
Any clustering experiment must check that behavior against the existing numerical contract.
See the [backend status at this revision](https://github.com/vx-lang/Vx/blob/ded616150af9a3372ad7e6badbc68ebb824c5c8b/docs/gpu_backends.md).

## Validation

The native library builds at the new pin without source changes.
The native suite passes 35 unit tests, 48 Faiss cases, 96 independent Lloyd cases,
six edge cases, and the allocation/sanitizer checks.
The contract harness passes three helper proofs and four mutations, while retaining two known gaps.
The code-generation harness passes its baseline and both intended mutation failures.

The exact flat-path corpus check passes: one selected test, no failures.
The Linux compiler, runtime, and kernel object build successfully.
Linux contract and code-generation checks also pass.
The packaged worker is installed and PostgreSQL reports healthy.
The volume and cluster identifier match their previous values.
The Linux object hash matches the November 4 object exactly.
PostgreSQL integration, worker failure/cancellation, and snapshot checks pass.
Synthetic validation passes 24 index builds, two centroid-only builds, and 120 recall queries.
The minimum synthetic recall is 1.0.
The larger real-data and instrumented-objective suites were not rerun for this update.
Rebuild the optional objective-reference image before using it with the new pin.
Its provenance check rejects a worker built with the old pin.

Evidence:

- [New-pin validation summary](evidence/vx_main_test_summary_2026-11-05.json).
- [PostgreSQL integration](evidence/vx_main_postgres_2026-11-05.txt), [extension behavior](evidence/vx_main_extension_2026-11-05.txt), [snapshot checks](evidence/vx_main_snapshot_2026-11-05.txt), and [synthetic recall](evidence/vx_main_recall_2026-11-05.txt).

- [Exact flat-path coverage check](evidence/vx_main_flat_coverage_2026-11-05.txt).
- [Native regression log](evidence/vx_main_native_2026-11-05.txt).
- [Installed image and database identity](evidence/vx_main_install_2026-11-05.json).
- [Linux contract checks](evidence/vx_main_linux_contracts_2026-11-05.jsonl) and [Linux code-generation checks](evidence/vx_main_linux_codegen_2026-11-05.jsonl).
- [Contract checks](evidence/vx_main_mac_contracts_2026-11-05.jsonl) and [code-generation checks](evidence/vx_main_mac_codegen_2026-11-05.jsonl).
- [Capability probes](evidence/vx_main_capabilities_2026-11-05.jsonl).
- [Kernel pointer audit](evidence/vx_main_pointer_audit_2026-11-05.txt) and [upstream audit fixture](evidence/vx_main_audit_fixture_2026-11-05.txt).
- [Metal output](evidence/vx_main_metal_output_2026-11-05.txt), [GPU dispatch](evidence/vx_main_metal_dispatch_2026-11-05.txt), and [CPU-copy dispatch](evidence/vx_main_metal_cpu_dispatch_2026-11-05.txt).

For reproducibility, the commands use the configured LLVM tools and the repository virtual environment.
The Linux build uses an archived source tree and its existing container toolchain.
No Vx compiler or stdlib implementation was changed for this audit.
