# M4 index proofs — November 5, 2026

> Historical validation record. Names and runtime identifiers below reflect the installation before the VxVector rename. See the [current setup](../README.md).

The kernel now proves that `nearest` returns a valid centroid index.
Its accumulation caller must meet the corresponding numeric requirements.
The `buffer_index` helper proves widened row-major centroid offsets fit within
`rows * dims`, given positive dimensions bounded by `i32::MAX`.
The helper uses machine-integer arithmetic, including its wrapping behavior.

## Compiler changes

The Vx feature branch starts from main commit
`ded616150af9a3372ad7e6badbc68ebb824c5c8b`.
The feature commit is `c9612a1f1c8a2819258b159774a615f9aca298a7`, recorded in `vx-revision.txt`.
These changes are additional to that main commit:

- Keep separate facts for each branch, then join the continuing alternatives.
- Keep outer-variable loop invariants after checking all break paths.
  Do not export range-loop induction variables or trust shadowed variables.
- Retain scalar call postconditions independently of input-buffer facts.
  Exclude facts that depend on changed arguments or replay argument calls.
- Use Z3's nonlinear integer mode for products of variable dimensions.
  Limit solver queries to five seconds. Treat errors and unknown results as failed proofs.
- State `return == v` for `i64::from(i32)`. Buffer contracts need that value relationship.

The compiler changes address parts of
[Vx #1358](https://github.com/vx-lang/Vx/issues/1358) and
[Vx #108](https://github.com/vx-lang/Vx/issues/108).
They do not claim to complete either issue.

## Kernel changes

`nearest` requires `k > 0` and ensures `0 <= return < k`.
Its loop keeps that bound as an invariant.
`accumulate_point` requires a bounded centroid index before updating counts and sums.
Training requires positive centroid counts and dimensions.
The public entry point checks those dimensions before entering training.

`buffer_index` computes the centroid offsets during sum initialization,
point accumulation, and centroid division.
The accumulation helper preserves the previous order of float32 additions and
rejects non-finite totals as before.

The accepted reduced examples now live directly under `tests/contracts/`.
The contract harness tests invalid indices and offsets separately.
Kernel mutations remove the nearest result bound, return an invalid index,
and return an offset past the buffer.
Each mutation must fail at its intended contract diagnostic.

## Validation and remaining work

Both compiler formal-verification suites pass.
The solver-result unit test and exact flat-corpus coverage check pass.
Fourteen focused compiler and fixture mutations exercised the intended assertions.
The Vextor contract and kernel-code-generation harnesses pass on the native compiler.
The final pinned native build and isolated candidate passed 35 unit tests, 48 Faiss cases,
96 independent Lloyd cases, six edge cases, and sanitizers.
Their kernel objects are byte-identical. See [build hashes](evidence/index_proofs_2026-11-05_build.json) and [the pinned native suite](evidence/index_proofs_2026-11-05_native.log).

M4 remains open. These changes do not prove sampled row bounds, every source-buffer
offset, or positive cluster counts at division.
Foreign allocation capacities, lifetimes, initialization, and required disjointness
remain caller obligations.
The installed PostgreSQL worker remains the earlier main build until a separate
validated worker installation occurs.
A successful hosted CPU workflow run must also be recorded.

The full Vx commit hook could not pass the ANE placement test because compiled
CoreML models were unavailable. A temporary Python 3.12 environment generated
model packages, but `coremlc` failed to compile them. This is an environment
limitation, not a successful hardware test. The compiler commit records the
hook exclusion. Formatting and workspace Clippy passed before committing.
