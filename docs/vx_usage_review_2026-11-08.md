# Review: how VxVector uses Vx (November 8, 2026)

This review looks at VxVector from the point of view of its goal. The goal is a
production-grade PostgreSQL extension written in Vx. It must not cut corners to get
things working. It must also flag what the Vx compiler and language are missing, so
that Vx improves in the process. The areas named for this are SIMD, provenance, the
newer loop syntax and dynamic topology change.

The review is based on reading the code and the docs. No tests were run for it.

## Summary

The extension engineering is on track. The language-improvement goal is not on track
yet. The Postgres side is careful and done the right way. But the Vx part is small and
reads like C, so it does not exercise the language much. SIMD, provenance, the newer
loop syntax and dynamic topology are almost untouched.

## What is going well

- The worker runs in a separate process. A crash, a timeout or a cancel kills the
  worker, not the PostgreSQL backend. A failed build rolls back its outputs.
- The checks compare against Faiss and against VectorChord's own trainer. They check
  recall on GloVe and SIFT, and they run sanitizer and allocation-failure tests. The
  README says clearly what is not established.
- The caller owns every buffer. The C ABI has a version (`v1`), and the memory budget
  is checked before allocation.
- The contract proofs on `nearest`, `buffer_index` and `accumulate_point` are real.
  They also led to real fixes in the Vx compiler.

## Where it falls short

### 1. Vx is a small kernel inside a C program

`src/kmeans.vx` has 363 lines. The C files `src/data_path.c`, `src/worker.c` and
`pg_vxvector/src/bridge.c` have about 800 lines together, and there is Rust glue too.
The parts with real logic are in C: row validation, the memory budget, and the
sampler (Floyd's algorithm with SplitMix64).

There are now two samplers: the C sampler, and the stride permutation in `train`.
These are two code paths that can disagree. If the goal is to test Vx, more of this
layer should move into Vx. Where it cannot move, record that as a Vx gap.

### 2. The Vx code is C written in Vx syntax

- `train` takes 20 arguments.
- Each buffer travels as a raw pointer plus a separate offset, such as `data_offset`.
- Flags are `i32` values (`spherical`, `write_assignments`, `metric`), not `bool` and
  enum values.

Every signature repeats the pointer, the offset and the length. Every `requires`
repeats `valid(p, n)`. This is the strongest language signal in the repository: Vx
needs a borrowed slice or view type that carries its length, so that the bounds come
from the type. The roadmap does not name this gap. `src/gpu_lloyd.vx` already uses
`tensor_view_2d`, while the CPU kernel uses raw pointers.

### 3. Memory safety inside the kernel is trusted, not proved

`nearest` and `train` are safe functions. They have no `unsafe` marker and no pointer
`requires`. Inside, they call `unsafe` helpers such as `distance` and `copy_vector`.
Nothing proves the `valid(...)` preconditions of those helpers. The 28 trusted clauses
that the audit lists are these calls.

So the proofs cover integer indices. Memory safety inside the kernel is trusted, not
only at the C boundary.

Vx does this by design. Its `docs/discussions/raw_pointer_facts.md` says that for a call
in an `unsafe` block, "the checker trusts the clause. The audit lists it". This is the
same rule as in Rust: an `unsafe` block is the programmer's promise. A test program
confirmed it on November 8. A safe function passed one pointer to an `unsafe fn` that
requires a million valid floats. Vx compiled it with no error or warning, and only
`--audit` listed the call:

```vx
unsafe fn first(p : *const f32, n : i32) -> f32
requires valid(p, n)
requires n > 0 {
  return unsafe { p[0] };
}

// A safe function with no pointer requirement.
fn read_anything(p : *const f32) -> f32 {
  return unsafe { first(p, 1000000) };
}

fn main() -> i32 {
  let x : f32 = 1.0;
  let p : *const f32 = &x;
  return read_anything(p) as i32;
}
```

So the fault is in VxVector, not in Vx. The safety of `nearest` and `train` depends on
pointers that their callers give. Each one must be an `unsafe fn` with the pointer
requirements on the outside (TODO B2).

Vx can still help with a lint. The lint warns when a safe function gives its own raw
pointer parameter to a trusted clause. This pattern almost always means a missing
`unsafe fn`.

### 4. Some details are "made to work", not "done right"

- `batch_size` has no effect in the Vx kernel. The `while` loop over batches computes
  the same result as one loop over `0..samples`. Only the C reader uses the option.
- Initialization takes evenly spaced samples, shifted by fixed constants (`104729`
  and `48271`). The usual production choice is k-means++ or a seeded random choice.
- The stride sampler is deterministic and follows heap order. Heap order often
  correlates with insert time.
- Distances use f64, but centroid sums use f32. The code detects overflow, but not
  the loss of precision in large clusters.

None of these is fatal. Each one is a corner that the project goal says not to cut.

### 5. The GPU prototype works against the language

- It writes lanes by hand (`spawn on(Topology::GPU) { for lane in 0..256 }`) inside a
  `while` loop. The README notes that the generated code has no parallel launch
  metadata.
- Its reduction scans every row again for each centroid. The cost is
  O(k × rows × dims).
- It uses f32 distances, while the CPU path uses f64. The two paths behave differently.

The right test of Vx here is `across` with a segmented reduction or atomics. Where Vx
cannot express this, file it as a Vx gap.

### 6. The language-improvement loop does not close

- The docs are much larger than the code. The roadmap has 822 lines, and
  `docs/evidence/` has about 50 dated files.
- Most Vx items in the roadmap say "evaluate" or "audit". They are not filed issues.
  A search of vx-lang/Vx for "vextor", "VxVector" and "kmeans" found no issues.
- The pin (`c9612a1f`) is a fork commit. Its proof fixes are not yet in Vx `main`.

The intended loop is: a small reproduction, then a Vx issue, then a Vx regression
fixture, then removal of the workaround in VxVector. Today the output is mostly
evidence files in this repository.

## Recommended direction

1. Pause the evidence work for a while. The CPU path has enough validation to build on.
2. Turn the roadmap's Vx candidates into Vx issues, each with a small reproduction.
   Before you file one, search for an existing issue and add the VxVector use case to
   it. Start with these:
   - slices `&[T]` (Vx #534), to replace the pointer, offset and length arguments;
   - vector loads from an `f32` buffer (Vx #478, part of #482), with f32-to-f64
     widening, reductions and scalar tails;
   - a lint for a safe function that gives its raw pointer to a trusted clause
     (section 3).
3. Make `nearest` and `train` into `unsafe fn` with real pointer contracts. Later,
   rewrite `distance` and `nearest` with slices and `<N x T>` vectors, and remove the
   pointer offsets. This tests SIMD, provenance and bounds proofs together.
4. Rewrite the GPU path with `across` and a correct reduction. Use it to test dynamic
   topology: one function that runs on CPU or GPU, chosen at run time, with the CPU
   fallback built in.
5. Move sampling into Vx and keep one sampler. Remove `batch_size` from the kernel ABI.

In short, the product work is responsible, but it uses Vx as a thin kernel inside a
C program. To improve the language, the Vx code must look like idiomatic Vx. Each
place where it cannot must become an issue in vx-lang/Vx.

## What Vx offers today

A short check of the Vx repository (November 8) gives this starting point for the
TODO list:

- **Vector values exist, but a kernel cannot load them from its data.** `<N x T>` is a
  native type, such as `<8 x f32>`, and its arithmetic is lane-wise. A load or store
  goes through a pointer to the vector type. But Vx rejects the cast from `*const f32`
  to `*const <8 x f32>` ("Unsupported cast"), so an `f32` buffer cannot be read as
  vectors. Vx #478 plans this cast, with the source pointer's alignment. It is phase 4
  of the vector-type tracker #482, and it waits for #477 (vector operations).
- **`std::simd` is not the right tool for a kernel.** Each function is a foreign call
  that handles four `f32` lanes through pointers (`simd_add_f32x4` and the others).
  One call per four floats is the pattern that the roadmap says to avoid. Vx #952 asks
  to rebuild `std::simd` on `<N x T>`.
- **There is no borrowed slice type for a raw buffer.** Vx has tensor rows and
  `tensor_view_2d`, which is a compiler built-in. Nothing carries a pointer and its
  length together for a plain `f32` buffer. Vx #534 asks for slices `&[T]` and
  `&mut [T]`. It is open and has no comments. Vx `docs/discussions/view_type.md` sets
  the borrow rules for tensor views, and says that slices must share them.
- **An `unsafe` block trusts the clauses of the calls in it.** This is by design. See
  section 3.
- **Block-parallel `across` exists for CPU,** according to the
  [November 5 update](vx_main_update_2026-11-05.md). The Apple GPU backend rejects f64, atomics and vector operations.

## TODO list

Each item has a "Done when" line. An item is done only when that line is true. Order
matters inside each group. The groups can run in parallel, except where an item names
another item.

Each Vx gap follows the same loop:

1. Write the smallest `.vx` program that shows the gap.
2. Search vx-lang/Vx for an existing issue. If one exists, add the program and the
   VxVector use case to it as a comment. If none exists, file an issue with the
   program, the current behavior and the required behavior.
3. Link the issue from this list and from the workaround in VxVector.
4. When Vx fixes it, update the pin, remove the workaround, and check the item.

### A. Close the language loop first

- [ ] A1. Make a `docs/vx_gaps.md` table with one row for each Vx gap: the issue link,
  the small program, the VxVector workaround and its status. Move the gap rows out of
  `docs/pg_vxvector.md` into this table.
  Done when: each gap that the roadmap names has a row, and each row has an issue link
  or the reason why it has none.
- [ ] A2. Propose a Vx lint for a safe function that gives its own raw pointer
  parameter to a trusted clause (section 3). Use the 20-line program from section 3.
  This is a request for a warning. It is not a soundness bug. Do B2 first, because B2
  is the real fix in VxVector.
  Done when: the request is filed, or a Vx maintainer declined it, and A1's table
  records the result.
- [ ] A3. Comment on Vx #534 (slices) with the VxVector use case from C1:
  - make a slice once, at the C boundary, from a pointer and a length;
  - take a sub-slice for each row or centroid, which the prover can check;
  - write `distance(a: &[f32], b: &[f32])` with no `valid` requirements.
  Done when: the comment is posted, and A1's table links it.
- [ ] A4. Comment on Vx #478 and the tracker #482 with what a distance kernel needs
  beyond the cast: an unaligned load from an offset into a row, a scalar or masked
  tail, a horizontal sum, f32-to-f64 widening, and FMA with stated behavior. First,
  read #477 to see which of these it already plans. Follow #952 for `std::simd`.
  Done when: the comment is posted, and A1's table links #477, #478, #482 and #952.
- [x] A5. Get the proof fixes in pin `c9612a1f` merged into Vx `main`, and pin a
  commit on `main`.
  Done when: `vx-revision.txt` names a commit that `main` contains.
  Done (November 8): the pin is main `31296d42`. #2927 and #2899 merged the fork
  fixes, except #2852 (keep scalar result bounds after later buffer writes), which
  is open. Until it lands, `run_restart` passes the result of `nearest` straight to
  `centroid_distance`, because a later call drops the bound that `nearest`
  ensures. Remove that helper and add #2852 to A1's table when it merges.

### B. Make the memory-safety claims true

- [x] B1. Find out if Vx accepts a safe `fn` that calls an `unsafe fn` with a `valid`
  precondition that nothing proves. Answer (November 8): yes, by design. The
  `unsafe` block trusts the clause, and only `--audit` lists it. See section 3.
- [ ] B2. Mark each internal function that reads a raw pointer as `unsafe fn`:
  `nearest` and `train`. Give each one the `valid` and `initialized` requirements that
  its callees need.
  Done when: `vxc src/kmeans.vx --audit` lists trusted clauses only in
  `vxvector_kmeans_v1`, which is the C boundary.
  Status (November 8): all but two. `nearest`, `train` and the new `run_restart`
  and `centroid_distance` are `unsafe fn` with `valid`, `initialized` and
  `!overlaps` requirements, and `copy_vector` ensures `initialized`. The audit
  went from 34 trusted clauses inside the kernel to 2. The two left are the
  copies from `data` into `current`: `!overlaps(p, q, n)` takes one length for
  both buffers, so it cannot say that two buffers of different lengths are
  disjoint, and a larger length would claim memory past the end of `current`.
  `scripts/check_kernel_codegen.py` checks this audit, with a mutation that
  reads one row past the end.
  Vx gaps found, for A1 (each has a workaround in `src/kmeans.vx`):
  - `!overlaps` has no form with two lengths (the two trusted clauses above).
  - The pointer checker does not track an offset that contains a call, such as
    `offset(p, i64::from(c) * w)`. An `as i64` cast works but gets lint W1007.
  - A fact ends when a variable it names leaves scope, instead of being restated
    in the variables that remain (`batch_start = batch_end`, loop counters).
  - The loop event scan treats a `let` whose type it cannot infer, such as
    `let r = i64::from(f(x))` or `let x = unsafe { ... }`, as a possible drop,
    and this ends every pointer fact carried across the loop.
  - The loop event scan treats a pointer clause in a nested loop invariant,
    such as `invariant initialized(q, i)`, as a call that can free memory.
  - A function whose body only passes its pointers to `!frees` callees did
    not get `!frees` inferred. `run_restart` states it, and the checker
    verifies it.
  - The prover does not model `/`. In `vxvector_kmeans_v1`, a null check joined to
    the integer checks in one long `||` condition lost their facts. A small
    program does not reproduce this yet.
  - Issue drafts for each gap: [vx_issue_drafts_2026-11-08.md](vx_issue_drafts_2026-11-08.md).
- [ ] B3. Write the size requirement of each C ABI buffer in
  `include/vxvector_kmeans.h`, as one `requires` per pointer in
  `vxvector_kmeans_v1`. The C side cannot prove these, but Vx can check every use of
  them.
  Done when: each pointer argument of `vxvector_kmeans_v1` has a `valid` requirement
  with the size from the header.
- [ ] B4. Prove the remaining M4 items: sampled row indices are less than `rows`, and
  each division by `count` has `count > 0`.
  Done when: the two boxes in `docs/pg_vxvector.md` M4 are checked, with a mutation
  that fails each proof.

### C. Write idiomatic Vx

- [ ] C1. Write the smallest program that needs a pointer and a length as one value,
  for the A3 comment on Vx #534. Use `distance` as the example.
- [x] C2. Until Vx #534 is done, define one struct in `src/kmeans.vx` for a row-major
  matrix: pointer, row count and column count. Pass that struct in place of each
  pointer and offset pair, such as `data` with `data_offset`.
  Done when: no function in `src/kmeans.vx` takes an `_offset` argument.
  Done by B2 (November 8), without the struct: each function takes a pointer to the
  start of its own region. A struct would undo B2, because a pointer read from a
  field has no facts (Vx #2832). Use slices (#534) when they exist.
- [x] C3. Replace the `i32` flags with `bool` and enum values in the internal
  functions: `spherical`, `write_assignments`, and the `metric` code. Convert them
  only once, in `vxvector_kmeans_v1`.
  Done (November 8): `spherical` is a `bool` and `metric` a `DistanceMetric`,
  converted once in `vxvector_kmeans_v1`. `write_assignments` is gone, because
  `train` no longer writes assignments (C4).
- [ ] C4. Split `train` into small functions: initialize the centroids, run one Lloyd
  iteration, compute the objective, and write the assignments.
  Done when: no function in `src/kmeans.vx` has more than eight arguments.
  Status (November 8): partly done. `run_restart` holds one restart.
  `sample_objective`, `assign_rows`, `sampler_offset` and `sampler_stride` are their
  own functions. `assign_rows` writes both the leaf assignments and the leaf parents
  in `vxvector_kmeans_v1`, so `train` lost `assignments` and `write_assignments`.
  `train` and `run_restart` still have 15 arguments, and `sample_objective` has 9.
  The eight-argument limit needs structs, and on Vx main a struct field carries no
  facts. A pointer field has no facts (#2832), and the prover does not use a
  `requires` on a number field or the fields of a struct literal (draft 9 in
  [the issue drafts](vx_issue_drafts_2026-11-08.md)). A struct for the sampler or
  the options would turn proved clauses back into trusted ones. Splitting the
  seeding and the Lloyd iteration out of `run_restart` without structs gives
  functions with 10 to 14 arguments, so that part waits for the gap. D1 removes
  `batch_size` and D2 changes the sampler, which removes some arguments.
- [ ] C5. Use the newest Vx loop syntax where it fits, for example a stepped range for
  the batch loop if Vx has one. Record each loop that the syntax cannot express as a
  Vx gap.

### D. Fix the "made to work" details

- [ ] D1. Remove `batch_size` from the Vx kernel arguments, because it does not change
  the result. Keep it in the C reader. Update the header and bump the ABI version if
  the signature changes.
- [ ] D2. Use one sampler. Move Floyd's algorithm with SplitMix64 from
  `src/data_path.c` into Vx, and remove the stride permutation from `train`.
  Done when: the array API and the streaming adapter choose the same sample for the
  same seed and the same rows.
- [ ] D3. Replace the evenly spaced initialization with k-means++ or a seeded random
  choice. Compare recall and objective with the current method on the M4 datasets.
  Done when: the new method is the default and the M4 recall gates pass.
- [ ] D4. Measure the loss of precision in the f32 centroid sums on the largest real
  cluster. If the error matters, use f64 sums or a compensated sum.
  Done when: a test states the allowed error, and the kernel meets it.
- [ ] D5. Give each magic constant a name and a comment that says where it comes
  from: `104729` and `48271`. Remove a constant if D2 or D3 makes it unused.

### E. SIMD

- [ ] E1. Build the current kernel with `-O3`. Record whether LLVM vectorizes the
  `distance` loops by itself. This is the baseline for the next items.
  Done when: the generated instructions and the timings are saved in `docs/evidence/`.
- [ ] E2. Write `distance` for L2 with `<8 x f32>` values. Load from the `f32` buffer
  through a cast to `*const <8 x f32>`. Handle the scalar tail when `dims` is not a
  multiple of 8. This is blocked: today Vx rejects the cast. Vx #478 adds it.
  Until then, keep the 20-line program that shows the rejected cast in A1's table.
  When #478 is done, record what is still hard: alignment, provenance of the cast
  pointer, bounds proofs for the vector load, and widening to f64.
- [ ] E3. Decide the numerical contract: f32 or f64 accumulation, the order of the
  reduction, and FMA. Keep the scalar version as the reference.
  Done when: the contract is written in this repository, and a test compares SIMD and
  scalar results within a stated tolerance, including near ties.
- [ ] E4. Do the same for inner product, cosine, centroid accumulation and
  normalization, but only where E1's profile shows that the loop matters.
- [ ] E5. Do not use `std::simd` for the kernel. Vx #952 already asks to rebuild that
  module on `<N x T>` values instead of foreign calls. Link #952 from A1's table.

### F. GPU and dynamic topology

- [ ] F1. Rewrite the assignment step in `src/gpu_lloyd.vx` with `across`, not
  hand-written lanes in a `while` loop.
  Done when: the generated code has parallel launch metadata.
- [ ] F2. Replace the reduction that scans all rows for each centroid. Use per-block
  partial sums and a merge, or a segmented reduction. Record which form Vx cannot
  express.
- [ ] F3. Use the same distance code and the same numerical contract (E3) on CPU and
  GPU. Do not keep two versions with f32 on one side and f64 on the other.
- [ ] F4. Write one Lloyd step that runs on the CPU or the GPU, chosen at run time.
  Use `on Topology::Any` or the current Vx form, and fall back to the CPU when no GPU
  is present. This is the test of dynamic topology change.
  Done when: one Vx function runs on both targets, and the CPU fallback passes M4.
- [ ] F5. Run F1 to F4 on a real NVIDIA GPU. Compiling for A100 is not enough.

### G. Provenance

- [ ] G1. List each place where a pointer is made from another pointer: `offset`,
  `offset_mut`, the `as *const f32` casts, and the vector casts from E2. For each one,
  record whether Vx tracks which buffer the new pointer belongs to.
- [ ] G2. Write a program where `nearest` reads past the end of `centers` through an
  `offset`. Check that Vx rejects it at compile time.
  Done when: Vx rejects the program, or a Vx issue for it is in A1's table.

### H. Docs

- [ ] H1. Move the dated update notes, the historical pins and the old validation runs
  out of `README.md` into `docs/`. Keep the README to setup, use and tests.
- [ ] H2. Stop adding new dated evidence files until groups B to E have progress.
  Record a new run only when it checks a change.
- [ ] H3. Link this review and `docs/vx_gaps.md` from `docs/pg_vxvector.md`.
