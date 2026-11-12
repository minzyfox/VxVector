# Step 1: refreshed CPU baseline (November 9, 2026)

This record completes Step 1 of the [active roadmap](pg_vxvector.md).
It applies to the revisions and hashes below, not to later kernels or compilers.
Standalone macOS checks only. PostgreSQL, the Linux worker, and recall were not rerun.

## Revisions

| Item | Value |
| --- | --- |
| Vx | `7e3cf6a3167fb23e0d49643bae68c385909c587e` (main, November 9), 64 commits after the old pin `31296d42` |
| Compiler | `vxc 0.0.3 (7e3cf6a3)`, debug profile, built with `VX_VERSION_WITH_COMMIT=1` |
| Compiler SHA-256 | `79324525373f999094a2c882aba5e25e1fd2ce18b3d18b504f0900eb830070cd` |
| `libvx_std_core` SHA-256 | `51d824bb1a055912b89ab74666cf8088fca4340d98acd4818493213a551b7642` |
| `src/kmeans.vx` SHA-256 | `e38a5421ccad40e07de62f4b080e402e07ea9a18d219e94d90ed9e4b7253d94b` |
| `kmeans.o` SHA-256 | `bae9b4d2d36be02d8f0005dfc934cb3c0ae74c53ebb2b8f0bd0ac4e1e67ce374` |
| `libvxvector_kmeans.dylib` SHA-256 | `af88e8ec8978d986c56c33f268062f3f28c01b6d6b1e2a20339480ea1f938523` |
| `vxvector-worker` SHA-256 | `64b340a660c3c3cfaba1d0265367fa4b80d65f56025fd61db29a08be107a9781` |
| Platform | macOS 27.2 arm64 |

[vx-revision.txt](../vx-revision.txt) now pins `7e3cf6a3`.
The [metadata file](evidence/step1_metadata_2026-11-09.json) lists all hashes, including the C sources and headers.

## Kernel change required by the new compiler

Vx `7e3cf6a3` stops with an internal compiler error on the previous kernel.
`normalize` branches on `squared > 0.0`, where `squared` is an `f64`.
The prover adds that branch condition to the facts of the index checks inside the branch.
It declares `squared` as an `Int` and writes `0.0` into a `QF_LIA` query.
z3 rejects the query. Since vx-lang/Vx#3011, a rejected query panics instead of being dropped silently.
The #3011 description says earlier compilers dropped rejected facts without a word. That explains why the old pin built this kernel; it was not rerun to confirm.

This reproduction fails with exit code 101:

```vx
unsafe fn scale(v : *mut f32, n : i32) -> void
requires valid(v, n)
requires initialized(v, n)
{
  let mut s: f64 = 0.0;
  for d in 0..n {
    s += f64::from(unsafe { v[d] });
  }
  if s > 0.0 {
    for d in 0..n {
      unsafe { v[d] = 1.0; }
    }
  }
}
```

The kernel now tests the value through `fn positive(value: f64) -> bool`.
A call result is an opaque name to the prover, so no float term reaches z3.
Behavior is unchanged. The fix belongs in the prover: a float literal or float comparison should be left out of the facts, as the `Div` operator is.
Treating `f64` names as integers is also unsound in principle: `0 < x < 1` has no integer solution.
Filed as [vx-lang/Vx#3145](https://github.com/vx-lang/Vx/issues/3145).

The other float comparisons, in cosine `distance`, have no later index check in their branches, so they produce no query.

Vx `7e3cf6a3` includes vx-lang/Vx#2852. Inlining `centroid_distance` into `sample_objective` now compiles.
The helper stays because it is a clearer boundary; its comment no longer claims the workaround is required.

## Results

All checks passed on the final source:

| Check | Result |
| --- | --- |
| Flat code generator | Build log contains `[flat-codegen] emitted module via the flat path` |
| Native unit tests (`tests/`) | 35 ran, 0 skipped, OK |
| Matched Faiss 1.15.1 cases | 48 passed |
| Independent float64 Lloyd cases | 96 passed |
| Zero/duplicate/empty-cluster cases | 6 passed |
| Data path allocation-failure and ASan/UBSan checks | Passed |
| Contract helpers ([check_contracts.py](../scripts/check_contracts.py)) | 5 proofs pass, 6 mutations fail at intended diagnostics |
| Kernel code generation ([check_kernel_codegen.py](../scripts/check_kernel_codegen.py)) | 8 cases pass, including 2 mutations |

The commands were `scripts/build.sh`, `scripts/check_cpu.sh`, `scripts/check_kernel_codegen.py`, and `scripts/check_contracts.py`, with `VX_ROOT` set to the Vx checkout.
On this machine the tool shell ran under Rosetta. The Rust linker then fails on `libxcrun`, so Vx and VxVector were built under `arch -arm64`.

### Native mutations

The November 4 mutations predate the November 9 kernel refactoring. They were repeated on isolated copies of the final source:

| Mutation | Change | Failing tests |
| --- | --- | --- |
| `norm_overflow` | `normalize` squares in f32 | `test_large_and_tiny_spherical_vectors_keep_their_direction` (6 subtests), `test_cosine_large_vectors_have_distinct_assignments` |
| `distance_overflow` | cosine `distance` squares norms in f32 | Same two tests, cosine subtests only |
| `sum_overflow` | `accumulate_point` returns 0 for a non-finite sum | `test_centroid_sum_overflow_returns_an_error` (3 subtests) |

Each mutation fails only the tests aimed at it.

## Audit of the current kernel

`vxc src/kmeans.vx --audit` lists 26 trusted clauses:

- 24 at the C boundary, in `vxvector_kmeans_v1`'s calls to `train` and `assign_rows`. They cover foreign buffer capacities and initialization. C code and runtime checks remain responsible for them.
- 2 `!overlaps(destination, source, length)` in `run_restart`. `!overlaps` takes one length, so it cannot state that two buffers of different lengths are disjoint.

No other trusted clause exists inside the kernel. `nearest`, `train`, and their helpers are `unsafe fn` with pointer contracts, so the November 8 review's finding 3 no longer applies to this source.

## Remaining proof gaps

These are current for Vx `7e3cf6a3`. They do not block Steps 2 or 3.

- Foreign buffer validity at the C boundary is trusted (24 clauses above).
- `!overlaps` for different lengths: [draft 4](vx_issue_drafts_2026-11-08.md). Still reproduces.
- The prover does not model integer division: [draft 6](vx_issue_drafts_2026-11-08.md). `spaced_position` keeps its extra remainder, and the build still warns that the prover ignores a `Div` condition.
- Float values: the prover cannot use float facts at all, and crashes on float branch conditions over index checks (above).
- Numerical properties (finite outputs, unit norms, objective decrease) are tested, not proved.
- Drafts 1, 2, 3, 5, and 9 still reproduce on `7e3cf6a3`. Draft 8 is a fragment and draft 7 has no reproduction; neither was rerun.
- Hosted CI has no recorded run.

## Not covered

- The Linux worker, PostgreSQL image, SQL suite, snapshot check, and real-data recall were not rebuilt or rerun. The installed database worker still uses an older Vx build.
  Revalidate them when the refreshed worker is packaged, as the roadmap says.
- Release-profile and `-O3` builds were not checked.
- `VX_CHECK_GPU` was not run; the GPU prototype is out of scope for Step 1.

## Evidence

- [Metadata, hashes, and mutation results](evidence/step1_metadata_2026-11-09.json)
- [CPU suite output](evidence/step1_cpu_2026-11-09.txt)
- [Contract checks](evidence/step1_contracts_2026-11-09.jsonl)
- [Code-generation checks](evidence/step1_codegen_2026-11-09.jsonl)
- [Kernel audit](evidence/step1_audit_2026-11-09.txt)
