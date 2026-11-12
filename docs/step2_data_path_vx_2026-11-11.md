# Step 2: the C data path in Vx (November 11, 2026)

This record completes Step 2 of the [active roadmap](pg_vxvector.md), except for the Linux and PostgreSQL checks listed under [Not covered](#not-covered).
It applies to the revisions and hashes below.
The [metadata file](evidence/step2_metadata_2026-11-11.json) lists every hash.

## Revisions

| Item | Value |
| --- | --- |
| Vx | `df51035df9cd0c7f2477a803f7e6ea985cd9cf71` (main, November 11). The pin moves from `7e3cf6a3`. |
| Compiler | `vxc 0.0.3 (df51035d)`, debug profile, built with `VX_VERSION_WITH_COMMIT=1` |
| Compiler SHA-256 | `06b1ebd610774c95d642b0763604ea9b57f0de95bd5482e3364237aa2018080b` |
| `src/kmeans.vx` SHA-256 | `cfc74f54b5a5155e1a5fa7924a13d08222c5891f94259f12208f98de2bde8738` |
| `kmeans.o` SHA-256 | `b7538c44239f892601b15c1f60b88d8de1a58f1237833799e37e4af9293f9833` |
| `libvxvector_kmeans.dylib` SHA-256 | `58c020aa6256d1e8f0f1a5df0d988aba1be60c322ca8b577eca3f0ddf597f659` |
| Platform | macOS 27.2 arm64 |

## What moved to Vx

[src/kmeans.vx](../src/kmeans.vx) now holds the data path's logic:

| Part | Vx function |
| --- | --- |
| Option checks, buffer sizes, memory budget | `vxvector_plan_cpu_reason_v1`, `valid_options`, `add_buffer` |
| SplitMix64 and Floyd's selection with its hash table | `splitmix64`, `random_below`, `table_slots`, `first_slot`, `find_slot`, `select_rows`, `choose_rows` |
| Sort of the sampled rows | `pair_positions`, `swap_pairs`, `sift_down`, `sort_positions` |
| Row reads and checks through the C reader | `read_batch` |
| Batch-to-sample copy | `gather` |
| Run: sample, read, train, check centroids | `vxvector_run_cpu_reason_v1`, `all_finite` |

[src/data_path.c](../src/data_path.c) went from 277 lines to 176. It keeps the public C API and three things that stay in C on purpose:

- **Allocation.** The data path owns its result buffers, and `vxvector_free_cpu_result_v1` releases them. [tests/data_path_memory.c](../tests/data_path_memory.c) fails each allocation in turn by replacing `malloc`. Vx can allocate, so this is not a Vx gap.
- **The clock.** Vx's clock is in its runtime library. The Linux worker and the PostgreSQL extension link the Vx object without that library. C passes `clock_gettime` in as a function pointer.
- **Messages.** Vx returns one of 12 outcomes. C maps each to its status code and text.

The roadmap's exit condition asked for C only where Vx cannot express the code. None of the three is a Vx gap. Each is a choice about linkage or testing, recorded here and in the file's header comment.

## One sampler, and kernel ABI v2

The kernel had its own sampler, a seeded stride permutation. It is gone.
Both APIs now choose `min(rows, k * sampling_factor)` rows the same way:

- every row in order, when that is all of them;
- otherwise SplitMix64 and Floyd's algorithm, in the order of selection.

Floyd's algorithm also returns every row in order when it selects all of them, so the first case only skips the hash table.

`vxvector_kmeans_v1` is replaced by `vxvector_kmeans_v2` in [include/vxvector_kmeans.h](../include/vxvector_kmeans.h):

- `batch_size` is removed. It never changed the result. `vxvector_cpu_options.batch_size` remains the reader's batch size.
- `sample_rows` (int64, the sample count) and `sample_table` (int64, the table size) are new caller-owned scratch buffers.

The data path passes its free sort buffer and hash table as that scratch, so `vxvector_plan_cpu_v1` sizes are unchanged.

Results change:

- **Direct API.** Sampled training now uses Floyd's selection instead of the stride. Centroids for sampled inputs differ from v1.
- **Data path.** The sampled rows are unchanged. The kernel now trains on them in selection order instead of permuting them again. Its centroids differ from those of the old build.

PostgreSQL recall must be rechecked with the next worker build. See [Not covered](#not-covered).

## Results

All checks passed on the final source. Logs are in [docs/evidence](evidence/).

| Check | Result |
| --- | --- |
| Flat code generator | Used |
| Old C versus new Vx, same inputs ([comparison](evidence/step2_c_to_vx_comparison_2026-11-11.json)) | `vxvector_plan_cpu_v1`: 35,846 cases, 0 differences in status, message, or sizes. Sample rows: 204 cases, 0 differences |
| Native unit tests | 37 ran, OK |
| Array API versus streaming data path | Same centroids and parents in 27 cases: three sizes, three seeds, flat and two-level trees, all metrics |
| Independent sampler ([scripts/sampler_reference.py](../scripts/sampler_reference.py)) | Matches the old C sampler in all 204 cases and the new library in the tests |
| Matched Faiss 1.15.1 cases | 48 passed, with the reference using the new sampler |
| Independent float64 Lloyd cases | 96 passed |
| Zero, duplicate, and empty-cluster cases | 6 passed |
| Allocation failure, ASan, UBSan | Passed. The real Vx data path runs, and a wrapper forces the two k-means outcomes |
| Code generation and audit ([check_kernel_codegen.py](../scripts/check_kernel_codegen.py)) | 12 cases passed, including 4 new mutations |
| Contract helpers ([check_contracts.py](../scripts/check_contracts.py)) | 5 proofs, 6 mutations, on `df51035d` |
| Plan-only C build with the Vx object | Links with only `abort` and `write` from libc. Plan matches |

The comparison ran [step2_capture_2026-11-11.py](evidence/step2_capture_2026-11-11.py) on the library at `0ac7c98` and on the final library.

### Proofs and new mutations

| Property | How it is checked | Mutation |
| --- | --- | --- |
| A sampled row index read from memory is below `rows` | `sample_row` ensures it. A check states it, because the prover cannot see memory | `sample_row_bound` fails with E8001 |
| A hash probe stays in the table | `first_slot` and `find_slot` ensure it | `first_slot_bound` fails with E8001 |
| The table has more slots than rows to choose | `table_slots` ensures it | `table_too_small` fails with E8001 |
| Each copy into the sample stays in bounds | `buffer_index` preconditions in `gather` and `read_batch` | `gather_destination` fails with E8006 |
| Buffer sizes do not overflow | Tested, not proved. The prover ignores `/` and cannot use the `u64` constants | none |

### Audit

`vxc src/kmeans.vx --audit` lists 46 trusted clauses:

- 28 at `vxvector_kmeans_v2`. These are the 24 from before, plus 4 for the two scratch buffers.
- 16 at `vxvector_run_cpu_reason_v1`, for the buffers that `src/data_path.c` allocates.
- The same 2 `!overlaps` clauses in `run_restart`.

The plan has none. The audit lists only clauses that calls trust. It does not list reads through a pointer that has no facts, such as the reader's row values in `read_batch`. Those reads rely on the reader contract in the header.

## Vx findings

1. **An assigned call result loses its `ensures`: [vx-lang/Vx#3040](https://github.com/vx-lang/Vx/issues/3040), open.** `let r = f(p, ..);` keeps `f`'s postcondition. `x = f(p, ..);` drops it when the prover cannot model an argument, such as a pointer or a pointer cast. The `gather` call passes pointer casts.
   The workaround in `vxvector_run_cpu_reason_v1` checks `selected` after `gather`. This reproduction still fails on `df51035d` with E8001:

   ```vx
   unsafe fn bounded(values: *const i64, limit: i64, first: i64) -> i64
   requires limit > 0i64 && first >= 0i64 && first <= limit
   requires valid(values, limit)
   requires initialized(values, limit)
   ensures return >= 0i64 && return <= limit
   {
     return first;
   }

   unsafe fn caller(values: *const i64, limit: i64) -> i64
   requires limit > 0i64
   requires valid(values, limit)
   requires initialized(values, limit)
   ensures return >= 0i64 && return <= limit
   {
     let mut got = 0i64;
     got = unsafe { bounded(values, limit, got) };
     return got;
   }
   ```

   A call with only number arguments keeps its `ensures` when assigned, as #3040 describes.
2. **vx-lang/Vx#3145 is still open on `df51035d`.** A float literal in a `let` inside a function with a loop invariant crashes the prover too. The new `no_seconds` helper works around it, like `positive`.
3. **Known gaps hit again.**
   - A fact ends with the scope of a variable it names. `sift_down` and the run loop declare `child`, `right`, and `height` outside their loops.
   - `i64::from(..)` in a contract is an opaque call. Contracts use `as` casts and accept lint W1007.
4. **Two unlocated warnings.** The build warns twice that the prover ignores a condition with a cast of unknown integer type. The warning has no line number.
5. **What worked without workarounds.**
   - Calling the C reader through `fn(*mut u8, *mut InputRow) -> i32`.
   - Reading and writing C structs through pointers with matching layouts.
   - `u64` wrapping arithmetic for SplitMix64. Vx has no hexadecimal literals, so the constants are decimal.

## Not covered

- Docker was not running.
  - The Linux worker, the extension build, the PostgreSQL image, the SQL suite, the snapshot check, and real-data recall were not rebuilt or rerun.
  - [pg_vxvector/build.rs](../pg_vxvector/build.rs) and [scripts/build_extension.sh](../scripts/build_extension.sh) now link `build/linux/kmeans.o` into the extension. That build is untested.
  - The plan runs Vx code inside the PostgreSQL backend. A failed Vx runtime check there calls `abort`.
- Release-profile and `-O3` builds were not checked.
- During this step, the shared Vx checkout was fast-forwarded from `7e3cf6a3` to `df51035d` outside this work. The pin follows it, and every result above uses `df51035d`.
