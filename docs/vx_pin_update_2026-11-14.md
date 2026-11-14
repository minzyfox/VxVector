# Vx pin update to 423a10db (November 14, 2026)

This record moves the Vx pin from `df51035d` to `423a10db`, 66 commits later on Vx `main`.
It revalidates the native macOS checks of the [Step 2 record](step2_data_path_vx_2026-11-11.md).
No VxVector code changed. One comment in `src/kmeans.vx` now names the new revision.
Standalone macOS checks only. The Linux worker, the extension, PostgreSQL, and recall were not rerun.

## Revisions

| Item | Value |
| --- | --- |
| Vx | `423a10dbbd91230df290494f9705a746dc444baf` (main, November 14). The pin moves from `df51035d`. |
| Compiler | `vxc 0.0.4 (423a10db)`, debug profile, built with `VX_VERSION_WITH_COMMIT=1` |
| Compiler SHA-256 | `18889bbc6e837615e38f3ef6c76ceb50a78f909931645132e43670e83b61a6b3` |
| `src/kmeans.vx` SHA-256 | `53f3962d49951ab26b11b878e06db9c79904bcb741f42ea53febb88cc1179a82` |
| `kmeans.o` SHA-256 | `b7538c44239f892601b15c1f60b88d8de1a58f1237833799e37e4af9293f9833`, the same as Step 2 |
| `libvxvector_kmeans.dylib` SHA-256 | `36d3121c3b458c34ceaf4e6119a638f294bf0fd810a3a42524f560178c554008` |
| Platform | macOS 27.2 arm64 |

The [metadata file](evidence/pin_update_metadata_2026-11-14.json) lists every hash.

The new compiler emits the same `kmeans.o`, byte for byte, as `df51035d` did.
The kernel's machine code is unchanged, so its results are the same as in Step 2.
The library hash changes because it links Vx's rebuilt `libvx_std_core.dylib`.

## Results

All checks passed. Logs are in [docs/evidence](evidence/).

| Check | Result |
| --- | --- |
| Native build ([scripts/build.sh](../scripts/build.sh)) | Built, flat code generator |
| Native unit tests ([log](evidence/pin_update_cpu_2026-11-14.txt)) | 37 ran, OK |
| Matched Faiss, independent float64 Lloyd, zero/duplicate/empty-cluster cases | 48, 96, and 6 passed |
| Allocation failure, ASan, UBSan | Passed |
| Contract helpers ([log](evidence/pin_update_contracts_2026-11-14.jsonl)) | 5 proofs, 6 mutations |
| Code generation and audit ([log](evidence/pin_update_codegen_2026-11-14.jsonl)) | 12 cases passed |
| `vxc src/kmeans.vx --audit` ([log](evidence/pin_update_audit_2026-11-14.txt)) | The same 46 trusted clauses as Step 2 |

The two new checker rules, a cast from a number to `bool` (vx-lang/Vx#3405) and unary minus on an unsigned integer (#3404), do not affect the kernel.

## Step 2 findings at this revision

The reproductions are in [pin_update_repros_2026-11-14](evidence/pin_update_repros_2026-11-14/). Their output is in [the repro log](evidence/pin_update_repros_2026-11-14.txt).

| Step 2 finding | At `423a10db` |
| --- | --- |
| An assigned call result loses its `ensures` ([vx-lang/Vx#3040](https://github.com/vx-lang/Vx/issues/3040)) | Open. The reproduction fails with E8001. The check after `gather` stays. |
| A float literal or float comparison crashes the prover ([vx-lang/Vx#3145](https://github.com/vx-lang/Vx/issues/3145)) | Open. Both reproductions panic in z3. `positive` and `no_seconds` stay. |
| Two prover warnings have no line number | Fixed. They point at `src/kmeans.vx:249` (`/`) and `src/kmeans.vx:556` (a cast of unknown integer type). |
| Facts end with the scope of a variable they name. `i64::from(..)` is opaque in contracts | Not rechecked. Lint W1007 still fires. |

## Vx changes relevant to later steps

- **Apple GPU loops (#3261).** The Metal printer writes a counted kernel loop as a `for` loop instead of `while (true)` with a `break`, so the Metal compiler can unroll it. This matters for Step 3.
- **Struct fields behind an indexed pointer (#3366).** `p[i].f` now reads and writes the element. It used to crash the compiler. The kernel copies C structs into locals and does not need this.

## Build notes

- Run the Vx build under `arch -arm64` when the shell runs under Rosetta. Otherwise `xcrun` cannot load and `vxc` fails to link.
- `cargo build` of the whole Vx workspace also failed to link `cargo-vx-bench` under Rosetta. `cargo build --bin vxc` is enough for VxVector.

## Not covered

- The Linux worker, the extension build, the PostgreSQL image, the SQL suite, the snapshot check, and real-data recall were not rebuilt or rerun. The Step 2 gap remains.
- Release-profile and `-O3` builds were not checked.
