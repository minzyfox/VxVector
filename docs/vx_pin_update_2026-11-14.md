# Vx pin update to 423a10db (November 14, 2026)

This record moves the Vx pin from `df51035d` to `423a10db`, 66 commits later on Vx `main`.
It revalidates the native macOS checks of the [Step 2 record](step2_data_path_vx_2026-11-11.md).
No VxVector code changed. One comment in `src/kmeans.vx` now names the new revision.
It also rebuilds the Linux worker and the extension and reruns PostgreSQL and recall. See [Linux and PostgreSQL](#linux-and-postgresql).

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

## Linux and PostgreSQL

This section closes the Linux item that Step 2 left open.
The Linux worker, the extension, and the PostgreSQL image were rebuilt at `423a10db` and checked.
Docker Desktop 28.5.1 ran the `linux/amd64` containers under emulation on macOS arm64.
The [Linux metadata file](evidence/linux_metadata_2026-11-14.json) lists every hash.

| Item | Value |
| --- | --- |
| Linux `kmeans.o` SHA-256 | `67fdbd582304d3b8fea869431a862b94f605c03b3b3bec22b401b717ab66e3f2`, flat code generator |
| Linux compiler SHA-256 | `b4b5f467eeeb846b4765fb30389d9141d06d651cdc19cba06a8853bf778d52b0` |
| `pg_vxvector.so` SHA-256 | `5dde9970fa442159c871d5145cc73ff38a35d58e2622326a2ac018a0ce933fd2` |
| Installed worker SHA-256 | `e339fc0fa34354d48499f0282b3649f31ba7fb67c37bd7240fe345e70df09a64` |

The Linux compiler reports `vxc 0.0.4` with no commit. The build container receives a `git archive` without `.git`, so the commit stamp is empty. The build info records the pin separately.

The extension now links the Vx object, so the plan's Vx code runs inside the PostgreSQL backend. This build is the first since that change. It compiled and linked without errors, and every check below passed.

| Check | Result |
| --- | --- |
| Image checks ([log](evidence/linux_postgres_2026-11-14.txt)) | Passed |
| Extension, worker isolation, cancellation, rollback ([log](evidence/linux_extension_2026-11-14.txt)) | Passed |
| Snapshot ([log](evidence/linux_snapshot_2026-11-14.txt)) | Passed. Concurrent inserts and updates were excluded |
| Synthetic SQL recall ([log](evidence/linux_recall_2026-11-14.txt)) | 24 index builds, 2 centroid-only builds, 120 queries, minimum recall 1.0 |
| Real-data recall, GloVe-25 and SIFT-128 ([log](evidence/linux_real_recall_2026-11-14.jsonl)) | 24 builds, 3,072 queries, 24 setting comparisons. All gates passed |

Step 2 changed the centroids, so real-data recall was the check most likely to move. It moved slightly up:

| Over the 24 settings | Milestone 4 ([log](evidence/m4_full_2026-11-04.txt)) | This run |
| --- | --- | --- |
| Lowest Vx mean recall | 0.9328 | 0.9422 |
| Mean Vx recall | 0.9900 | 0.9919 |
| Largest shortfall against VectorChord's internal training | 0.0109 | 0.0016 |

### Image platform fix

The image build failed on this machine. BuildKit looked for an `arm64` variant of the local base image `vxvector-postgres:17-dev`, which exists only for `amd64`.
[scripts/build_extension.sh](../scripts/build_extension.sh) now passes `--platform` with the base image's own platform.
Real-data recall ran on an image built with `DOCKER_DEFAULT_PLATFORM=linux/amd64` before that fix.
The image from the fixed script installs the same worker, library, and extension files, byte for byte. The image, extension, snapshot, and synthetic recall checks passed on both images.

## Not covered

- Internal objective comparisons ([internal_objectives.md](internal_objectives.md)) were not rerun.
- Release-profile and `-O3` builds were not checked.
