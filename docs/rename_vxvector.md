# VxVector naming

The project is **VxVector**. The PostgreSQL extension is **`pg_vxvector`**.
The GitHub repository and local checkout directory retain their existing `vextor` locations.

| Surface | Previous name | Current name |
| --- | --- | --- |
| PostgreSQL extension | `pg_vextor` | `pg_vxvector` |
| SQL function | `vextor.create_vector_index` | `vxvector.create_vector_index` |
| Worker setting | `vextor.worker_path` | `vxvector.worker_path` |
| C ABI prefix | `vextor_` | `vxvector_` |
| Environment variables | `VEXTOR_*` | `VXVECTOR_*` |
| Native library | `libvextor_kmeans` | `libvxvector_kmeans` |
| Worker executable | `vextor-worker` | `vxvector-worker` |
| Extension source directory | `pg_vextor/` | `pg_vxvector/` |
| Compose project | `vextor` | `vxvector` |
| PostgreSQL database and role | `vextor` | `vxvector` |
| Worker installation directory | `/usr/local/libexec/vextor/` | `/usr/local/libexec/vxvector/` |

Use a fresh database with the renamed image and configuration:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vchord;
CREATE EXTENSION IF NOT EXISTS pg_vxvector;
```

The rename does not provide SQL aliases or an automatic upgrade from `pg_vextor`.
Existing application SQL and environment variable names must use the new names.
The binary worker protocol retains its version and layout.

Historical evidence retains its original contents and measured hashes.
Historical validation reports identify their earlier installation names explicitly.
The objective parser accepts the previous capture marker when replaying those records.
New objective captures use `VXVECTOR_CENTROIDS`.

## Validation

The renamed native library and worker passed 35 unit tests, 48 Faiss comparisons,
96 independent Lloyd comparisons, six edge cases, and allocation/sanitizer checks.
The native contract and generated-code mutation checks passed.
The renamed Rust/C extension compiled successfully for PostgreSQL 17 on Linux.

The fresh `vxvector-postgres-1` service passed extension installation, SQL index
builds, worker isolation/cancellation, rollback, and snapshot checks.
All 120 synthetic recall queries passed across 24 index builds, with minimum recall 1.0.
Native and Linux contract and code-generation mutation checks passed.

The Linux builder's Z3 4.8.12 timed out on an existing variable-offset proof.
This validation updated that build container to Z3 5.1.0, matching the native solver.
The compiler source and five-second proof budget were unchanged.
Build environments using the older solver may need the same update.
The Linux compiler was built from the pinned Git archive; its revision and binary
hash are recorded separately because that archive build prints an unstamped version.
Real-dataset comparisons and the full compiler suite were not rerun.

See [installed metadata and build hashes](evidence/rename_vxvector_2026-11-05.json),
[native checks](evidence/rename_vxvector_native_2026-11-05.log),
[PostgreSQL integration](evidence/rename_vxvector_postgres_2026-11-05.log), and
[synthetic recall](evidence/rename_vxvector_recall_2026-11-05.jsonl).
