# Internal VectorChord objective comparison

## What this checks

This optional M4 check compares Vx leaf-centroid quality with VectorChord 1.1.1's
internal trainer. Both centroid sets are scored independently on the same source
vectors. The existing exact-scan recall checks also run.

The reference image exports centroid levels after internal clustering and before
VectorChord rotates them for index storage. The export is an opt-in PostgreSQL
NOTICE. It does not modify centroid values or the training algorithm.
The source archive and capture-point hashes are checked before patching.

This is a source build with Rust 1.93.0, the pg17 feature, and optimization/debug
levels zero. The ordinary packaged VectorChord binary has a different build profile.
The reference build manifest records that distinction and the exact library hash.
The primary development database keeps its ordinary image.

## Objective and acceptance gate

For each source vector, select the leaf centroid with the smallest distance.
Accumulate costs in float64, in bounded batches, without calling Vx or Faiss.
The raw mean objective uses squared L2, negative inner product, or cosine distance.

The quality gate uses a nonnegative cost. L2 and cosine use their raw mean costs.
For inner product, it uses `1 + mean negative dot product` on unit-normalized
source vectors and spherical centroids. This avoids comparing ratios of negative objectives.
Tiny negative costs from rounding are clamped to zero.

The gate was defined before measuring the reference builds:

```text
mean Vx cost <= 1.10 * mean internal-trainer cost + 1e-6
```

Means cover the two runs per dataset, metric, and tree shape. This permits a 10%
relative increase with an absolute tolerance. It is a bounded regression gate,
not a statistical equivalence result or a proof of optimal clustering.

Inputs are the same GloVe-25 and SIFT-128 subsets used by the real-data recall check:
8,192 source rows and 32 held-out queries per dataset. GloVe covers cosine and
inner product. SIFT covers L2. Both tree shapes use 32 leaves; two-level trees
request four coarse centroids. Both trainers use sampling factor 40, ten iterations,
and one training attempt. The full source subset is scored, rather than either
trainer's distinct sample.

Vx uses seeds 0 and 17. Internal clustering keeps VectorChord's initializer seed
`[7; 32]`, with independent source sampling on each build. These are paired quality
comparisons, not matched-initialization differential tests.
Coarse-stage objectives are recorded on each trainer's own leaves as diagnostics.
Those leaf sets differ, so the coarse-stage values are not a matched training problem.
Singleton routing roots are excluded from the quality gate; Vextor uses a zero root.

## Run the comparison

First build the ordinary database image and prepare the Python environment and
GloVe/SIFT files using [the README](../README.md#run-tests).
Use the same Docker context. On the current development machine it is `colima`.

Build the separate reference image. Its first build installs a pinned Rust toolchain
inside the build container and compiles VectorChord dependencies. This took about
14 minutes on the emulated development setup; subsequent builds reuse Cargo caches:

```sh
./scripts/build_objective_reference.sh
```

This creates `vextor-postgres:17-objectives`. It does not replace the ordinary image.
Start a separate database on port 15433 with its own Compose project and data volume:

```sh
VEXTOR_POSTGRES_IMAGE=vextor-postgres:17-objectives VEXTOR_POSTGRES_PORT=15433 \
  docker-compose -p vextor-objectives up -d --no-build --wait
python3 -B scripts/check_centroid_capture.py --container vextor-objectives-postgres-1
build/correctness-venv/bin/python -B scripts/check_real_recall.py \
  --container vextor-objectives-postgres-1 --capture-objectives
```

Use `docker compose` if the Compose plugin is installed.
The capture check verifies that export is disabled by default, then checks a known
unrotated mean of `[1,0]` at both captured levels. The full comparison requires
24 centroid captures, six objective-quality groups, and 3,072 recall records.
Missing, duplicate, nonfinite, malformed, or unnormalized angular captures fail.
Both checks roll back their SQL fixtures.

The generated build files remain under ignored `build/objectives/`.
Its `reference-build.json` is also packaged inside the reference image.
The harness verifies the installed library hash against that manifest before testing.

Run only the objective scorer's focused tests without Docker:

```sh
build/correctness-venv/bin/python -B -m unittest discover -s tests -p test_objectives.py -v
```

Stop the reference database while retaining its data volume:

```sh
VEXTOR_POSTGRES_IMAGE=vextor-postgres:17-objectives VEXTOR_POSTGRES_PORT=15433 \
  docker-compose -p vextor-objectives down
```

## Results

On November 4, 2026, all six quality groups passed using 24 captured centroid
sets. The largest Vx mean cost increase was 0.43%, below the predefined 10% gate.
The same 24 builds passed all 3,072 recall queries and 24 recall-setting comparisons.
These results apply to the bounded fixtures and reference build described above.

| Dataset / metric | Coarse centroids | Vx mean cost | Internal mean cost |
| --- | ---: | ---: | ---: |
| GloVe / cosine | 0 | 0.346668 | 0.349190 |
| GloVe / cosine | 4 | 0.346668 | 0.347330 |
| GloVe / inner product | 0 | 0.346668 | 0.345576 |
| GloVe / inner product | 4 | 0.346668 | 0.345189 |
| SIFT / squared L2 | 0 | 73686.045 | 73746.266 |
| SIFT / squared L2 | 4 | 73686.045 | 73977.651 |

The initial database run completed and rolled back its fixtures, then exposed a
parser bug: PostgreSQL's `json_agg` formatted Vx captures over multiple lines.
The corrected parser scored the saved captures without rebuilding the indexes.
A regression test covers both multiline SQL output and NOTICE records, and the
live known-answer capture check passed again after the correction.

Ten scorer/parser tests pass. Three temporary mutations failed their intended
assertions: farthest-centroid scoring, incorrect inner-product cost conversion,
and a disabled quality gate. The native suite also passed its 25 existing tests,
48 Faiss cases, 96 Lloyd cases, six edge cases, and sanitizer checks.

Evidence:

- [Raw database capture, including the initial parser failure](evidence/m4_objective_capture_2026-11-04.txt).
- [Corrected scoring and recall results](evidence/m4_objectives_2026-11-04.jsonl)
  and [the exact replay command](evidence/m4_objective_replay_2026-11-04.txt).
- [Reference build manifest](evidence/m4_objective_build_2026-11-04.json).
- [Live capture smoke check](evidence/m4_objective_smoke_2026-11-04.json).
- [Scorer/parser tests](evidence/m4_objective_scorer_2026-11-04.txt),
  [mutation checks](evidence/m4_objective_mutations_2026-11-04.txt), and
  [native regression log](evidence/m4_objective_native_2026-11-04.txt).
