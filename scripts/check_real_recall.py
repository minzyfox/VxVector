"""Compare Vx and internal VectorChord builds on real ANN-Benchmarks data.

Fixtures live in one rolled-back transaction. Conservative recall@10 must be
at least 0.95 per query. At each practical setting, Vx mean recall may trail
internal training by at most 0.05 across the specified repeats and queries.
These fixed gates are correctness checks, not production performance claims.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

import h5py
import numpy as np

from objective_comparison import compare_captures
from check_centroid_capture import reference_build, parse_captures


def vector(row):
    return '[' + ','.join(format(float(x), '.9g') for x in row) + ']'


def load(path, rows, queries, normalize):
    with h5py.File(path, 'r') as dataset:
        # A fixed random subset avoids relying on the publisher's row ordering.
        indices = np.sort(np.random.default_rng(731).choice(len(dataset['train']), rows, replace=False))
        points = np.asarray(dataset['train'][indices], dtype='float32')
        held_out = np.asarray(dataset['test'][:queries], dtype='float32')
    if normalize:
        for values in (points, held_out):
            norms = np.linalg.norm(values.astype('float64'), axis=1, keepdims=True)
            if np.any(norms == 0):
                raise AssertionError('Real angular fixtures must have nonzero norms')
            values[:] = values / norms
    if not np.isfinite(points).all() or not np.isfinite(held_out).all():
        raise AssertionError('Dataset contains nonfinite components')
    return points, held_out, indices


def check_results(records, seeds, queries):
    expected = 3 * 2 * len(seeds) * 2 * 4 * queries
    if len(records) != expected:
        raise AssertionError(f'Expected {expected} real-data queries, found {len(records)}')
    groups = {}
    for row in records:
        if row['conservative'] and row['recall'] < 0.95:
            raise AssertionError(f'Conservative recall below 0.95: {row}')
        key = (row['dataset'], row['metric'], row['roots'], row['probes'], row['epsilon'])
        groups.setdefault(key, {}).setdefault(row['trainer'], []).append(row['recall'])
    summaries = []
    for key, trainers in groups.items():
        vx, internal = (float(np.mean(trainers[name])) for name in ('vx', 'internal'))
        summaries.append(dict(dataset=key[0], metric=key[1], roots=key[2], probes=key[3],
                              epsilon=key[4], vx_mean=vx, internal_mean=internal, difference=vx-internal))
    for row in summaries:
        print(json.dumps(dict(summary=row)), flush=True)
    failures = [row for row in summaries if row['difference'] < -0.05 - 1e-12]
    if failures:
        raise AssertionError(f'Vx mean recall trails internal training by over 0.05: {failures}')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=Path('build/datasets'))
    parser.add_argument('--container', default=os.environ.get('VXVECTOR_POSTGRES_CONTAINER', 'vxvector-postgres-1'))
    parser.add_argument('--rows', type=int, default=8192)
    parser.add_argument('--queries', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 17])
    parser.add_argument('--capture-objectives', action='store_true',
                        help='Require centroid exports from the instrumented VectorChord test image')
    args = parser.parse_args()
    if args.rows < 128 or args.queries < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Use at least 128 rows, positive queries, and distinct nonnegative seeds')
    if args.capture_objectives:
        print(json.dumps(dict(objective_reference=reference_build(args.container))), flush=True)
    schema = 'vxvector_real_' + uuid.uuid4().hex[:12]
    sql = [f'''BEGIN ISOLATION LEVEL REPEATABLE READ;
      SET LOCAL statement_timeout = '300s'; SET LOCAL jit = off;
      SET LOCAL max_parallel_workers_per_gather = 0;
      SET LOCAL max_parallel_maintenance_workers = 0;
      SET LOCAL vchordrq.max_scan_tuples = -1;
      CREATE SCHEMA {schema};
      SELECT json_build_object('postgres', version(), 'extensions',
        (SELECT json_object_agg(extname, extversion) FROM pg_extension));
    ''']
    objective_cases = {}
    if args.capture_objectives:
        sql.append("SET LOCAL client_min_messages = notice; SET LOCAL vxvector_test.capture_centroids = 'on';")
    print(json.dumps(dict(platform=platform.platform(), numpy=np.__version__, h5py=h5py.__version__,
                          rows=args.rows, queries=args.queries, seeds=args.seeds,
                          conservative_minimum=0.95, maximum_mean_recall_loss=0.05)), flush=True)
    for dataset, metrics in [('glove-25-angular', [('cos', '<=>'), ('ip', '<#>')]),
                             ('sift-128-euclidean', [('l2', '<->')])]:
        path = args.datasets / (dataset + '.hdf5')
        points, held_out, indices = load(path, args.rows, args.queries, dataset.startswith('glove'))
        print(json.dumps(dict(dataset=dataset, file_sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                              points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                              queries_sha256=hashlib.sha256(held_out.tobytes()).hexdigest(),
                              indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                              dims=points.shape[1], normalized=dataset.startswith('glove'))), flush=True)
        table = f'{schema}.items'
        sql.append(f'CREATE TABLE {table}(id integer PRIMARY KEY, features vector({points.shape[1]}));')
        sql.append(f'COPY {table} FROM STDIN;\n' + '\n'.join(f'{i}\t{vector(row)}' for i, row in enumerate(points)) + '\n\\.\n')
        sql.append(f'CREATE TABLE {schema}.queries(id integer PRIMARY KEY, features vector({points.shape[1]}));')
        sql.append(f'COPY {schema}.queries FROM STDIN;\n' + '\n'.join(f'{i}\t{vector(row)}' for i, row in enumerate(held_out)) + '\n\\.\n')
        sql.append(f'ANALYZE {table};')
        for metric, operator in metrics:
            spherical = str(metric != 'l2').lower()
            sql.append(f'''SET LOCAL enable_indexscan = off; SET LOCAL enable_indexonlyscan = off;
              SET LOCAL enable_bitmapscan = off; SET LOCAL enable_seqscan = on;
              CREATE TABLE {schema}.truth AS
              SELECT q.id AS query, n.* FROM {schema}.queries q CROSS JOIN LATERAL
                (SELECT i.id, i.features {operator} q.features AS distance FROM {table} i
                 ORDER BY distance, i.id LIMIT 10) n;
              DO $plan$ DECLARE p json; BEGIN
                EXECUTE 'EXPLAIN (FORMAT JSON) SELECT id FROM {table}
                  ORDER BY features {operator} ''{vector(held_out[0])}''::vector LIMIT 10' INTO p;
                IF position('Seq Scan' IN p::text) = 0 OR position('Index Scan' IN p::text) > 0 THEN
                  RAISE EXCEPTION 'Exact ground truth must use sequential scans: %', p;
                END IF;
              END $plan$;
              SET LOCAL enable_indexscan = on; SET LOCAL enable_seqscan = off;
            ''')
            for roots in (0, 4):
                lists = '32' if roots == 0 else '4,32'
                settings = [('8' if roots == 0 else '2,8', 1.9, False),
                            ('16' if roots == 0 else '4,16', 1.9, False),
                            ('32' if roots == 0 else '4,32', 1.9, False),
                            ('32' if roots == 0 else '4,32', 4.0, True)]
                for seed in args.seeds:
                    case = f'{dataset}_{metric}_{roots}_{seed}'
                    if args.capture_objectives:
                        objective_cases[case] = dict(dataset=dataset, metric=metric, roots=roots,
                                                     seed=seed, points=points)
                    for trainer in ('vx', 'internal'):
                        if args.capture_objectives:
                            sql.append(f"SELECT set_config('vxvector_test.case','{case}',true);")
                        # Only one ANN index exists at a time, so the checked plan cannot choose its competitor.
                        if trainer == 'vx':
                            build = f'''SELECT vxvector.create_vector_index('{table}', 'features', ARRAY[{lists}],
                              sampling_factor=>40, batch_size=>127, kmeans_iterations=>10, kmeans_nredo=>1,
                              distance_operator=>'{metric}', spherical_centroids=>{spherical}, seed=>{seed},
                              centroid_table=>'centers', index_name=>'candidate');'''
                        else:
                            build = f'''CREATE INDEX candidate ON {table} USING vchordrq(features vector_{metric if metric != 'cos' else 'cosine'}_ops)
                              WITH(options=$options$residual_quantization = false
                              [build.internal]
                              lists = [{lists}]
                              sampling_factor = 40
                              kmeans_iterations = 10
                              spherical_centroids = {spherical}
                              build_threads = 1
                              kmeans_algorithm.lloyd = {{}}
                              $options$);'''
                        sql.append('\\timing on\n' + build + '\n\\timing off\n')
                        if args.capture_objectives and trainer == 'vx':
                            levels = [f'(SELECT json_agg(vector::real[] ORDER BY id) FROM {schema}.centers WHERE id > {roots})']
                            if roots:
                                levels.append(f'(SELECT json_agg(vector::real[] ORDER BY id) FROM {schema}.centers WHERE id BETWEEN 1 AND {roots})')
                            levels.append(f'(SELECT json_agg(vector::real[] ORDER BY id) FROM {schema}.centers WHERE id = 0)')
                            sql.append(f"SELECT 'VXVECTOR_CENTROIDS ' || json_build_object('trainer','vx','case','{case}', 'levels',json_build_array({','.join(levels)}))::text;")
                        for probes, epsilon, conservative in settings:
                            sql.append(f'''SET LOCAL vchordrq.probes = '{probes}';
                              SET LOCAL vchordrq.epsilon = '{epsilon}';
                              DO $recall$
                              DECLARE q record; p json; ids integer[]; ds double precision[];
                                exact_ids integer[]; cutoff double precision; tol double precision;
                                strict_total integer; strict_hits integer; boundary_hits integer;
                                recall double precision; query_seconds double precision; started timestamptz;
                              BEGIN
                                FOR q IN SELECT * FROM {schema}.queries ORDER BY id LOOP
                                  EXECUTE format('EXPLAIN (FORMAT JSON) SELECT id FROM {table}
                                    ORDER BY features {operator} %L::vector LIMIT 10',q.features) INTO p;
                                  IF position('"Index Name": "candidate"' IN p::text) = 0 THEN
                                    RAISE EXCEPTION 'Expected candidate index: %',p;
                                  END IF;
                                  started := clock_timestamp();
                                  EXECUTE format('SELECT array_agg(id ORDER BY distance,id), array_agg(distance ORDER BY distance,id)
                                    FROM (SELECT id, features {operator} %L::vector AS distance
                                    FROM {table} ORDER BY distance LIMIT 10) n',q.features) INTO ids,ds;
                                  query_seconds := extract(epoch FROM clock_timestamp()-started);
                                  IF cardinality(ids) IS DISTINCT FROM 10 OR
                                     (SELECT count(DISTINCT x) FROM unnest(ids) x) <> 10 THEN
                                    RAISE EXCEPTION 'Index must return ten distinct IDs';
                                  END IF;
                                  SELECT max(distance),array_agg(id ORDER BY distance,id) INTO cutoff,exact_ids
                                    FROM {schema}.truth WHERE query=q.id;
                                  tol := 1e-6 + 1e-5 * abs(cutoff);
                                  SELECT count(*),count(*) FILTER(WHERE id=ANY(ids)) INTO strict_total,strict_hits
                                    FROM {schema}.truth WHERE query=q.id AND distance < cutoff-tol;
                                  SELECT count(*) INTO boundary_hits FROM unnest(ds) d WHERE abs(d-cutoff)<=tol;
                                  recall := (strict_hits + least(10-strict_total,boundary_hits))/10.0;
                                  RAISE NOTICE 'REAL_RESULT %',json_build_object('dataset','{dataset}', 'metric','{metric}',
                                    'roots',{roots}, 'seed',{seed}, 'trainer','{trainer}', 'query',q.id,
                                    'probes','{probes}', 'epsilon',{epsilon}, 'conservative',{str(conservative).lower()},
                                    'recall',recall, 'exact_ids',exact_ids, 'index_ids',ids,
                                    'query_seconds',query_seconds);
                                END LOOP;
                              END $recall$;
                            ''')
                        sql.append(f'DROP INDEX {schema}.candidate;')
                        if trainer == 'vx':
                            sql.append(f'DROP TABLE {schema}.centers;')
            sql.append(f'DROP TABLE {schema}.truth;')
        sql.append(f'DROP TABLE {table}; DROP TABLE {schema}.queries;')
    sql.append('ROLLBACK;')
    result = subprocess.run(['docker', 'exec', '-i', args.container, 'psql', '-X', '-At',
                             '-v', 'ON_ERROR_STOP=1', '-U', 'vxvector', '-d', 'vxvector'],
                            input='\n'.join(sql), text=True, capture_output=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    records = [json.loads(line.split('REAL_RESULT ', 1)[1])
               for line in result.stderr.splitlines() if 'REAL_RESULT ' in line]
    summaries = check_results(records, args.seeds, args.queries)
    if args.capture_objectives:
        captures = parse_captures(result.stdout + '\n' + result.stderr)
        objective_summary = compare_captures(captures, objective_cases)
        print(json.dumps(dict(objectives=objective_summary)), flush=True)
    print(json.dumps(dict(result='passed', builds=3*2*len(args.seeds)*2,
                          recall_queries=len(records), setting_comparisons=len(summaries))), flush=True)


if __name__ == '__main__':
    main()
