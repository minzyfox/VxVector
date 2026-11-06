"""Check CPU SQL builds against exact scans in one rolled-back snapshot.

The fixed synthetic suite requires tie-aware recall@10 >= 0.95 per query.
All leaves are probed. This is a correctness baseline, not a speed claim.
"""

import argparse
import json
import os
import platform
import subprocess
import uuid


MIN_RECALL = 0.95


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default=os.environ.get('VXVECTOR_POSTGRES_CONTAINER', 'vxvector-postgres-1'))
    parser.add_argument('--epsilon', type=float, default=4.0, help='VectorChord reranking quality (0..4)')
    args = parser.parse_args()
    if not 0 <= args.epsilon <= 4:
        parser.error('epsilon must be between 0 and 4')
    schema = 'vxvector_m4_' + uuid.uuid4().hex[:12]
    statements = [f"""
        BEGIN ISOLATION LEVEL REPEATABLE READ;
        SET LOCAL statement_timeout = '120s';
        SET LOCAL jit = off;
        SET LOCAL vchordrq.epsilon = '{args.epsilon}';
        SET LOCAL vchordrq.max_scan_tuples = -1;
        CREATE SCHEMA {schema};
        SELECT json_build_object('postgres', version(), 'extensions',
          (SELECT json_object_agg(extname, extversion) FROM pg_extension),
          'minimum_recall', {MIN_RECALL}, 'epsilon', {args.epsilon}, 'fixture', 'directional-sine-v1');
    """]
    for dims in (128, 768, 2000, 3072):
        statements.append(f"""
          CREATE TABLE {schema}.source_{dims} (id integer PRIMARY KEY, features vector({dims}));
          INSERT INTO {schema}.source_{dims}
            SELECT i, ARRAY(SELECT (CASE WHEN d = i % 4 + 1 THEN 3.0 ELSE 0.0 END
                + sin(i * 13.0 + d * 7.0) * 0.1)::real FROM generate_series(1,{dims}) d)::vector
            FROM generate_series(1,256) i;
          ANALYZE {schema}.source_{dims};
        """)
        for metric, operator in (('l2', '<->'), ('ip', '<#>'), ('cos', '<=>')):
            for roots in (0, 2):
                # Vary each option while keeping the matrix small enough for emulation.
                seed = 0 if roots == 0 else 17
                factor = 8 if roots == 0 else 40
                batch = 1 if roots == 0 else 13
                restarts = 1 if roots == 0 else 3
                lists = '4' if roots == 0 else '2,4'
                probes = '4' if roots == 0 else '2,4'
                tag = f'{metric}_{roots}_{dims}'
                spherical = 'false' if metric == 'l2' else 'true'
                statements.append(f"""
                  SELECT vxvector.create_vector_index('{schema}.source_{dims}', 'features', ARRAY[{lists}],
                    sampling_factor=>{factor}, batch_size=>{batch}, kmeans_iterations=>8,
                    kmeans_nredo=>{restarts}, distance_operator=>'{metric}',
                    spherical_centroids=>{spherical}, seed=>{seed},
                    centroid_table=>'centers_{tag}', index_name=>'idx_{tag}');
                  SET LOCAL vchordrq.probes = '{probes}';
                  DO $check$
                  DECLARE
                    query vector;
                    plan json;
                    exact_ids integer[];
                    exact_distances double precision[];
                    found_ids integer[];
                    found_distances double precision[];
                    cutoff double precision;
                    epsilon double precision;
                    strict_total integer;
                    strict_hits integer;
                    boundary_hits integer;
                    recall double precision;
                  BEGIN
                    IF (SELECT count(*) FROM {schema}.centers_{tag}) <> {5 + roots} THEN
                      RAISE EXCEPTION 'Wrong centroid node count';
                    END IF;
                    FOR q IN 1..5 LOOP
                      SELECT ARRAY(SELECT (CASE WHEN d = q % 4 + 1 THEN 3.0 ELSE 0.0 END
                        + sin(q * 19.0 + d * 11.0) * 0.1)::real
                        FROM generate_series(1,{dims}) d)::vector INTO query;
                      PERFORM set_config('enable_indexscan','off',true);
                      PERFORM set_config('enable_indexonlyscan','off',true);
                      PERFORM set_config('enable_bitmapscan','off',true);
                      PERFORM set_config('enable_seqscan','on',true);
                      EXECUTE format('EXPLAIN (FORMAT JSON) SELECT id FROM {schema}.source_{dims}
                        ORDER BY features {operator} %L::vector LIMIT 10', query) INTO plan;
                      IF position('Seq Scan' IN plan::text) = 0 OR position('Index Scan' IN plan::text) > 0 THEN
                        RAISE EXCEPTION 'Ground truth did not use an exact sequential scan: %', plan;
                      END IF;
                      EXECUTE format('SELECT array_agg(id ORDER BY distance,id),
                        array_agg(distance ORDER BY distance,id) FROM
                        (SELECT id, features {operator} %L::vector AS distance
                         FROM {schema}.source_{dims} ORDER BY distance,id LIMIT 10) nearest', query)
                        INTO exact_ids, exact_distances;
                      PERFORM set_config('enable_indexscan','on',true);
                      PERFORM set_config('enable_indexonlyscan','on',true);
                      PERFORM set_config('enable_seqscan','off',true);
                      EXECUTE format('EXPLAIN (FORMAT JSON) SELECT id FROM {schema}.source_{dims}
                        ORDER BY features {operator} %L::vector LIMIT 10', query) INTO plan;
                      IF position('idx_{tag}' IN plan::text) = 0 THEN
                        RAISE EXCEPTION 'Recall query did not use the VxVector index: %', plan;
                      END IF;
                      EXECUTE format('SELECT array_agg(id ORDER BY distance,id),
                        array_agg(distance ORDER BY distance,id) FROM
                        (SELECT id, features {operator} %L::vector AS distance
                         FROM {schema}.source_{dims} ORDER BY distance LIMIT 10) nearest', query)
                        INTO found_ids, found_distances;
                      IF coalesce(cardinality(found_ids),0) <> 10 OR
                         (SELECT count(DISTINCT id) FROM unnest(found_ids) id) <> 10 THEN
                        RAISE EXCEPTION 'Index must return ten distinct neighbors';
                      END IF;
                      cutoff := exact_distances[10];
                      epsilon := 1e-6 + 1e-5 * abs(cutoff);
                      SELECT count(*), count(*) FILTER (WHERE id = ANY(found_ids))
                        INTO strict_total, strict_hits FROM unnest(exact_ids,exact_distances) n(id,distance)
                        WHERE distance < cutoff - epsilon;
                      SELECT count(*) INTO boundary_hits FROM unnest(found_distances) distance
                        WHERE abs(distance - cutoff) <= epsilon;
                      recall := (strict_hits + least(10-strict_total,boundary_hits)) / 10.0;
                      IF recall < {MIN_RECALL} THEN
                        RAISE EXCEPTION 'Recall below target for {tag}, query %: %, exact ids %, index ids %, exact distances %, index distances %',
                          q, recall, exact_ids, found_ids, exact_distances, found_distances;
                      END IF;
                      RAISE NOTICE 'M4_RESULT %', json_build_object('case','{tag}', 'query',q,
                        'rows',256, 'dims',{dims}, 'metric','{metric}', 'roots',{roots},
                        'seed',{seed}, 'sampling_factor',{factor}, 'batch_size',{batch},
                        'iterations',8, 'restarts',{restarts}, 'spherical',{spherical},
                        'probes','{probes}', 'epsilon',{args.epsilon}, 'recall_at_10',recall,
                        'cutoff',cutoff, 'tie_epsilon',epsilon,
                        'exact_ids',exact_ids, 'index_ids',found_ids);
                    END LOOP;
                  END $check$;
                  DROP INDEX {schema}.idx_{tag};
                """)
    statements.append(f"""
      DO $budget$
      DECLARE small jsonb; large jsonb; rejected boolean := false;
      BEGIN
        SELECT vxvector.create_vector_index('{schema}.source_128','features',ARRAY[4],
          sampling_factor=>8,batch_size=>13,skip_index_build=>true,
          centroid_table=>'budget_small') INTO small;
        INSERT INTO {schema}.source_128
          SELECT i, ARRAY(SELECT (CASE WHEN d = i % 4 + 1 THEN 3.0 ELSE 0.0 END
            + sin(i * 13.0 + d * 7.0) * 0.1)::real FROM generate_series(1,128) d)::vector
          FROM generate_series(257,8192) i;
        SELECT vxvector.create_vector_index('{schema}.source_128','features',ARRAY[4],
          sampling_factor=>8,batch_size=>13,skip_index_build=>true,
          centroid_table=>'budget_large') INTO large;
        IF (small->>'rows')::int <> 256 OR (large->>'rows')::int <> 8192 OR
          (small->>'samples')::int <> 32 OR (large->>'samples')::int <> 32 OR
          (small->>'buffer_bytes')::bigint <> (large->>'buffer_bytes')::bigint THEN
          RAISE EXCEPTION 'Bounded SQL sampling metadata changed with source size: %, %', small,large;
        END IF;
        BEGIN
          PERFORM vxvector.create_vector_index('{schema}.source_128','features',ARRAY[4],
            memory_budget_bytes=>1,skip_index_build=>true,centroid_table=>'budget_rejected');
        EXCEPTION WHEN OTHERS THEN
          IF SQLERRM NOT LIKE '%memory budget%' THEN RAISE; END IF;
          rejected := true;
        END;
        IF NOT rejected OR to_regclass('{schema}.budget_rejected') IS NOT NULL THEN
          RAISE EXCEPTION 'Budget rejection must leave no output';
        END IF;
        RAISE NOTICE 'M4_BUDGET %', json_build_object('small',small,'large',large,'rejected_budget',1);
      END $budget$;
      ROLLBACK;
    """)
    result = subprocess.run(['docker', 'exec', '-i', args.container, 'psql', '-X', '-At',
                             '-v', 'ON_ERROR_STOP=1', '-U', 'vxvector', '-d', 'vxvector'],
                            input='\n'.join(statements), text=True, capture_output=True)
    print(json.dumps(dict(client_platform=platform.platform())), flush=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    records = [json.loads(line.split('M4_RESULT ', 1)[1])
               for line in result.stderr.splitlines() if 'M4_RESULT ' in line]
    if len(records) != 120:
        raise AssertionError(f'Expected 120 recall queries, found {len(records)}')
    print(json.dumps(dict(result='passed', index_builds=24, centroid_only_builds=2, recall_queries=len(records),
                          minimum_recall=min(row['recall_at_10'] for row in records))), flush=True)


if __name__ == '__main__':
    main()
