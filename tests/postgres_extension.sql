\set ON_ERROR_STOP on
CREATE EXTENSION IF NOT EXISTS pg_vextor;
BEGIN;
CREATE SCHEMA vextor_m3_smoke;
CREATE TABLE vextor_m3_smoke.source (id integer PRIMARY KEY, features vector(2));
INSERT INTO vextor_m3_smoke.source
SELECT i, ARRAY[(i % 4) * 10, (i % 3)]::vector FROM generate_series(1, 64) AS i;
CREATE TEMP TABLE vextor_builds (result jsonb);
INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source', 'features', ARRAY[4], batch_size => 1,
    skip_index_build => true, centroid_table => 'sample_centroids');
INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source', 'features', ARRAY[4], batch_size => 100,
    skip_index_build => true, centroid_table => 'repeat_centroids');

DO $$
BEGIN
    IF EXISTS ((SELECT * FROM vextor_m3_smoke.sample_centroids EXCEPT SELECT * FROM vextor_m3_smoke.repeat_centroids)
               UNION ALL
               (SELECT * FROM vextor_m3_smoke.repeat_centroids EXCEPT SELECT * FROM vextor_m3_smoke.sample_centroids)) THEN
        RAISE EXCEPTION 'Changing batch size changed the centroids';
    END IF;
    IF (SELECT count(*) FROM vextor_m3_smoke.sample_centroids) <> 5 OR
       (SELECT count(*) FROM vextor_m3_smoke.sample_centroids WHERE id=0 AND parent IS NULL AND vector='[0,0]') <> 1 THEN
        RAISE EXCEPTION 'Flat centroid table has the wrong root or node count';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE schemaname='vextor_m3_smoke' AND indexdef LIKE '%USING vchordrq%') THEN
        RAISE EXCEPTION 'skip_index_build created an index';
    END IF;
    IF EXISTS (SELECT FROM vextor_builds WHERE result->>'backend' <> 'cpu' OR
               (result->>'samples')::integer <> 64 OR result->>'index' IS NOT NULL OR
               (result->>'buffer_bytes')::bigint > 268435456) THEN
        RAISE EXCEPTION 'CPU result metadata is invalid';
    END IF;
END $$;

INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source', 'features', ARRAY[4],
    centroid_table => 'flat_centroids', index_name => 'flat_idx');
SET LOCAL enable_seqscan = off;
SET LOCAL vchordrq.probes = '4';
DO $$
DECLARE plan json;
BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON) SELECT id FROM vextor_m3_smoke.source ORDER BY features <-> ''[10,0]'' LIMIT 5' INTO plan;
    IF position('flat_idx' IN plan::text) = 0 THEN
        RAISE EXCEPTION 'Flat Vx index was not selected: %', plan;
    END IF;
END $$;
EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF)
SELECT id FROM vextor_m3_smoke.source ORDER BY features <-> '[10,0]' LIMIT 5;
DROP INDEX vextor_m3_smoke.flat_idx;

INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source', 'features', ARRAY[2,4],
    centroid_table => 'tree_centroids', index_name => 'tree_idx');
SET LOCAL vchordrq.probes = '2,4';
DO $$
DECLARE plan json;
BEGIN
    IF (SELECT count(*) FROM vextor_m3_smoke.tree_centroids) <> 7 OR
       EXISTS (SELECT FROM vextor_m3_smoke.tree_centroids WHERE id > 2 AND parent NOT IN (1,2)) THEN
        RAISE EXCEPTION 'The two-level Vx tree is malformed';
    END IF;
    EXECUTE 'EXPLAIN (FORMAT JSON) SELECT id FROM vextor_m3_smoke.source ORDER BY features <-> ''[10,0]'' LIMIT 5' INTO plan;
    IF position('tree_idx' IN plan::text) = 0 THEN
        RAISE EXCEPTION 'Two-level Vx index was not selected: %', plan;
    END IF;
END $$;
EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF)
SELECT id FROM vextor_m3_smoke.source ORDER BY features <-> '[10,0]' LIMIT 5;

INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source','features',ARRAY[4],distance_operator=>'ip',
    spherical_centroids=>true,skip_index_build=>true,centroid_table=>'ip_centroids');
INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source','features',ARRAY[4],distance_operator=>'cos',
    spherical_centroids=>true,skip_index_build=>true,centroid_table=>'cos_centroids');

CREATE FUNCTION vextor_m3_smoke.expect_error(statement text, pattern text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE failed boolean := false;
BEGIN
    BEGIN
        EXECUTE statement;
    EXCEPTION WHEN OTHERS THEN
        failed := true;
        IF SQLERRM !~ pattern THEN
            RAISE EXCEPTION 'Wrong error: %, expected %', SQLERRM, pattern;
        END IF;
    END;
    IF NOT failed THEN RAISE EXCEPTION 'Statement unexpectedly succeeded: %', statement; END IF;
END $$;

SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4],distance_operator=>'bad')$$,
    'Invalid CPU data path');
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','id',ARRAY[4])$$,
    'pgvector vector type');
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4],memory_budget_bytes=>1)$$,
    'memory budget');
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[NULL]::integer[])$$,
    'lists cannot contain NULL');

CREATE TABLE vextor_m3_smoke.mixed (features vector);
INSERT INTO vextor_m3_smoke.mixed VALUES ('[1,2]'),('[1,2,3]');
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.mixed','features',ARRAY[1])$$,
    'mixed vector dimensions');
INSERT INTO vextor_m3_smoke.source VALUES (100,NULL);
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[1])$$,
    'NULL vector');
DELETE FROM vextor_m3_smoke.source WHERE id=100;

-- A duplicate index name fails after centroid writes. The subtransaction must undo them.
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4],centroid_table=>'rolled_back_centroids',index_name=>'tree_idx')$$,
    'already exists');
DO $$
BEGIN
    IF to_regclass('vextor_m3_smoke.rolled_back_centroids') IS NOT NULL THEN
        RAISE EXCEPTION 'Failed index build left its centroid table';
    END IF;
END $$;

CREATE ROLE vextor_m3_reader NOLOGIN;
GRANT USAGE ON SCHEMA vextor, vextor_m3_smoke TO vextor_m3_reader;
CREATE TABLE vextor_m3_smoke.private_vectors (features vector(2));
INSERT INTO vextor_m3_smoke.private_vectors VALUES ('[1,2]');
CREATE TABLE vextor_m3_smoke.protected_vectors (features vector(2));
INSERT INTO vextor_m3_smoke.protected_vectors VALUES ('[1,2]');
ALTER TABLE vextor_m3_smoke.protected_vectors ENABLE ROW LEVEL SECURITY;
CREATE POLICY visible_vectors ON vextor_m3_smoke.protected_vectors USING (true);
GRANT SELECT ON vextor_m3_smoke.protected_vectors TO vextor_m3_reader;
SET LOCAL ROLE vextor_m3_reader;
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.private_vectors','features',ARRAY[1])$$,
    'permission denied');
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.protected_vectors','features',ARRAY[1])$$,
    'row-level security');
SET LOCAL ROLE NONE;

SET LOCAL vextor.worker_path = '/bin/false';
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4])$$,
    'worker closed|worker exited');
SET LOCAL vextor.worker_path = '/tmp/vextor-crash-worker';
SELECT vextor_m3_smoke.expect_error(
    $$SELECT vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4])$$,
    'worker closed|worker exited');
SET LOCAL vextor.worker_path = '/tmp/vextor-sleep-worker';
DO $$
DECLARE failed boolean := false;
BEGIN
    BEGIN
        PERFORM vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4],worker_timeout_ms=>100);
    EXCEPTION WHEN query_canceled THEN
        failed := true;
        IF SQLERRM NOT LIKE '%worker timed out%' THEN RAISE; END IF;
    END;
    IF NOT failed THEN RAISE EXCEPTION 'Worker timeout did not fire'; END IF;
END $$;
SET LOCAL statement_timeout = '200ms';
DO $$
DECLARE failed boolean := false;
BEGIN
    BEGIN
        PERFORM vextor.create_vector_index('vextor_m3_smoke.source','features',ARRAY[4]);
    EXCEPTION WHEN query_canceled THEN
        failed := true;
        IF SQLERRM NOT LIKE '%statement timeout%' THEN RAISE; END IF;
    END;
    IF NOT failed THEN RAISE EXCEPTION 'Statement cancellation did not fire'; END IF;
END $$;
SET LOCAL statement_timeout = '0';
SET LOCAL vextor.worker_path = DEFAULT;

-- pgvector types and VectorChord operator classes need not share a schema.
DROP INDEX vextor_m3_smoke.tree_idx;
CREATE SCHEMA vextor_vector_types;
ALTER EXTENSION vector SET SCHEMA vextor_vector_types;
INSERT INTO vextor_builds SELECT vextor.create_vector_index(
    'vextor_m3_smoke.source','features',ARRAY[4],
    centroid_table=>'relocated_centroids',index_name=>'relocated_idx');
SET LOCAL vchordrq.probes = '4';
EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF)
SELECT id FROM vextor_m3_smoke.source
ORDER BY features OPERATOR(vextor_vector_types.<->) '[10,0]' LIMIT 5;

SELECT result FROM vextor_builds;
ROLLBACK;
SELECT 'CPU extension, worker isolation, cancellation, and rollback checks passed' AS result;
