\set ON_ERROR_STOP on
BEGIN;

DO $$
BEGIN
    IF current_setting('server_version_num')::integer NOT BETWEEN 170000 AND 179999 THEN
        RAISE EXCEPTION 'PostgreSQL 17 is required';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector') OR
       NOT EXISTS (SELECT FROM pg_extension WHERE extname = 'vchord') THEN
        RAISE EXCEPTION 'pgvector and VectorChord must be installed';
    END IF;
    IF NOT ('vchord' = ANY(string_to_array(replace(current_setting('shared_preload_libraries'), ' ', ''), ','))) THEN
        RAISE EXCEPTION 'VectorChord must be preloaded';
    END IF;
END
$$;

SELECT version();
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vchord') ORDER BY extname;
SHOW shared_preload_libraries;

CREATE SCHEMA vxvector_image_smoke;
CREATE TABLE vxvector_image_smoke.items (id integer PRIMARY KEY, features vector(3) NOT NULL);
INSERT INTO vxvector_image_smoke.items
SELECT i, ARRAY[(i % 4) * 10 + (i % 17) * 0.001, (i % 7) * 0.001, (i % 11) * 0.001]::vector
FROM generate_series(1, 1024) AS i;

CREATE TABLE vxvector_image_smoke.flat_centroids (
    id integer NOT NULL UNIQUE, parent integer, vector vector(3) NOT NULL
);
INSERT INTO vxvector_image_smoke.flat_centroids VALUES
    (0, NULL, '[0,0,0]'),
    (1, 0, '[0.008,0.003,0.005]'),
    (2, 0, '[10.008,0.003,0.005]'),
    (3, 0, '[20.008,0.003,0.005]'),
    (4, 0, '[30.008,0.003,0.005]');

CREATE INDEX vxvector_image_flat_idx ON vxvector_image_smoke.items
USING vchordrq (features vector_l2_ops) WITH (options = $$
residual_quantization = false
[build.external]
table = 'vxvector_image_smoke.flat_centroids'
$$);
ANALYZE vxvector_image_smoke.items;
SET LOCAL enable_seqscan = off;
SET LOCAL vchordrq.probes = '4';

DO $$
DECLARE
    plan json;
    found integer;
BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON) SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> ''[10,0,0]'' LIMIT 5' INTO plan;
    IF position('vxvector_image_flat_idx' IN plan::text) = 0 THEN
        RAISE EXCEPTION 'The flat-tree query did not select the VectorChord index: %', plan;
    END IF;
    SELECT count(*) INTO found FROM (
        SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> '[10,0,0]' LIMIT 5
    ) AS nearest;
    IF found <> 5 THEN
        RAISE EXCEPTION 'The flat-tree query returned % rows instead of five', found;
    END IF;
END
$$;
EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF)
SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> '[10,0,0]' LIMIT 5;

DROP INDEX vxvector_image_smoke.vxvector_image_flat_idx;
CREATE TABLE vxvector_image_smoke.tree_centroids (
    id integer NOT NULL UNIQUE, parent integer, vector vector(3) NOT NULL
);
INSERT INTO vxvector_image_smoke.tree_centroids VALUES
    (0, NULL, '[0,0,0]'),
    (1, 0, '[5.008,0.003,0.005]'),
    (2, 0, '[25.008,0.003,0.005]'),
    (3, 1, '[0.008,0.003,0.005]'),
    (4, 1, '[10.008,0.003,0.005]'),
    (5, 2, '[20.008,0.003,0.005]'),
    (6, 2, '[30.008,0.003,0.005]');
CREATE INDEX vxvector_image_tree_idx ON vxvector_image_smoke.items
USING vchordrq (features vector_l2_ops) WITH (options = $$
residual_quantization = false
[build.external]
table = 'vxvector_image_smoke.tree_centroids'
$$);
SET LOCAL vchordrq.probes = '2,4';

DO $$
DECLARE
    plan json;
    found integer;
BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON) SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> ''[10,0,0]'' LIMIT 5' INTO plan;
    IF position('vxvector_image_tree_idx' IN plan::text) = 0 THEN
        RAISE EXCEPTION 'The two-level query did not select the VectorChord index: %', plan;
    END IF;
    SELECT count(*) INTO found FROM (
        SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> '[10,0,0]' LIMIT 5
    ) AS nearest;
    IF found <> 5 THEN
        RAISE EXCEPTION 'The two-level query returned % rows instead of five', found;
    END IF;
END
$$;
EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF)
SELECT id FROM vxvector_image_smoke.items ORDER BY features <-> '[10,0,0]' LIMIT 5;

ROLLBACK;
\echo 'PostgreSQL image checks passed; smoke tables and indexes were rolled back.'
