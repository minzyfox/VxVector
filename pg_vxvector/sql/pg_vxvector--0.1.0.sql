CREATE SCHEMA vxvector;

CREATE FUNCTION vxvector.create_vector_index(
    source_table regclass,
    column_name text,
    lists integer[],
    sampling_factor integer DEFAULT 256,
    batch_size integer DEFAULT 1024,
    kmeans_iterations integer DEFAULT 10,
    kmeans_nredo integer DEFAULT 1,
    distance_operator text DEFAULT 'l2',
    skip_index_build boolean DEFAULT false,
    spherical_centroids boolean DEFAULT false,
    residual_quantization boolean DEFAULT false,
    seed integer DEFAULT 42,
    memory_budget_bytes bigint DEFAULT 268435456,
    centroid_table text DEFAULT NULL,
    index_name text DEFAULT NULL,
    worker_timeout_ms integer DEFAULT 300000
) RETURNS jsonb
AS 'MODULE_PATHNAME', 'create_vector_index_wrapper'
LANGUAGE C VOLATILE PARALLEL UNSAFE;
