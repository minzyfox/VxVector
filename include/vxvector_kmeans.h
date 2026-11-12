#ifndef VXVECTOR_KMEANS_H
#define VXVECTOR_KMEANS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vxvector_metric {
    VXVECTOR_L2 = 0,
    VXVECTOR_INNER_PRODUCT = 1,
    VXVECTOR_COSINE = 2
};

/*
 * Version 2 of the standalone CPU ABI. Train leaf centroids, then optional
 * root centroids, from row-major float32 vectors. All buffers are caller-owned
 * and must not overlap, except that the function uses separate regions of nodes
 * for roots and leaves.
 *
 * Training uses samples = min(rows, leaf_count * sampling_factor) rows. With
 * samples equal to rows, it uses every row in order. Otherwise SplitMix64 and
 * Floyd's algorithm choose them from seed, as vxvector_run_cpu_v1 does, so the
 * two APIs train on the same rows. Root training samples the leaves the same way.
 *
 * Required capacities in elements:
 *   points:       rows * dims
 *   nodes:        (1 + root_count + leaf_count) * dims
 *   parents:      1 + root_count + leaf_count
 *   assignments:  rows
 *   current:      leaf_count * dims
 *   sums:         leaf_count * dims
 *   counts:       leaf_count
 *   sample_rows:  samples
 *   sample_table: the smallest power of two that is at least 2 * samples
 *
 * Node 0 is a zero-vector root with parent -1. Roots, when requested, follow
 * node 0. Leaves follow the roots. A leaf assignment is a zero-based leaf index.
 * Parents contain node IDs. With root_count=0, every leaf has parent 0.
 *
 * The function returns 0 on success. It returns 1 for invalid arguments
 * or float32 centroid-sum overflow. Output buffers are unspecified on failure.
 * Input vectors must be finite. The caller must provide valid capacities.
 * The function cannot inspect those capacities.
 *
 * Version 1 also took batch_size, which did not change the result. Its sampler
 * visited rows with a seeded stride and differed from the data path's.
 */
int32_t vxvector_kmeans_v2(const float *points, int32_t rows, int32_t dims,
                           int32_t root_count, int32_t leaf_count,
                           int32_t sampling_factor, int32_t iterations,
                           int32_t restarts, int32_t metric, int32_t spherical,
                           int32_t seed, float *nodes, int32_t *parents,
                           int32_t *assignments, float *current, float *sums,
                           int32_t *counts, int64_t *sample_rows,
                           int64_t *sample_table);

#ifdef __cplusplus
}
#endif

#endif
