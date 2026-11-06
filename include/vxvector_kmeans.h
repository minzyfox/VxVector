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
 * Version 1 of the standalone CPU ABI. Train leaf centroids, then optional
 * root centroids, from row-major float32
 * vectors. All buffers are caller-owned and must not overlap, except that the
 * function uses separate regions of nodes for roots and leaves.
 *
 * Required capacities in elements:
 *   points:      rows * dims
 *   nodes:       (1 + root_count + leaf_count) * dims
 *   parents:     1 + root_count + leaf_count
 *   assignments: rows
 *   current:     leaf_count * dims
 *   sums:        leaf_count * dims
 *   counts:      leaf_count
 *
 * Node 0 is a zero-vector root with parent -1. Roots, when requested, follow
 * node 0. Leaves follow the roots. A leaf assignment is a zero-based leaf index.
 * Parents contain node IDs. With root_count=0, every leaf has parent 0.
 *
 * The function returns 0 on success. It returns 1 for invalid arguments
 * or float32 centroid-sum overflow. Output buffers are unspecified on failure.
 * Input vectors must be finite. The caller must provide valid capacities.
 * The function cannot inspect those capacities.
 */
int32_t vxvector_kmeans_v1(const float *points, int32_t rows, int32_t dims,
                     int32_t root_count, int32_t leaf_count,
                     int32_t sampling_factor, int32_t batch_size,
                     int32_t iterations, int32_t restarts, int32_t metric,
                     int32_t spherical, int32_t seed,
                     float *nodes, int32_t *parents, int32_t *assignments,
                     float *current, float *sums, int32_t *counts);

#ifdef __cplusplus
}
#endif

#endif
