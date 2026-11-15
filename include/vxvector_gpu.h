#ifndef VXVECTOR_GPU_H
#define VXVECTOR_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vxvector_gpu_status {
    VXVECTOR_GPU_OK = 0,
    /* A float32 centroid sum was not finite, as status 1 of the CPU kernel. */
    VXVECTOR_GPU_OVERFLOW = 1,
    VXVECTOR_GPU_INVALID = 2,
    /* No Metal device, or VX_METAL=0. The function never runs on the CPU. */
    VXVECTOR_GPU_UNAVAILABLE = 3
};

/*
 * Step 3: one restart of L2 Lloyd's algorithm on the Apple GPU, macOS only.
 *
 * samples holds n row-major float32 vectors of dims elements, already chosen
 * and in sample order. The policy is in docs/step3_metal_l2.md. It matches one
 * restart of the CPU kernel with restarts=1, the L2 metric, no spherical
 * normalization and no roots, given the same samples, except that the GPU adds
 * squared distances in float32 instead of float64.
 *
 * Writes k * dims centroids, n labels of the final assignment, and k cluster
 * sizes of the last update. All buffers are caller-owned. The samples are
 * copied once into GPU memory, which stays resident for every iteration.
 *
 * launch_work bounds the multiply-adds of each GPU launch, 0 for the default
 * of 2^26. macOS can stop a long launch (docs/step3_metal_l2.md), so the work
 * is split into short launches. The split does not change the result.
 *
 * Requires 0 < k <= n, dims > 0, iterations > 0, launch_work >= 0,
 * n * dims < 2^31 and k + iterations < 2^31.
 */
int32_t vxvector_gpu_l2_v1(const float *samples, int32_t n, int32_t dims,
                           int32_t k, int32_t iterations, int64_t launch_work,
                           float *centroids, int32_t *labels, int32_t *counts);

/*
 * Step 4: whether this library can train, and the device name. Returns
 * VXVECTOR_GPU_OK, or VXVECTOR_GPU_UNAVAILABLE when there is no Metal device
 * or VX_METAL=0. device receives the name, or an empty string.
 */
int32_t vxvector_gpu_probe_v1(char *device, size_t size);

/*
 * Step 4: a trainer with the arguments of vxvector_kmeans_v2, for the data
 * path (vxvector_train_fn in vxvector_data_path.h). points holds the sample,
 * and every row trains. It supports L2 with one restart, no spherical
 * normalization, and no roots. It writes the same layout as the CPU kernel:
 * node 0 is a zero vector with parent -1, then the leaves with parent 0.
 * current, sums, counts, sample_rows and sample_table are unused. Returns 0, 1
 * for a request it rejects or a sum overflow (as the CPU kernel), or 2 when
 * the GPU fails.
 */
int32_t vxvector_gpu_kmeans_v1(const float *points, int32_t rows, int32_t dims,
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
