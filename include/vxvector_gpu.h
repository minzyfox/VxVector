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
    /* No usable GPU backend. The function never runs on the CPU. */
    VXVECTOR_GPU_UNAVAILABLE = 3
};

/* The direct Metal k-means API. */
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
 * Step 5: vxvector_gpu_l2_v1 for any metric of enum vxvector_metric in
 * vxvector_kmeans.h: L2 (0), inner product (1), or cosine (2). Inner product
 * and cosine add their dot products and squared norms in float32, where the
 * CPU uses float64. The cosine distance takes sqrt(left) * sqrt(right),
 * because left * right can overflow float32.
 */
int32_t vxvector_gpu_train_v1(const float *samples, int32_t n, int32_t dims,
                              int32_t k, int32_t iterations, int32_t metric,
                              int64_t launch_work, float *centroids, int32_t *labels,
                              int32_t *counts);

/*
 * Step 5: the time of the calls on this thread since the last reset.
 * Allocation includes release, which waits for any GPU work on the memory.
 * Training is the Vx call, which waits for the GPU before it returns.
 */
struct vxvector_gpu_timings {
    double allocation_seconds, copy_in_seconds, training_seconds, copy_out_seconds;
    uint64_t calls;
};
void vxvector_gpu_reset_timings_v1(void);
void vxvector_gpu_timings_v1(struct vxvector_gpu_timings *timings);

/* The backend and the Vx revision that built this library. */
const char *vxvector_gpu_version_v1(void);

/*
 * Whether this library can train, and the device name. Returns
 * VXVECTOR_GPU_OK or VXVECTOR_GPU_UNAVAILABLE. device receives the name, or
 * an empty string.
 */
int32_t vxvector_gpu_probe_v1(char *device, size_t size);

/*
 * Steps 4 and 5: one restart for the data path (vxvector_gpu_restart_fn in
 * vxvector_data_path.h). points holds the sample, and every row trains.
 * Restart r starts from the seeds of the CPU kernel's restart r. With
 * spherical, the CPU normalizes the centers in float64 after seeding and
 * after each update, as the CPU kernel does. Writes
 * leaf_count * dims centers and rows labels of the final assignment. Returns
 * 0, 1 for a request it rejects or a sum overflow (as the CPU kernel), or 2
 * when the GPU fails.
 */
int32_t vxvector_gpu_restart_v1(const float *points, int32_t rows, int32_t dims,
                                int32_t leaf_count, int32_t iterations, int32_t metric,
                                int32_t spherical, int32_t restart, float *centers,
                                int32_t *labels);

/*
 * Step 5: labels of the nearest of the k centers for each of the rows points,
 * as the final assignment of a restart computes them. For the data path, when
 * an earlier restart than the last is the best. Returns 0, 1 for a request it
 * rejects, or 2 when the GPU fails.
 */
int32_t vxvector_gpu_assign_v1(const float *points, int32_t rows, int32_t dims,
                               const float *centers, int32_t k, int32_t metric,
                               int32_t *labels);

#ifdef __cplusplus
}
#endif

#endif
