/* The C side of Step 3: GPU memory and the call into src/gpu_kmeans.vx.
 *
 * C allocates with vx_metal_alloc, so Metal uses each tensor in place. The
 * allocation also proves that a Metal device is present and enabled, because
 * the Vx runtime would otherwise run the kernels on the CPU without saying so.
 * vx_metal_free waits for the GPU work that uses the memory before it frees it.
 */
#include "vxvector_gpu.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

void *vx_metal_alloc(size_t bytes);
int vx_metal_free(void *memory);

/* A rank-1 tensor is 5 arguments and a rank-2 tensor is 7: the allocated
 * pointer, the aligned pointer, the offset, the sizes, and the strides. */
#define TENSOR1(p, n) (p), (p), 0, (int64_t)(n), 1
#define TENSOR2(p, r, c) (p), (p), 0, (int64_t)(r), (int64_t)(c), (int64_t)(c), 1

/* vxvector_gpu_l2_run_v1 in src/gpu_kmeans.vx. */
void vxvector_gpu_l2_run_v1(
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* centers */
    int32_t *, int32_t *, int64_t, int64_t, int64_t,                 /* labels */
    int32_t *, int32_t *, int64_t, int64_t, int64_t,                 /* counts */
    int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t, int64_t, /* overflow */
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* samples */
    int32_t, int32_t, int32_t, int32_t);

int32_t vxvector_gpu_l2_v1(const float *samples, int32_t n, int32_t dims,
                           int32_t k, int32_t iterations, float *centroids,
                           int32_t *labels, int32_t *counts)
{
    if (!samples || !centroids || !labels || !counts || k <= 0 || dims <= 0 ||
        n < k || iterations <= 0 || (int64_t)n * dims >= INT32_MAX ||
        (int64_t)k + iterations >= INT32_MAX)
        return VXVECTOR_GPU_INVALID;

    const size_t sample_bytes = (size_t)n * (size_t)dims * sizeof(float);
    const size_t center_bytes = (size_t)k * (size_t)dims * sizeof(float);
    float *g_samples = vx_metal_alloc(sample_bytes);
    float *g_centers = vx_metal_alloc(center_bytes);
    int32_t *g_labels = vx_metal_alloc((size_t)n * sizeof(int32_t));
    int32_t *g_counts = vx_metal_alloc((size_t)k * sizeof(int32_t));
    int32_t *g_overflow = vx_metal_alloc((size_t)k * (size_t)dims * sizeof(int32_t));
    int32_t status = VXVECTOR_GPU_UNAVAILABLE;

    if (g_samples && g_centers && g_labels && g_counts && g_overflow) {
        memcpy(g_samples, samples, sample_bytes);
        memset(g_overflow, 0, (size_t)k * (size_t)dims * sizeof(int32_t));
        vxvector_gpu_l2_run_v1(
            TENSOR2(g_centers, k, dims), TENSOR1(g_labels, n),
            TENSOR1(g_counts, k), TENSOR2(g_overflow, k, dims),
            TENSOR2(g_samples, n, dims), n, k, dims, iterations);
        /* The Vx function waits for the GPU before it returns. */
        status = VXVECTOR_GPU_OK;
        for (int64_t i = 0; i < (int64_t)k * dims; i++)
            if (g_overflow[i])
                status = VXVECTOR_GPU_OVERFLOW;
        if (status == VXVECTOR_GPU_OK) {
            memcpy(centroids, g_centers, center_bytes);
            memcpy(labels, g_labels, (size_t)n * sizeof(int32_t));
            memcpy(counts, g_counts, (size_t)k * sizeof(int32_t));
        }
    }
    if (g_samples)
        vx_metal_free(g_samples);
    if (g_centers)
        vx_metal_free(g_centers);
    if (g_labels)
        vx_metal_free(g_labels);
    if (g_counts)
        vx_metal_free(g_counts);
    if (g_overflow)
        vx_metal_free(g_overflow);
    return status;
}
