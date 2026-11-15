/* The C side of Step 3: GPU memory and the call into src/gpu_kmeans.vx.
 *
 * C allocates with vx_metal_alloc, so Metal uses each tensor in place. The
 * allocation also proves that a Metal device is present and enabled, because
 * the Vx runtime would otherwise run the kernels on the CPU without saying so.
 * vx_metal_free waits for the GPU work that uses the memory before it frees it.
 */
#include "vxvector_gpu.h"

#define _DARWIN_C_SOURCE /* CLOCK_MONOTONIC_RAW */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#ifndef VXVECTOR_VX_REVISION
#define VXVECTOR_VX_REVISION "unknown"
#endif

/* The GPU time of the calls since the last reset, on this thread. */
static _Thread_local struct vxvector_gpu_timings timings;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

void vxvector_gpu_reset_timings_v1(void) { memset(&timings, 0, sizeof timings); }

void vxvector_gpu_timings_v1(struct vxvector_gpu_timings *out)
{
    if (out)
        *out = timings;
}

const char *vxvector_gpu_version_v1(void) { return "Metal, Vx " VXVECTOR_VX_REVISION; }

void *vx_metal_alloc(size_t bytes);
int vx_metal_free(void *memory);

/* A rank-1 tensor is 5 arguments and a rank-2 tensor is 7: the allocated
 * pointer, the aligned pointer, the offset, the sizes, and the strides. */
#define TENSOR1(p, n) (p), (p), 0, (int64_t)(n), 1
#define TENSOR2(p, r, c) (p), (p), 0, (int64_t)(r), (int64_t)(c), (int64_t)(c), 1

/* vxvector_gpu_assign_all_v1 in src/gpu_kmeans.vx. */
void vxvector_gpu_assign_all_v1(
    int32_t *, int32_t *, int64_t, int64_t, int64_t,                 /* labels */
    float *, float *, int64_t, int64_t, int64_t,                     /* best */
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* samples */
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* centers */
    int32_t, int32_t, int32_t, int32_t, int64_t);

/* vxvector_gpu_l2_run_v2 in src/gpu_kmeans.vx. */
void vxvector_gpu_l2_run_v2(
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* centers */
    int32_t *, int32_t *, int64_t, int64_t, int64_t,                 /* labels */
    float *, float *, int64_t, int64_t, int64_t,                     /* best */
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* sums */
    int32_t *, int32_t *, int64_t, int64_t, int64_t,                 /* counts */
    int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t, int64_t, /* overflow */
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,   /* samples */
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t,
    void (*)(float *, int32_t, int32_t));

/* At most about 2 ms of GPU time per launch on an M2 Pro. */
#define DEFAULT_LAUNCH_WORK ((int64_t)1 << 26)

static int32_t vxvector_gpu_train_restart(const float *, int32_t, int32_t, int32_t,
                                          int32_t, int32_t, int32_t, int32_t, int64_t,
                                          float *, int32_t *, int32_t *);

/* Normalizes each of count rows of dims values as normalize in src/kmeans.vx
 * does: the squared norm added in float64 in order, its float64 square root,
 * and each value divided in float64 and rounded to float32. A zero row stays.
 * Contraction would fuse w * w + squared into one rounding, which the CPU
 * does not, so it is off. src/gpu_kmeans.vx calls this between kernels. */
#pragma STDC FP_CONTRACT OFF
void vxvector_gpu_normalize_rows(float *rows, int32_t count, int32_t dims)
{
    for (int64_t r = 0; r < count; r++) {
        float *v = rows + r * dims;
        double squared = 0.0;
        for (int32_t d = 0; d < dims; d++) {
            double wide = (double)v[d];
            squared += wide * wide;
        }
        if (squared > 0.0) {
            double norm = sqrt(squared);
            for (int32_t d = 0; d < dims; d++)
                v[d] = (float)((double)v[d] / norm);
        }
    }
}

int32_t vxvector_gpu_l2_v1(const float *samples, int32_t n, int32_t dims,
                           int32_t k, int32_t iterations, int64_t launch_work,
                           float *centroids, int32_t *labels, int32_t *counts)
{
    return vxvector_gpu_train_v1(samples, n, dims, k, iterations, 0, launch_work,
                                 centroids, labels, counts);
}

int32_t vxvector_gpu_train_v1(const float *samples, int32_t n, int32_t dims,
                              int32_t k, int32_t iterations, int32_t metric,
                              int64_t launch_work, float *centroids, int32_t *labels,
                              int32_t *counts)
{
    return vxvector_gpu_train_restart(samples, n, dims, k, iterations, metric, 0, 0, launch_work,
                                      centroids, labels, counts);
}

static int32_t vxvector_gpu_train_restart(const float *samples, int32_t n, int32_t dims,
                                          int32_t k, int32_t iterations, int32_t metric,
                                          int32_t spherical, int32_t restart, int64_t launch_work,
                                          float *centroids, int32_t *labels, int32_t *counts)
{
    if (!samples || !centroids || !labels || !counts || k <= 0 || dims <= 0 ||
        n < k || iterations <= 0 || launch_work < 0 || metric < 0 || metric > 2 || restart < 0 ||
        (int64_t)n * dims >= INT32_MAX || (int64_t)k + iterations >= INT32_MAX)
        return VXVECTOR_GPU_INVALID;
    if (launch_work == 0)
        launch_work = DEFAULT_LAUNCH_WORK;

    const size_t center_values = (size_t)k * (size_t)dims;
    const size_t sample_bytes = (size_t)n * (size_t)dims * sizeof(float);
    double start = now();
    float *g_samples = vx_metal_alloc(sample_bytes);
    float *g_centers = vx_metal_alloc(center_values * sizeof(float));
    float *g_sums = vx_metal_alloc(center_values * sizeof(float));
    float *g_best = vx_metal_alloc((size_t)n * sizeof(float));
    int32_t *g_labels = vx_metal_alloc((size_t)n * sizeof(int32_t));
    int32_t *g_counts = vx_metal_alloc((size_t)k * sizeof(int32_t));
    int32_t *g_overflow = vx_metal_alloc(center_values * sizeof(int32_t));
    int32_t status = VXVECTOR_GPU_UNAVAILABLE;
    timings.allocation_seconds += now() - start;

    if (g_samples && g_centers && g_sums && g_best && g_labels && g_counts && g_overflow) {
        start = now();
        memcpy(g_samples, samples, sample_bytes);
        memset(g_overflow, 0, center_values * sizeof(int32_t));
        timings.copy_in_seconds += now() - start;
        start = now();
        vxvector_gpu_l2_run_v2(
            TENSOR2(g_centers, k, dims), TENSOR1(g_labels, n), TENSOR1(g_best, n),
            TENSOR2(g_sums, k, dims), TENSOR1(g_counts, k),
            TENSOR2(g_overflow, k, dims), TENSOR2(g_samples, n, dims),
            n, k, dims, iterations, metric, spherical != 0,
            (int32_t)(((int64_t)restart * 104729) % n),
            launch_work, vxvector_gpu_normalize_rows);
        /* The Vx function waits for the GPU before it returns. */
        timings.training_seconds += now() - start;
        start = now();
        status = VXVECTOR_GPU_OK;
        for (size_t i = 0; i < center_values; i++)
            if (g_overflow[i])
                status = VXVECTOR_GPU_OVERFLOW;
        if (status == VXVECTOR_GPU_OK) {
            memcpy(centroids, g_centers, center_values * sizeof(float));
            memcpy(labels, g_labels, (size_t)n * sizeof(int32_t));
            memcpy(counts, g_counts, (size_t)k * sizeof(int32_t));
        }
        timings.copy_out_seconds += now() - start;
    }
    start = now();
    void *buffers[] = {g_samples, g_centers, g_sums, g_best, g_labels, g_counts, g_overflow};
    for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++)
        if (buffers[i])
            vx_metal_free(buffers[i]);
    timings.allocation_seconds += now() - start;
    timings.calls += 1;
    return status;
}

int vxvector_gpu_device_name(char *name, size_t size); /* src/gpu_device.m */

int32_t vxvector_gpu_probe_v1(char *device, size_t size)
{
    if (device && size)
        device[0] = '\0';
    /* vx_metal_alloc fails without a device and when VX_METAL=0, the cases in
     * which the runtime would run the kernels on the CPU. */
    void *page = vx_metal_alloc(1);
    if (!page)
        return VXVECTOR_GPU_UNAVAILABLE;
    vx_metal_free(page);
    if (device && size)
        vxvector_gpu_device_name(device, size);
    return VXVECTOR_GPU_OK;
}

int32_t vxvector_gpu_restart_v1(const float *points, int32_t rows, int32_t dims,
                                int32_t leaf_count, int32_t iterations, int32_t metric,
                                int32_t spherical, int32_t restart, float *centers,
                                int32_t *labels)
{
    if (!points || !centers || !labels || leaf_count <= 0 || dims <= 0)
        return 1;
    int32_t *counts = malloc((size_t)leaf_count * sizeof(int32_t));
    if (!counts)
        return 2;
    int32_t status = vxvector_gpu_train_restart(points, rows, dims, leaf_count, iterations,
                                                metric, spherical, restart, 0, centers,
                                                labels, counts);
    free(counts);
    if (status == VXVECTOR_GPU_UNAVAILABLE)
        return 2;
    return status == VXVECTOR_GPU_OK ? 0 : 1;
}

int32_t vxvector_gpu_assign_v1(const float *points, int32_t rows, int32_t dims,
                               const float *centers, int32_t k, int32_t metric,
                               int32_t *labels)
{
    if (!points || !centers || !labels || k <= 0 || dims <= 0 || rows < k || metric < 0 ||
        metric > 2 || (int64_t)rows * dims >= INT32_MAX)
        return 1;
    const size_t sample_bytes = (size_t)rows * (size_t)dims * sizeof(float);
    const size_t center_bytes = (size_t)k * (size_t)dims * sizeof(float);
    float *g_samples = vx_metal_alloc(sample_bytes);
    float *g_centers = vx_metal_alloc(center_bytes);
    float *g_best = vx_metal_alloc((size_t)rows * sizeof(float));
    int32_t *g_labels = vx_metal_alloc((size_t)rows * sizeof(int32_t));
    int32_t status = 2;
    if (g_samples && g_centers && g_best && g_labels) {
        double start = now();
        memcpy(g_samples, points, sample_bytes);
        memcpy(g_centers, centers, center_bytes);
        timings.copy_in_seconds += now() - start;
        start = now();
        vxvector_gpu_assign_all_v1(TENSOR1(g_labels, rows), TENSOR1(g_best, rows),
                                   TENSOR2(g_samples, rows, dims), TENSOR2(g_centers, k, dims),
                                   rows, k, dims, metric, DEFAULT_LAUNCH_WORK);
        timings.training_seconds += now() - start;
        start = now();
        memcpy(labels, g_labels, (size_t)rows * sizeof(int32_t));
        timings.copy_out_seconds += now() - start;
        status = 0;
    }
    timings.calls += 1;
    void *buffers[] = {g_samples, g_centers, g_best, g_labels};
    for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++)
        if (buffers[i])
            vx_metal_free(buffers[i]);
    return status;
}
