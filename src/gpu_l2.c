/* The C side of the GPU library: GPU memory and the call into
 * src/gpu_kmeans.vx, for every GPU.
 *
 * C allocates through src/gpu_backend.h, so the Vx runtime uses each tensor in
 * place, and copies the inputs in and the results out. The backend file of the
 * GPU (src/gpu_backend_metal.m or src/gpu_backend_cuda.c) does the rest.
 */
/* CLOCK_MONOTONIC_RAW, which -std=c11 hides. Before any include. */
#define _DARWIN_C_SOURCE
#define _GNU_SOURCE
#include "vxvector_gpu.h"
#include "gpu_backend.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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

const char *vxvector_gpu_version_v1(void)
{
    static char version[64];
    if (!version[0])
        snprintf(version, sizeof version, "%s, Vx %s", vxvector_gpu_backend_api,
                 VXVECTOR_VX_REVISION);
    return version;
}

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

static int32_t vxvector_gpu_train_restart(const float *, int32_t, int32_t, int32_t,
                                          int32_t, int32_t, int32_t, int32_t, int64_t,
                                          float *, int32_t *, int32_t *);

/* Normalizes each of count rows of dims values as normalize in src/kmeans.vx
 * does: the squared norm added in float64 in order, its float64 square root,
 * and each value divided in float64 and rounded to float32. A zero row stays.
 * Contraction would fuse w * w + squared into one rounding, which the CPU
 * does not, so it is off. src/gpu_kmeans.vx calls this between kernels. */
#ifdef __clang__
#pragma STDC FP_CONTRACT OFF /* GCC has no such pragma, and has -ffp-contract=off. */
#endif
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

/* The normalize callback of src/gpu_kmeans.vx. The Vx loop does not wait before
 * it calls through a function pointer, so this waits for the GPU, then
 * normalizes the k centers on the host and copies them back. scratch holds
 * k * dims floats, and a failure sets failed, because the callback returns
 * nothing. Both belong to the call on this thread. */
static _Thread_local float *normalize_scratch;
static _Thread_local int normalize_failed;

static void normalize_centers(float *centers, int32_t k, int32_t dims)
{
    size_t bytes = (size_t)k * (size_t)dims * sizeof(float);
    if (normalize_failed || vxvector_gpu_backend_wait() != 0 ||
        vxvector_gpu_backend_copy_out(normalize_scratch, centers, bytes) != 0) {
        normalize_failed = 1;
        return;
    }
    vxvector_gpu_normalize_rows(normalize_scratch, k, dims);
    if (vxvector_gpu_backend_copy_in(centers, normalize_scratch, bytes) != 0)
        normalize_failed = 1;
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
        launch_work = vxvector_gpu_backend_launch_work;

    const size_t center_values = (size_t)k * (size_t)dims;
    const size_t sample_bytes = (size_t)n * (size_t)dims * sizeof(float);
    /* Host memory: zeros for the overflow flags, then the flags read back,
     * and the centers that normalize_centers works on. */
    int32_t *overflow = calloc(center_values, sizeof(int32_t));
    float *scratch = malloc(center_values * sizeof(float));
    double start = now();
    float *g_samples = vxvector_gpu_backend_alloc(sample_bytes);
    float *g_centers = vxvector_gpu_backend_alloc(center_values * sizeof(float));
    float *g_sums = vxvector_gpu_backend_alloc(center_values * sizeof(float));
    float *g_best = vxvector_gpu_backend_alloc((size_t)n * sizeof(float));
    int32_t *g_labels = vxvector_gpu_backend_alloc((size_t)n * sizeof(int32_t));
    int32_t *g_counts = vxvector_gpu_backend_alloc((size_t)k * sizeof(int32_t));
    int32_t *g_overflow = vxvector_gpu_backend_alloc(center_values * sizeof(int32_t));
    int32_t status = VXVECTOR_GPU_UNAVAILABLE;
    timings.allocation_seconds += now() - start;

    if (overflow && scratch && g_samples && g_centers && g_sums && g_best && g_labels &&
        g_counts && g_overflow) {
        start = now();
        int failed = vxvector_gpu_backend_copy_in(g_samples, samples, sample_bytes) != 0 ||
                     vxvector_gpu_backend_copy_in(g_overflow, overflow,
                                                  center_values * sizeof(int32_t)) != 0;
        timings.copy_in_seconds += now() - start;
        if (!failed) {
            start = now();
            normalize_scratch = scratch;
            normalize_failed = 0;
            vxvector_gpu_l2_run_v2(
                TENSOR2(g_centers, k, dims), TENSOR1(g_labels, n), TENSOR1(g_best, n),
                TENSOR2(g_sums, k, dims), TENSOR1(g_counts, k),
                TENSOR2(g_overflow, k, dims), TENSOR2(g_samples, n, dims),
                n, k, dims, iterations, metric, spherical != 0,
                (int32_t)(((int64_t)restart * 104729) % n),
                launch_work, normalize_centers);
            failed = normalize_failed || vxvector_gpu_backend_wait() != 0;
            normalize_scratch = NULL;
            timings.training_seconds += now() - start;
        }
        if (!failed) {
            start = now();
            failed = vxvector_gpu_backend_copy_out(overflow, g_overflow,
                                                   center_values * sizeof(int32_t)) != 0;
            status = VXVECTOR_GPU_OK;
            for (size_t i = 0; !failed && i < center_values; i++)
                if (overflow[i])
                    status = VXVECTOR_GPU_OVERFLOW;
            if (!failed && status == VXVECTOR_GPU_OK)
                failed = vxvector_gpu_backend_copy_out(centroids, g_centers,
                                                       center_values * sizeof(float)) != 0 ||
                         vxvector_gpu_backend_copy_out(labels, g_labels,
                                                       (size_t)n * sizeof(int32_t)) != 0 ||
                         vxvector_gpu_backend_copy_out(counts, g_counts,
                                                       (size_t)k * sizeof(int32_t)) != 0;
            timings.copy_out_seconds += now() - start;
        }
        if (failed)
            status = VXVECTOR_GPU_UNAVAILABLE;
    }
    start = now();
    void *buffers[] = {g_samples, g_centers, g_sums, g_best, g_labels, g_counts, g_overflow};
    for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++)
        vxvector_gpu_backend_free(buffers[i]);
    timings.allocation_seconds += now() - start;
    timings.calls += 1;
    free(overflow);
    free(scratch);
    return status;
}

int32_t vxvector_gpu_probe_v1(char *device, size_t size)
{
    if (device && size)
        device[0] = '\0';
    /* The allocation fails when there is no usable GPU. On Metal that includes
     * VX_METAL=0, in which the runtime would run the kernels on the CPU. */
    void *page = vxvector_gpu_backend_alloc(1);
    if (!page)
        return VXVECTOR_GPU_UNAVAILABLE;
    vxvector_gpu_backend_free(page);
    if (device && size)
        vxvector_gpu_backend_device_name(device, size);
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
    float *g_samples = vxvector_gpu_backend_alloc(sample_bytes);
    float *g_centers = vxvector_gpu_backend_alloc(center_bytes);
    float *g_best = vxvector_gpu_backend_alloc((size_t)rows * sizeof(float));
    int32_t *g_labels = vxvector_gpu_backend_alloc((size_t)rows * sizeof(int32_t));
    int32_t status = 2;
    if (g_samples && g_centers && g_best && g_labels) {
        double start = now();
        int failed = vxvector_gpu_backend_copy_in(g_samples, points, sample_bytes) != 0 ||
                     vxvector_gpu_backend_copy_in(g_centers, centers, center_bytes) != 0;
        timings.copy_in_seconds += now() - start;
        if (!failed) {
            start = now();
            vxvector_gpu_assign_all_v1(TENSOR1(g_labels, rows), TENSOR1(g_best, rows),
                                       TENSOR2(g_samples, rows, dims),
                                       TENSOR2(g_centers, k, dims), rows, k, dims, metric,
                                       vxvector_gpu_backend_launch_work);
            failed = vxvector_gpu_backend_wait() != 0;
            timings.training_seconds += now() - start;
        }
        if (!failed) {
            start = now();
            failed = vxvector_gpu_backend_copy_out(labels, g_labels,
                                                   (size_t)rows * sizeof(int32_t)) != 0;
            timings.copy_out_seconds += now() - start;
        }
        if (!failed)
            status = 0;
    }
    timings.calls += 1;
    void *buffers[] = {g_samples, g_centers, g_best, g_labels};
    for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++)
        vxvector_gpu_backend_free(buffers[i]);
    return status;
}
