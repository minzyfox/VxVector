/* A stand-in for libvxvector_gpu, so tests of backend selection do not depend
 * on the machine's GPU. FAKE_GPU_PROBE=unavailable makes the probe fail.
 * FAKE_GPU_TRAIN=fail makes training fail after it starts, as a GPU fault
 * would. Otherwise training writes leaves of 7.0, which the CPU never would
 * for the test inputs, so a test can tell which backend trained. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int32_t vxvector_gpu_probe_v1(char *device, size_t size) {
    const char *probe = getenv("FAKE_GPU_PROBE");
    if (probe && strcmp(probe, "unavailable") == 0)
        return 3;
    snprintf(device, size, "Fake GPU");
    return 0;
}

int32_t vxvector_gpu_kmeans_v1(const float *points, int32_t rows, int32_t dims,
                               int32_t root_count, int32_t leaf_count,
                               int32_t sampling_factor, int32_t iterations,
                               int32_t restarts, int32_t metric, int32_t spherical,
                               int32_t seed, float *nodes, int32_t *parents,
                               int32_t *assignments, float *current, float *sums,
                               int32_t *counts, int64_t *sample_rows,
                               int64_t *sample_table) {
    (void)points, (void)root_count, (void)sampling_factor, (void)iterations, (void)restarts,
        (void)metric, (void)spherical, (void)seed, (void)current, (void)sums, (void)counts,
        (void)sample_rows, (void)sample_table;
    const char *train = getenv("FAKE_GPU_TRAIN");
    if (train && strcmp(train, "fail") == 0)
        return 2;
    for (int32_t d = 0; d < dims; d++)
        nodes[d] = 0.0f;
    for (int64_t i = dims; i < (int64_t)(1 + leaf_count) * dims; i++)
        nodes[i] = 7.0f;
    parents[0] = -1;
    for (int32_t i = 1; i <= leaf_count; i++)
        parents[i] = 0;
    for (int32_t r = 0; r < rows; r++)
        assignments[r] = 0;
    return 0;
}
