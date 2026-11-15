/* A stand-in for libvxvector_gpu, so tests of backend selection do not depend
 * on the machine's GPU. FAKE_GPU_PROBE=unavailable makes the probe fail.
 * FAKE_GPU_TRAIN=fail makes training fail after it starts, as a GPU fault
 * would. Otherwise each restart r writes centers of 7.0 + r, which the CPU
 * never would for the test inputs, so a test can tell which backend trained.
 * FAKE_GPU_BEST=r makes restart r the closest to the samples. */
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

static int failing(void) {
    const char *train = getenv("FAKE_GPU_TRAIN");
    return train && strcmp(train, "fail") == 0;
}

int32_t vxvector_gpu_restart_v1(const float *points, int32_t rows, int32_t dims,
                                int32_t leaf_count, int32_t iterations, int32_t metric,
                                int32_t spherical, int32_t restart, float *centers,
                                int32_t *labels) {
    (void)points, (void)iterations, (void)metric, (void)spherical;
    if (failing())
        return 2;
    /* Restart FAKE_GPU_BEST gets centers of 0, the nearest to small samples. */
    const char *best = getenv("FAKE_GPU_BEST");
    float value = best && atoi(best) == restart ? 0.0f : 7.0f + (float)restart;
    for (int64_t i = 0; i < (int64_t)leaf_count * dims; i++)
        centers[i] = value;
    for (int32_t r = 0; r < rows; r++)
        labels[r] = 0;
    return 0;
}

int32_t vxvector_gpu_assign_v1(const float *points, int32_t rows, int32_t dims,
                               const float *centers, int32_t k, int32_t metric,
                               int32_t *labels) {
    (void)points, (void)dims, (void)centers, (void)k, (void)metric;
    if (failing())
        return 2;
    for (int32_t r = 0; r < rows; r++)
        labels[r] = 0;
    return 0;
}

struct timings { double allocation, copy_in, training, copy_out; uint64_t calls; };
static struct timings fake_timings;
void vxvector_gpu_reset_timings_v1(void) { memset(&fake_timings, 0, sizeof fake_timings); }
void vxvector_gpu_timings_v1(struct timings *out) {
    fake_timings.training = 0.25;
    *out = fake_timings;
}
const char *vxvector_gpu_version_v1(void) { return "Fake GPU 1"; }
