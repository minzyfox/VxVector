/* Check actual allocations and failure cleanup independently of the Vx runtime. */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vextor_data_path.h"
#include "vextor_kmeans.h"

union allocation_header {
    max_align_t alignment;
    size_t bytes;
};

static size_t live_bytes, peak_bytes, allocation_calls, fail_allocation;
static int kmeans_calls, kmeans_status, nonfinite_output;

static void *tracked_malloc(size_t bytes) {
    ++allocation_calls;
    if (allocation_calls == fail_allocation)
        return NULL;
    union allocation_header *header = malloc(sizeof(*header) + bytes);
    assert(header);
    header->bytes = bytes;
    live_bytes += bytes;
    if (live_bytes > peak_bytes)
        peak_bytes = live_bytes;
    return header + 1;
}

static void tracked_free(void *pointer) {
    if (!pointer)
        return;
    union allocation_header *header = (union allocation_header *)pointer - 1;
    assert(live_bytes >= header->bytes);
    live_bytes -= header->bytes;
    free(header);
}

#define malloc tracked_malloc
#define free tracked_free
#include "../src/data_path.c"
#undef malloc
#undef free

int32_t vextor_kmeans_v1(const float *points, int32_t rows, int32_t dims,
                         int32_t roots, int32_t leaves, int32_t sampling,
                         int32_t batch, int32_t iterations, int32_t restarts,
                         int32_t metric, int32_t spherical, int32_t seed,
                         float *nodes, int32_t *parents, int32_t *assignments,
                         float *current, float *sums, int32_t *counts) {
    (void)points; (void)sampling; (void)batch; (void)iterations; (void)restarts;
    (void)metric; (void)spherical; (void)seed; (void)current; (void)sums; (void)counts;
    ++kmeans_calls;
    for (int32_t i = 0; i < 1 + roots + leaves; ++i) {
        parents[i] = i == 0 ? -1 : 0;
        for (int32_t d = 0; d < dims; ++d)
            nodes[(size_t)i * dims + d] = nonfinite_output ? INFINITY : 0.0f;
    }
    for (int32_t i = 0; i < rows; ++i)
        assignments[i] = 0;
    return kmeans_status;
}

struct source {
    uint64_t count, position, calls;
    int fail;
    float values[2];
};

static int32_t next_row(void *context, struct vextor_input_row *row) {
    struct source *source = context;
    ++source->calls;
    if (source->fail)
        return -1;
    if (source->position == source->count)
        return 0;
    source->values[0] = (float)source->position++;
    source->values[1] = 0.0f;
    *row = (struct vextor_input_row){source->values, 2, VEXTOR_INPUT_VECTOR};
    return 1;
}

static void reset(void) {
    assert(live_bytes == 0);
    peak_bytes = allocation_calls = fail_allocation = 0;
    kmeans_calls = kmeans_status = nonfinite_output = 0;
}

int main(void) {
    struct vextor_cpu_options o = {
        .row_count = 2000, .memory_budget_bytes = UINT64_MAX, .dims = 2,
        .leaf_count = 2, .sampling_factor = 3, .batch_size = 4,
        .iterations = 4, .restarts = 2, .seed = 42
    };
    struct vextor_cpu_plan plan;
    struct vextor_data_error error;
    struct vextor_cpu_result result;
    assert(vextor_plan_cpu_v1(&o, &plan, &error) == VEXTOR_DATA_OK);
    assert(allocation_calls == 0);
    o.memory_budget_bytes = plan.buffer_bytes - 1;
    struct source source = {.count = o.row_count};
    assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_BUDGET);
    assert(allocation_calls == 0 && source.calls == 0 && kmeans_calls == 0);
    o.memory_budget_bytes = plan.buffer_bytes;

    for (int large = 0; large < 2; ++large) {
        reset();
        o.row_count = large ? 200000 : 2000;
        source = (struct source){.count = o.row_count};
        assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_OK);
        assert(peak_bytes == plan.buffer_bytes);
        assert(result.plan.sample_count == 6);
        assert(source.calls == o.row_count + 1 && kmeans_calls == 1);
        for (uint64_t i = 0; i < result.plan.sample_count; ++i)
            assert(result.sample[2 * i] == (float)result.sample_rows[i]);
        vextor_free_cpu_result_v1(&result);
        assert(live_bytes == 0);
    }
    size_t successful_allocations = allocation_calls;
    for (size_t allocation = 1; allocation <= successful_allocations; ++allocation) {
        reset();
        fail_allocation = allocation;
        source = (struct source){.count = o.row_count};
        assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_MEMORY);
        assert(live_bytes == 0 && source.calls == 0 && kmeans_calls == 0);
        assert(result.sample == NULL && result.nodes == NULL);
    }
    reset();
    source = (struct source){.count = o.row_count, .fail = 1};
    assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_SOURCE);
    assert(live_bytes == 0 && kmeans_calls == 0);
    for (int nonfinite = 0; nonfinite < 2; ++nonfinite) {
        reset();
        kmeans_status = nonfinite ? 0 : 1;
        nonfinite_output = nonfinite;
        source = (struct source){.count = o.row_count};
        assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_KMEANS);
        assert(live_bytes == 0 && kmeans_calls == 1);
        assert(result.sample == NULL && result.nodes == NULL);
    }
    reset();
    o.row_count = INT32_MAX;
    o.dims = INT32_MAX;
    o.leaf_count = INT32_MAX - 1;
    o.sampling_factor = 1;
    source = (struct source){.count = o.row_count};
    assert(vextor_run_cpu_v1(&o, next_row, &source, &result, &error) == VEXTOR_DATA_OVERFLOW);
    assert(allocation_calls == 0 && source.calls == 0 && kmeans_calls == 0);
    vextor_free_cpu_result_v1(&result);
    puts("Data path allocation, bounded-memory, and failure checks passed");
    return 0;
}
