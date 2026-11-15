/* Check actual allocations and failure cleanup of the C data path. The real Vx
 * data path runs through a wrapper that can replace its outcome with a k-means
 * failure, which real input cannot cause. */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vxvector_data_path.h"

union allocation_header {
    max_align_t alignment;
    size_t bytes;
};

static size_t live_bytes, peak_bytes, allocation_calls, fail_allocation;
static int run_calls, forced_outcome;

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
#define vxvector_run_reason_v2 checked_run
#include "../src/data_path.c"
#undef malloc
#undef free
#undef vxvector_run_reason_v2

int32_t vxvector_run_reason_v2(const struct vxvector_cpu_options *options,
                                   const struct vxvector_cpu_plan *plan,
                                   vxvector_next_row_fn next, void *context,
                                   struct vxvector_input_row *row, double (*clock)(void),
                                   const struct vxvector_cpu_buffers *buffers, int32_t backend,
                                   int32_t (*gpu_restart)(const float *, int32_t, int32_t,
                                                          int32_t, int32_t, int32_t, int32_t,
                                                          float *, int32_t *),
                                   int32_t (*gpu_assign)(const float *, int32_t, int32_t,
                                                         const float *, int32_t, int32_t,
                                                         int32_t *));

int32_t checked_run(const struct vxvector_cpu_options *options,
                    const struct vxvector_cpu_plan *plan,
                    vxvector_next_row_fn next, void *context,
                    struct vxvector_input_row *row, double (*clock)(void),
                    const struct vxvector_cpu_buffers *buffers, int32_t backend,
                    int32_t (*gpu_restart)(const float *, int32_t, int32_t, int32_t, int32_t,
                                           int32_t, int32_t, float *, int32_t *),
                    int32_t (*gpu_assign)(const float *, int32_t, int32_t, const float *,
                                          int32_t, int32_t, int32_t *)) {
    ++run_calls;
    int32_t outcome = vxvector_run_reason_v2(options, plan, next, context, row, clock, buffers,
                                             backend, gpu_restart, gpu_assign);
    return outcome == OUTCOME_OK && forced_outcome ? forced_outcome : outcome;
}

struct source {
    uint64_t count, position, calls;
    int fail;
    float values[2];
};

static int32_t next_row(void *context, struct vxvector_input_row *row) {
    struct source *source = context;
    ++source->calls;
    if (source->fail)
        return -1;
    if (source->position == source->count)
        return 0;
    source->values[0] = (float)source->position++;
    source->values[1] = 0.0f;
    *row = (struct vxvector_input_row){source->values, 2, VXVECTOR_INPUT_VECTOR};
    return 1;
}

static void reset(void) {
    assert(live_bytes == 0);
    peak_bytes = allocation_calls = fail_allocation = 0;
    run_calls = forced_outcome = 0;
}

int main(void) {
    struct vxvector_cpu_options o = {
        .row_count = 2000, .memory_budget_bytes = UINT64_MAX, .dims = 2,
        .leaf_count = 2, .sampling_factor = 3, .batch_size = 4,
        .iterations = 4, .restarts = 2, .seed = 42
    };
    struct vxvector_cpu_plan plan;
    struct vxvector_data_error error;
    struct vxvector_cpu_result result;
    assert(vxvector_plan_cpu_v1(&o, &plan, &error) == VXVECTOR_DATA_OK);
    assert(allocation_calls == 0);
    o.memory_budget_bytes = plan.buffer_bytes - 1;
    struct source source = {.count = o.row_count};
    assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_BUDGET);
    assert(allocation_calls == 0 && source.calls == 0 && run_calls == 0);
    o.memory_budget_bytes = plan.buffer_bytes;

    for (int large = 0; large < 2; ++large) {
        reset();
        o.row_count = large ? 200000 : 2000;
        source = (struct source){.count = o.row_count};
        assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_OK);
        assert(peak_bytes == plan.buffer_bytes);
        assert(result.plan.sample_count == 6);
        assert(source.calls == o.row_count + 1 && run_calls == 1);
        for (uint64_t i = 0; i < result.plan.sample_count; ++i)
            assert(result.sample[2 * i] == (float)result.sample_rows[i]);
        vxvector_free_cpu_result_v1(&result);
        assert(live_bytes == 0);
    }
    size_t successful_allocations = allocation_calls;
    for (size_t allocation = 1; allocation <= successful_allocations; ++allocation) {
        reset();
        fail_allocation = allocation;
        source = (struct source){.count = o.row_count};
        assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_MEMORY);
        assert(live_bytes == 0 && source.calls == 0 && run_calls == 0);
        assert(result.sample == NULL && result.nodes == NULL);
    }
    reset();
    source = (struct source){.count = o.row_count, .fail = 1};
    assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_SOURCE);
    assert(live_bytes == 0 && run_calls == 1);
    for (int nonfinite = 0; nonfinite < 2; ++nonfinite) {
        reset();
        forced_outcome = nonfinite ? OUTCOME_NONFINITE_CENTROID : OUTCOME_KMEANS_REJECTED;
        source = (struct source){.count = o.row_count};
        assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_KMEANS);
        assert(live_bytes == 0 && run_calls == 1);
        assert(result.sample == NULL && result.nodes == NULL);
    }
    reset();
    o.row_count = INT32_MAX;
    o.dims = INT32_MAX;
    o.leaf_count = INT32_MAX - 1;
    o.sampling_factor = 1;
    source = (struct source){.count = o.row_count};
    assert(vxvector_run_cpu_v1(&o, next_row, &source, &result, &error) == VXVECTOR_DATA_OVERFLOW);
    assert(allocation_calls == 0 && source.calls == 0 && run_calls == 0);
    vxvector_free_cpu_result_v1(&result);
    puts("Data path allocation, bounded-memory, and failure checks passed");
    return 0;
}
