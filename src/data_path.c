#define _POSIX_C_SOURCE 200809L
/* The C side of the CPU data path. The option checks, buffer sizing, sampling,
 * row reads and checks, and clustering are in src/kmeans.vx. This file keeps
 * the parts that stay in C:
 *
 * - Allocation. The data path owns its result buffers, which
 *   vxvector_free_cpu_result_v1 releases, and tests/data_path_memory.c injects
 *   allocation failures here.
 * - The clock. Vx's clock is in its runtime library, which the PostgreSQL
 *   extension and the Linux worker do not link.
 * - The messages. Vx returns an outcome, and C writes its status code and text.
 */
#include "vxvector_data_path.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The outcomes of src/kmeans.vx, in its order. */
enum vxvector_data_outcome {
    OUTCOME_OK,
    OUTCOME_INVALID_OPTION,
    OUTCOME_SAMPLE_OVERFLOW,
    OUTCOME_BUFFER_OVERFLOW,
    OUTCOME_BUDGET,
    OUTCOME_SOURCE_SHORT,
    OUTCOME_SOURCE_LONG,
    OUTCOME_SOURCE_FAILED,
    OUTCOME_INVALID_ROW,
    OUTCOME_NONFINITE_ROW,
    OUTCOME_KMEANS_REJECTED,
    OUTCOME_NONFINITE_CENTROID,
    OUTCOME_COUNT
};

static const struct {
    int32_t code;
    const char *message;
} outcomes[OUTCOME_COUNT] = {
    {VXVECTOR_DATA_OK, ""},
    {VXVECTOR_DATA_INVALID, "Invalid CPU data path shape or option"},
    {VXVECTOR_DATA_OVERFLOW, "Sample count exceeds the Vx i32 ABI"},
    {VXVECTOR_DATA_OVERFLOW, "CPU buffer sizes exceed the addressable allocation limit"},
    {VXVECTOR_DATA_BUDGET, "CPU buffers exceed the configured memory budget"},
    {VXVECTOR_DATA_SOURCE, "Source ended before the declared row count"},
    {VXVECTOR_DATA_SOURCE, "Source exceeds the declared row count"},
    {VXVECTOR_DATA_SOURCE, "Source row reader failed"},
    {VXVECTOR_DATA_INVALID, "Source contains a null vector, unsupported type, or mixed dimensions"},
    {VXVECTOR_DATA_INVALID, "Source contains a nonfinite vector component"},
    {VXVECTOR_DATA_KMEANS, "Vx CPU k-means rejected the collected sample"},
    {VXVECTOR_DATA_KMEANS, "Vx CPU k-means produced a nonfinite centroid"},
};

/* The buffers of one run. This mirrors CpuBuffers in src/kmeans.vx. */
struct vxvector_cpu_buffers {
    float *sample;
    uint64_t *sample_rows;
    float *nodes;
    int32_t *parents;
    int32_t *assignments;
    float *batch;
    float *current;
    float *sums;
    int32_t *counts;
    uint64_t *positions;
    uint64_t *table;
    double *timings;
};

/* Defined in src/kmeans.vx. */
int32_t vxvector_plan_cpu_reason_v1(const struct vxvector_cpu_options *options,
                                    struct vxvector_cpu_plan *plan);
int32_t vxvector_run_cpu_reason_v1(const struct vxvector_cpu_options *options,
                                   const struct vxvector_cpu_plan *plan,
                                   vxvector_next_row_fn next, void *context,
                                   struct vxvector_input_row *row, double (*clock)(void),
                                   const struct vxvector_cpu_buffers *buffers);

static int32_t fail(struct vxvector_data_error *error, int32_t code,
                    const char *message) {
    if (error) {
        error->code = code;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return code;
}

static int32_t report(struct vxvector_data_error *error, int32_t outcome) {
    if (outcome < 0 || outcome >= OUTCOME_COUNT)
        return fail(error, VXVECTOR_DATA_INVALID, "Unknown Vx data path outcome");
    return fail(error, outcomes[outcome].code, outcomes[outcome].message);
}

int32_t vxvector_plan_cpu_v1(const struct vxvector_cpu_options *o,
                             struct vxvector_cpu_plan *plan,
                             struct vxvector_data_error *error) {
    return report(error, vxvector_plan_cpu_reason_v1(o, plan));
}

#ifndef VXVECTOR_PLAN_ONLY
static double seconds(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0)
        return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

void vxvector_free_cpu_result_v1(struct vxvector_cpu_result *result) {
    if (!result)
        return;
    free(result->sample);
    free(result->sample_rows);
    free(result->nodes);
    free(result->parents);
    free(result->assignments);
    memset(result, 0, sizeof(*result));
}

int32_t vxvector_run_cpu_v1(const struct vxvector_cpu_options *o,
                            vxvector_next_row_fn next, void *context,
                            struct vxvector_cpu_result *result,
                            struct vxvector_data_error *error) {
    if (!result)
        return fail(error, VXVECTOR_DATA_INVALID, "A CPU result destination is required");
    memset(result, 0, sizeof(*result));
    if (!next)
        return fail(error, VXVECTOR_DATA_INVALID, "A source row reader is required");
    struct vxvector_cpu_result r = {0};
    int32_t status = vxvector_plan_cpu_v1(o, &r.plan, error);
    if (status != VXVECTOR_DATA_OK)
        return status;
    struct vxvector_cpu_plan p = r.plan;
    size_t dims = (size_t)o->dims;
    r.sample = malloc((size_t)p.sample_count * dims * sizeof(float));
    r.sample_rows = malloc((size_t)p.sample_count * sizeof(uint64_t));
    r.nodes = malloc((size_t)p.node_count * dims * sizeof(float));
    r.parents = malloc((size_t)p.node_count * sizeof(int32_t));
    r.assignments = malloc((size_t)p.sample_count * sizeof(int32_t));
    struct vxvector_cpu_buffers b = {
        .sample = r.sample, .sample_rows = r.sample_rows, .nodes = r.nodes,
        .parents = r.parents, .assignments = r.assignments,
        .batch = malloc((size_t)p.batch_rows * dims * sizeof(float)),
        .current = malloc((size_t)o->leaf_count * dims * sizeof(float)),
        .sums = malloc((size_t)o->leaf_count * dims * sizeof(float)),
        .counts = malloc((size_t)o->leaf_count * sizeof(int32_t)),
        /* (row, destination) pairs. They are the kernel's sample scratch later. */
        .positions = malloc((size_t)p.sample_count * 2 * sizeof(uint64_t)),
        .table = malloc((size_t)p.hash_slots * sizeof(uint64_t)),
        .timings = &r.read_seconds,
    };
    if (!r.sample || !r.sample_rows || !r.nodes || !r.parents || !r.assignments ||
        !b.batch || !b.current || !b.sums || !b.counts || !b.positions || !b.table) {
        status = fail(error, VXVECTOR_DATA_MEMORY, "Could not allocate CPU data path buffers");
        goto cleanup;
    }

    struct vxvector_input_row row = {0};
    status = report(error, vxvector_run_cpu_reason_v1(o, &p, next, context, &row, seconds, &b));
    if (status == VXVECTOR_DATA_OK) {
        *result = r;
        memset(&r, 0, sizeof(r));
    }

cleanup:
    free(b.batch);
    free(b.current);
    free(b.sums);
    free(b.counts);
    free(b.positions);
    free(b.table);
    vxvector_free_cpu_result_v1(&r);
    return status;
}
#endif /* VXVECTOR_PLAN_ONLY */
