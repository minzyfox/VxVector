#define _POSIX_C_SOURCE 200809L
#include "vextor_data_path.h"
#include "vextor_kmeans.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct sample_position {
    uint64_t row, destination;
};

static int32_t fail(struct vextor_data_error *error, int32_t code,
                    const char *message) {
    if (error) {
        error->code = code;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return code;
}

static int add_buffer(uint64_t *total, uint64_t count, uint64_t width) {
    uint64_t limit = SIZE_MAX < PTRDIFF_MAX ? SIZE_MAX : PTRDIFF_MAX;
    if (count > limit / width || *total > limit - count * width)
        return 0;
    *total += count * width;
    return 1;
}

int32_t vextor_plan_cpu_v1(const struct vextor_cpu_options *o,
                          struct vextor_cpu_plan *plan,
                          struct vextor_data_error *error) {
    if (plan)
        memset(plan, 0, sizeof(*plan));
    if (!o || !plan || o->row_count == 0 || o->row_count > INT64_MAX ||
        o->dims <= 0 || o->root_count < 0 || o->leaf_count <= 0 ||
        o->root_count > o->leaf_count ||
        (uint64_t)o->leaf_count > o->row_count ||
        (int64_t)o->root_count + o->leaf_count >= INT32_MAX ||
        o->sampling_factor <= 0 || o->batch_size <= 0 || o->iterations <= 0 ||
        o->restarts <= 0 || o->metric < 0 || o->metric > 2 ||
        (o->spherical != 0 && o->spherical != 1) || o->seed < 0)
        return fail(error, VEXTOR_DATA_INVALID, "Invalid CPU data path shape or option");

    struct vextor_cpu_plan p = {0};
    uint64_t requested = (uint64_t)o->leaf_count * o->sampling_factor;
    p.sample_count = requested < o->row_count ? requested : o->row_count;
    if (p.sample_count > INT32_MAX)
        return fail(error, VEXTOR_DATA_OVERFLOW, "Sample count exceeds the Vx i32 ABI");
    p.batch_rows = (uint64_t)o->batch_size < o->row_count ?
                   (uint64_t)o->batch_size : o->row_count;
    p.node_count = 1 + (uint64_t)o->root_count + o->leaf_count;
    p.hash_slots = 1;
    while (p.hash_slots < 2 * p.sample_count)
        p.hash_slots *= 2;
    uint64_t dims = (uint64_t)o->dims;
    if (!add_buffer(&p.buffer_bytes, p.sample_count * dims, sizeof(float)) ||
        !add_buffer(&p.buffer_bytes, p.batch_rows * dims, sizeof(float)) ||
        !add_buffer(&p.buffer_bytes, p.node_count * dims, sizeof(float)) ||
        !add_buffer(&p.buffer_bytes, p.node_count, sizeof(int32_t)) ||
        !add_buffer(&p.buffer_bytes, p.sample_count, sizeof(int32_t)) ||
        !add_buffer(&p.buffer_bytes, (uint64_t)o->leaf_count * dims, sizeof(float)) ||
        !add_buffer(&p.buffer_bytes, (uint64_t)o->leaf_count * dims, sizeof(float)) ||
        !add_buffer(&p.buffer_bytes, (uint64_t)o->leaf_count, sizeof(int32_t)) ||
        !add_buffer(&p.buffer_bytes, p.sample_count, sizeof(uint64_t)) ||
        !add_buffer(&p.buffer_bytes, p.sample_count, sizeof(struct sample_position)) ||
        !add_buffer(&p.buffer_bytes, p.hash_slots, sizeof(uint64_t)))
        return fail(error, VEXTOR_DATA_OVERFLOW, "CPU buffer sizes exceed the addressable allocation limit");
    *plan = p;
    if (p.buffer_bytes > o->memory_budget_bytes)
        return fail(error, VEXTOR_DATA_BUDGET, "CPU buffers exceed the configured memory budget");
    return fail(error, VEXTOR_DATA_OK, "");
}

#ifndef VEXTOR_PLAN_ONLY
/* SplitMix64 defines the seed sequence independently of the platform C library. */
static uint64_t random_u64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t random_below(uint64_t *state, uint64_t bound) {
    uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t value;
    do {
        value = random_u64(state);
    } while (value < threshold);
    return value % bound;
}

static uint64_t hash_slot(uint64_t *table, uint64_t mask, uint64_t row) {
    uint64_t slot = (row * UINT64_C(0x9e3779b97f4a7c15)) & mask;
    while (table[slot] != UINT64_MAX && table[slot] != row)
        slot = (slot + 1) & mask;
    return slot;
}

static void sift_down(struct sample_position *positions, uint64_t root, uint64_t count) {
    while (2 * root + 1 < count) {
        uint64_t child = 2 * root + 1;
        if (child + 1 < count && positions[child].row < positions[child + 1].row)
            ++child;
        if (positions[root].row >= positions[child].row)
            break;
        struct sample_position temporary = positions[root];
        positions[root] = positions[child];
        positions[child] = temporary;
        root = child;
    }
}

/* Heapsort uses no hidden heap allocation beyond the budgeted positions. */
static void sort_positions(struct sample_position *positions, uint64_t count) {
    for (uint64_t i = count / 2; i > 0; --i)
        sift_down(positions, i - 1, count);
    for (uint64_t end = count - 1; end > 0; --end) {
        struct sample_position temporary = positions[0];
        positions[0] = positions[end];
        positions[end] = temporary;
        sift_down(positions, 0, end);
    }
}

static double seconds(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0)
        return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

void vextor_free_cpu_result_v1(struct vextor_cpu_result *result) {
    if (!result)
        return;
    free(result->sample);
    free(result->sample_rows);
    free(result->nodes);
    free(result->parents);
    free(result->assignments);
    memset(result, 0, sizeof(*result));
}

int32_t vextor_run_cpu_v1(const struct vextor_cpu_options *o,
                         vextor_next_row_fn next, void *context,
                         struct vextor_cpu_result *result,
                         struct vextor_data_error *error) {
    if (!result)
        return fail(error, VEXTOR_DATA_INVALID, "A CPU result destination is required");
    memset(result, 0, sizeof(*result));
    if (!next)
        return fail(error, VEXTOR_DATA_INVALID, "A source row reader is required");
    struct vextor_cpu_result r = {0};
    int32_t status = vextor_plan_cpu_v1(o, &r.plan, error);
    if (status != VEXTOR_DATA_OK)
        return status;
    struct vextor_cpu_plan p = r.plan;
    size_t dims = (size_t)o->dims;
    r.sample = malloc((size_t)p.sample_count * dims * sizeof(float));
    r.sample_rows = malloc((size_t)p.sample_count * sizeof(uint64_t));
    r.nodes = malloc((size_t)p.node_count * dims * sizeof(float));
    r.parents = malloc((size_t)p.node_count * sizeof(int32_t));
    r.assignments = malloc((size_t)p.sample_count * sizeof(int32_t));
    float *batch = malloc((size_t)p.batch_rows * dims * sizeof(float));
    float *current = malloc((size_t)o->leaf_count * dims * sizeof(float));
    float *sums = malloc((size_t)o->leaf_count * dims * sizeof(float));
    int32_t *counts = malloc((size_t)o->leaf_count * sizeof(int32_t));
    struct sample_position *positions = malloc((size_t)p.sample_count * sizeof(*positions));
    uint64_t *table = malloc((size_t)p.hash_slots * sizeof(uint64_t));
    if (!r.sample || !r.sample_rows || !r.nodes || !r.parents || !r.assignments ||
        !batch || !current || !sums || !counts || !positions || !table) {
        status = fail(error, VEXTOR_DATA_MEMORY, "Could not allocate CPU data path buffers");
        goto cleanup;
    }

    double start = seconds();
    memset(table, 0xff, (size_t)p.hash_slots * sizeof(uint64_t));
    uint64_t state = (uint64_t)o->seed;
    /* Floyd's algorithm selects distinct source positions with O(sample) storage. */
    for (uint64_t i = 0; i < p.sample_count; ++i) {
        uint64_t j = o->row_count - p.sample_count + i;
        uint64_t row = random_below(&state, j + 1);
        uint64_t slot = hash_slot(table, p.hash_slots - 1, row);
        if (table[slot] != UINT64_MAX) {
            row = j;
            slot = hash_slot(table, p.hash_slots - 1, row);
        }
        table[slot] = row;
        positions[i] = (struct sample_position){row, i};
        r.sample_rows[i] = row;
    }
    sort_positions(positions, p.sample_count);
    r.sample_seconds = seconds() - start;

    uint64_t selected = 0;
    for (uint64_t base = 0; base < o->row_count;) {
        uint64_t remaining = o->row_count - base;
        uint64_t rows = remaining < p.batch_rows ? remaining : p.batch_rows;
        start = seconds();
        for (uint64_t row = 0; row < rows; ++row) {
            struct vextor_input_row input = {0};
            int32_t read_status = next(context, &input);
            if (read_status != 1) {
                status = fail(error, VEXTOR_DATA_SOURCE,
                              read_status == 0 ? "Source ended before the declared row count" : "Source row reader failed");
                goto cleanup;
            }
            if (!input.values || input.type != VEXTOR_INPUT_VECTOR || input.dims != dims) {
                status = fail(error, VEXTOR_DATA_INVALID, "Source contains a null vector, unsupported type, or mixed dimensions");
                goto cleanup;
            }
            for (size_t d = 0; d < dims; ++d) {
                float value = input.values[d];
                if (!isfinite(value)) {
                    status = fail(error, VEXTOR_DATA_INVALID, "Source contains a nonfinite vector component");
                    goto cleanup;
                }
                batch[(size_t)row * dims + d] = value;
            }
        }
        r.read_seconds += seconds() - start;
        start = seconds();
        while (selected < p.sample_count && positions[selected].row < base + rows) {
            struct sample_position position = positions[selected++];
            memcpy(r.sample + (size_t)position.destination * dims,
                   batch + (size_t)(position.row - base) * dims,
                   dims * sizeof(float));
        }
        r.sample_seconds += seconds() - start;
        base += rows;
    }
    struct vextor_input_row extra = {0};
    start = seconds();
    int32_t read_status = next(context, &extra);
    r.read_seconds += seconds() - start;
    if (read_status != 0) {
        status = fail(error, VEXTOR_DATA_SOURCE,
                      read_status == 1 ? "Source exceeds the declared row count" : "Source row reader failed");
        goto cleanup;
    }

    start = seconds();
    status = vextor_kmeans_v1(r.sample, (int32_t)p.sample_count, o->dims,
                             o->root_count, o->leaf_count, o->sampling_factor,
                             o->batch_size, o->iterations, o->restarts, o->metric,
                             o->spherical, o->seed, r.nodes, r.parents,
                             r.assignments, current, sums, counts);
    r.clustering_seconds = seconds() - start;
    if (status != 0) {
        status = fail(error, VEXTOR_DATA_KMEANS, "Vx CPU k-means rejected the collected sample");
        goto cleanup;
    }
    for (uint64_t i = 0; i < p.node_count * dims; ++i) {
        if (!isfinite(r.nodes[i])) {
            status = fail(error, VEXTOR_DATA_KMEANS, "Vx CPU k-means produced a nonfinite centroid");
            goto cleanup;
        }
    }
    *result = r;
    memset(&r, 0, sizeof(r));
    status = fail(error, VEXTOR_DATA_OK, "");

cleanup:
    free(batch);
    free(current);
    free(sums);
    free(counts);
    free(positions);
    free(table);
    vextor_free_cpu_result_v1(&r);
    return status;
}

#endif /* VEXTOR_PLAN_ONLY */
