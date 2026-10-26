/* Stream generated vectors through the CPU adapter without a resident table. */
#define _POSIX_C_SOURCE 200809L
#include "vextor_data_path.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct generated_source {
    uint64_t rows, position;
    int32_t dims;
    float *values;
};

static int32_t next_row(void *context, struct vextor_input_row *row) {
    struct generated_source *source = context;
    if (source->position == source->rows)
        return 0;
    uint64_t i = source->position++;
    for (int32_t d = 0; d < source->dims; ++d)
        source->values[d] = (float)((i % 4) * 10 + (i % 11 * 13 + (uint64_t)d * 7) % 11) / 11.0f;
    *row = (struct vextor_input_row){source->values, (uint32_t)source->dims, VEXTOR_INPUT_VECTOR};
    return 1;
}

static int parse(const char *text, uint64_t maximum, uint64_t *value) {
    char *end;
    errno = 0;
    if (*text < '0' || *text > '9')
        return 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || parsed == 0 || parsed > maximum)
        return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static double seconds(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0)
        return 0;
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    uint64_t values[] = {10000, 16, 8, 40, 64, 64 * 1024 * 1024};
    if (argc > 7) {
        fprintf(stderr, "Usage: %s [rows [dims [leaves [sampling_factor [batch_size [budget_bytes]]]]]]\n", argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; ++i) {
        uint64_t maximum = i == 1 ? INT64_MAX : (i == 6 ? UINT64_MAX : INT32_MAX);
        if (!parse(argv[i], maximum, &values[i - 1])) {
            fprintf(stderr, "Invalid size argument: %s\n", argv[i]);
            return 1;
        }
    }
    struct vextor_cpu_options options = {
        .row_count = values[0], .memory_budget_bytes = values[5],
        .dims = (int32_t)values[1], .leaf_count = (int32_t)values[2],
        .sampling_factor = (int32_t)values[3], .batch_size = (int32_t)values[4],
        .iterations = 10, .restarts = 1, .seed = 42
    };
    struct vextor_cpu_plan plan;
    struct vextor_data_error error;
    if (vextor_plan_cpu_v1(&options, &plan, &error) != VEXTOR_DATA_OK) {
        fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    struct generated_source source = {
        .rows = options.row_count, .dims = options.dims,
        .values = malloc((size_t)options.dims * sizeof(float))
    };
    if (!source.values) {
        fprintf(stderr, "Could not allocate the source row\n");
        return 1;
    }
    struct vextor_cpu_result result;
    double start = seconds();
    int32_t status = vextor_run_cpu_v1(&options, next_row, &source, &result, &error);
    double total = seconds() - start;
    free(source.values);
    if (status != VEXTOR_DATA_OK) {
        fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    printf("{\"rows\":%" PRIu64 ",\"dims\":%d,\"leaves\":%d,"
           "\"sampling_factor\":%d,\"batch_size\":%d,\"sample_rows\":%" PRIu64 ","
           "\"buffer_bytes\":%" PRIu64 ",\"read_seconds\":%.9f,"
           "\"sample_seconds\":%.9f,\"clustering_seconds\":%.9f,\"total_seconds\":%.9f}\n",
           options.row_count, options.dims, options.leaf_count, options.sampling_factor,
           options.batch_size, result.plan.sample_count, result.plan.buffer_bytes,
           result.read_seconds, result.sample_seconds, result.clustering_seconds, total);
    vextor_free_cpu_result_v1(&result);
    return 0;
}
