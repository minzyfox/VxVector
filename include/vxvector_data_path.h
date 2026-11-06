#ifndef VXVECTOR_DATA_PATH_H
#define VXVECTOR_DATA_PATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vxvector_data_status {
    VXVECTOR_DATA_OK = 0,
    VXVECTOR_DATA_INVALID = 1,
    VXVECTOR_DATA_BUDGET = 2,
    VXVECTOR_DATA_OVERFLOW = 3,
    VXVECTOR_DATA_MEMORY = 4,
    VXVECTOR_DATA_SOURCE = 5,
    VXVECTOR_DATA_KMEANS = 6
};

enum vxvector_input_type { VXVECTOR_INPUT_VECTOR = 0 };

struct vxvector_cpu_options {
    uint64_t row_count;
    uint64_t memory_budget_bytes;
    int32_t dims, root_count, leaf_count, sampling_factor, batch_size;
    int32_t iterations, restarts, metric, spherical, seed;
};

struct vxvector_cpu_plan {
    uint64_t sample_count, batch_rows, node_count, hash_slots;
    uint64_t buffer_bytes;
};

struct vxvector_data_error {
    int32_t code;
    char message[256];
};

struct vxvector_input_row {
    const float *values;
    uint32_t dims;
    int32_t type;
};

/* Return 1 for a row, 0 for EOF, or a negative value for a source error.
 * Row storage must remain valid until the next callback. The caller must count
 * and scan the source in one snapshot and use a stable scan order. The reader
 * must not unwind or longjmp across this C call. Translate cancellation into a
 * negative return. Database adapters must also check the declared column type,
 * including for an empty table. Only dense pgvector vector/f32 input is accepted.
 */
typedef int32_t (*vxvector_next_row_fn)(void *context, struct vxvector_input_row *row);

struct vxvector_cpu_result {
    struct vxvector_cpu_plan plan;
    double read_seconds, sample_seconds, clustering_seconds;
    float *sample;
    uint64_t *sample_rows;
    float *nodes;
    int32_t *parents;
    int32_t *assignments;
};

/* Planning performs no allocation. buffer_bytes includes every heap buffer
 * owned by this data path, including sampling indexes and training scratch.
 * It excludes allocator overhead, the caller's row storage, and the Vx runtime.
 * Counts use 64 bits. The sampled row count and Vx options must fit the i32 ABI.
 */
int32_t vxvector_plan_cpu_v1(const struct vxvector_cpu_options *options,
                          struct vxvector_cpu_plan *plan,
                          struct vxvector_data_error *error);

/* Reject invalid requests and budget excess before allocation or reader calls.
 * Validate every source row, including rows outside the sample. Require exactly
 * row_count rows followed by EOF. Retain sampled rows in seeded selection order.
 * nodes, parents, and assignments follow vxvector_kmeans.h; assignments cover the
 * sample, not the full source. Release a previous result before reusing it.
 * On failure, result is empty. error may be NULL. No GPU library is needed.
 */
int32_t vxvector_run_cpu_v1(const struct vxvector_cpu_options *options,
                         vxvector_next_row_fn next, void *context,
                         struct vxvector_cpu_result *result,
                         struct vxvector_data_error *error);
void vxvector_free_cpu_result_v1(struct vxvector_cpu_result *result);

#ifdef __cplusplus
}
#endif
#endif
