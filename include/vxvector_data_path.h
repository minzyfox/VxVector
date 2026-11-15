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
    VXVECTOR_DATA_KMEANS = 6,
    /* Backend selection failed, or GPU training failed after it started. */
    VXVECTOR_DATA_GPU = 7
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

/* Backend selection (Step 4 of docs/pg_vxvector.md). */
enum vxvector_backend_mode {
    VXVECTOR_MODE_AUTO = 0, /* the GPU when it can run the request, else the CPU */
    VXVECTOR_MODE_CPU = 1,  /* the CPU, even when a GPU is available */
    VXVECTOR_MODE_GPU = 2   /* the GPU, or an error before any row is read */
};

/* What the caller found when it looked for the GPU library and a device. */
enum vxvector_gpu_probe {
    VXVECTOR_PROBE_READY = 0,
    VXVECTOR_PROBE_NOT_INSTALLED = 1,
    VXVECTOR_PROBE_UNAVAILABLE = 2
};

enum vxvector_backend { VXVECTOR_BACKEND_CPU = 0, VXVECTOR_BACKEND_GPU = 1 };

enum vxvector_backend_reason {
    VXVECTOR_REASON_CPU_REQUESTED = 0,
    VXVECTOR_REASON_GPU_SELECTED = 1,
    VXVECTOR_REASON_GPU_NOT_INSTALLED = 2,
    VXVECTOR_REASON_GPU_UNAVAILABLE = 3,
    VXVECTOR_REASON_GPU_METRIC = 4,
    VXVECTOR_REASON_GPU_RESTARTS = 5,
    VXVECTOR_REASON_GPU_SPHERICAL = 6,
    VXVECTOR_REASON_GPU_ROOTS = 7,
    VXVECTOR_REASON_GPU_SIZE = 8,
    VXVECTOR_REASON_GPU_BUDGET = 9
};

struct vxvector_backend_choice {
    int32_t backend, reason;
    /* GPU memory that training will allocate, beyond plan.buffer_bytes. */
    uint64_t gpu_bytes;
};

/* The GPU functions of the data path: vxvector_gpu_restart_v1 and
 * vxvector_gpu_assign_v1 in vxvector_gpu.h. Each returns 0 on success, 1 when
 * it rejects the request, and 2 when the GPU fails after training starts. */
struct vxvector_gpu_trainer {
    int32_t (*restart)(const float *points, int32_t rows, int32_t dims, int32_t leaf_count,
                       int32_t iterations, int32_t metric, int32_t spherical, int32_t restart,
                       float *centers, int32_t *labels);
    int32_t (*assign)(const float *points, int32_t rows, int32_t dims, const float *centers,
                      int32_t k, int32_t metric, int32_t *labels);
};

/* Choose the backend after planning and before any row is read or any GPU
 * memory is allocated. The GPU must be ready, fit the i32 kernel limits,
 * and fit memory_budget_bytes together with the CPU plan. In MODE_AUTO, any
 * of these failing selects the CPU, and choice gives the reason. In MODE_GPU
 * it returns VXVECTOR_DATA_GPU, and choice still gives the reason. */
int32_t vxvector_select_backend_v1(const struct vxvector_cpu_options *options,
                                   const struct vxvector_cpu_plan *plan, int32_t mode,
                                   int32_t probe, struct vxvector_backend_choice *choice,
                                   struct vxvector_data_error *error);

/* A sentence that explains a reason code. */
const char *vxvector_backend_reason_v1(int32_t reason);

/* vxvector_run_cpu_v1 with the backend that vxvector_select_backend_v1 chose.
 * gpu is required for the GPU and ignored for the CPU. On the GPU, each
 * restart trains there, the data path keeps the restart with the lowest
 * float64 objective, and roots train on the CPU. The result has the same
 * layout on both. A GPU failure after training starts returns
 * VXVECTOR_DATA_GPU. The data path does not train again on the CPU. */
int32_t vxvector_run_v2(const struct vxvector_cpu_options *options,
                        const struct vxvector_backend_choice *choice,
                        const struct vxvector_gpu_trainer *gpu,
                        vxvector_next_row_fn next, void *context,
                        struct vxvector_cpu_result *result,
                        struct vxvector_data_error *error);

#ifdef __cplusplus
}
#endif
#endif
