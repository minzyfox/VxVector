#include "vxvector_protocol.h"

#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#define GPU_LIBRARY "libvxvector_gpu.dylib"
#else
#define GPU_LIBRARY "libvxvector_gpu.so"
#endif

static int transfer(int fd, void *buffer, size_t bytes, int writing) {
    char *position = buffer;
    size_t remaining = bytes;
    while (remaining) {
        ssize_t count = writing ? write(fd, position, remaining) : read(fd, position, remaining);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return count == 0 && remaining == bytes ? 0 : -1;
        position += count;
        remaining -= (size_t)count;
    }
    return 1;
}

struct row_reader {
    uint32_t dims;
    float *values;
};

static int32_t next_row(void *context, struct vxvector_input_row *row) {
    struct row_reader *reader = context;
    uint32_t dims;
    int status = transfer(STDIN_FILENO, &dims, sizeof(dims), 0);
    if (status != 1)
        return status;
    if (dims != reader->dims)
        return -1;
    if (transfer(STDIN_FILENO, reader->values, (size_t)dims * sizeof(float), 0) != 1)
        return -1;
    *row = (struct vxvector_input_row){reader->values, dims, VXVECTOR_INPUT_VECTOR};
    return 1;
}

/* The optional GPU library. A CPU-only install has none, and the worker needs
 * no GPU library to load. VXVECTOR_GPU_LIBRARY names it; otherwise it is
 * next to the worker. */
/* The timings of the GPU library, as in vxvector_gpu.h. */
struct gpu_timings {
    double allocation_seconds, copy_in_seconds, training_seconds, copy_out_seconds;
    uint64_t calls;
};

struct gpu_library {
    int32_t (*probe)(char *device, size_t size);
    struct vxvector_gpu_trainer trainer;
    void (*reset_timings)(void);
    void (*timings)(struct gpu_timings *);
    const char *(*version)(void);
};

static int32_t load_gpu(const char *worker, struct gpu_library *gpu, char *device, size_t size) {
    char path[4096];
    const char *chosen = getenv("VXVECTOR_GPU_LIBRARY");
    if (!chosen || !*chosen) {
        const char *slash = strrchr(worker, '/');
        int length = slash ? (int)(slash - worker) : 1;
        snprintf(path, sizeof(path), "%.*s/%s", length, slash ? worker : ".", GPU_LIBRARY);
        chosen = path;
    }
    void *library = dlopen(chosen, RTLD_NOW | RTLD_LOCAL);
    if (!library)
        return VXVECTOR_PROBE_NOT_INSTALLED;
    *(void **)&gpu->probe = dlsym(library, "vxvector_gpu_probe_v1");
    *(void **)&gpu->trainer.restart = dlsym(library, "vxvector_gpu_restart_v1");
    *(void **)&gpu->trainer.assign = dlsym(library, "vxvector_gpu_assign_v1");
    *(void **)&gpu->reset_timings = dlsym(library, "vxvector_gpu_reset_timings_v1");
    *(void **)&gpu->timings = dlsym(library, "vxvector_gpu_timings_v1");
    *(void **)&gpu->version = dlsym(library, "vxvector_gpu_version_v1");
    if (!gpu->probe || !gpu->trainer.restart || !gpu->trainer.assign || !gpu->reset_timings ||
        !gpu->timings || !gpu->version)
        return VXVECTOR_PROBE_NOT_INSTALLED;
    return gpu->probe(device, size) == 0 ? VXVECTOR_PROBE_READY : VXVECTOR_PROBE_UNAVAILABLE;
}

/* Sends a reply in the format of the request's version. */
static int send_reply(int version, struct vxvector_worker_reply_v2 *reply) {
    if (version == 1)
        return transfer(STDOUT_FILENO, &reply->base, sizeof(reply->base), 1);
    return transfer(STDOUT_FILENO, reply, sizeof(*reply), 1);
}

static int fail(int version, struct vxvector_worker_reply_v2 *reply, int32_t status,
                const char *message) {
    reply->base.status = status;
    snprintf(reply->base.message, sizeof(reply->base.message), "%s", message);
    send_reply(version, reply);
    return 1;
}

int main(int argc, char **argv) {
    (void)argc;
    signal(SIGPIPE, SIG_IGN);
    struct vxvector_worker_request_v2 request = {0};
    struct vxvector_worker_reply_v2 reply = {0};
    struct vxvector_data_error error = {0};
    struct vxvector_cpu_plan plan;
    int version = 0;
    if (transfer(STDIN_FILENO, &request.magic, sizeof(request.magic), 0) == 1) {
        if (request.magic == VXVECTOR_WORKER_MAGIC)
            version = 1;
        else if (request.magic == VXVECTOR_WORKER_MAGIC_V2)
            version = 2;
    }
    size_t rest = version == 1 ? sizeof(struct vxvector_worker_request) - sizeof(request.magic)
                               : sizeof(request) - sizeof(request.magic);
    if (!version || transfer(STDIN_FILENO, &request.options, rest, 0) != 1)
        return fail(version ? version : 1, &reply, VXVECTOR_DATA_INVALID,
                    "Invalid or truncated worker request");
    /* A version 1 client expects the CPU and reports it, so it gets the CPU. */
    if (version == 1)
        request.backend_mode = VXVECTOR_MODE_CPU;

    uint64_t row_bytes = request.options.dims > 0 ? (uint64_t)request.options.dims * sizeof(float) : 0;
    if (request.options.memory_budget_bytes <= row_bytes)
        return fail(version, &reply, VXVECTOR_DATA_BUDGET,
                    "Memory budget cannot hold the worker input row");
    request.options.memory_budget_bytes -= row_bytes;
    reply.base.status = vxvector_plan_cpu_v1(&request.options, &plan, &error);
    if (reply.base.status != 0)
        return fail(version, &reply, reply.base.status, error.message);

    /* Choose the backend before any row is read or any GPU memory allocated. */
    struct gpu_library gpu = {0};
    struct vxvector_backend_choice choice = {0};
    int32_t probe = VXVECTOR_PROBE_NOT_INSTALLED;
    if (request.backend_mode != VXVECTOR_MODE_CPU)
        probe = load_gpu(argv[0], &gpu, reply.device, sizeof(reply.device));
    int32_t selected = vxvector_select_backend_v1(&request.options, &plan, request.backend_mode,
                                                  probe, &choice, &error);
    reply.backend = choice.backend;
    reply.reason = choice.reason;
    reply.gpu_bytes = choice.gpu_bytes;
    snprintf(reply.reason_text, sizeof(reply.reason_text), "%s",
             vxvector_backend_reason_v1(choice.reason));
    if (selected != 0)
        return fail(version, &reply, selected, error.message);
    if (choice.backend != VXVECTOR_BACKEND_GPU)
        reply.device[0] = '\0';
    else
        snprintf(reply.gpu_version, sizeof(reply.gpu_version), "%s", gpu.version());

    reply.base.dims = request.options.dims;
    reply.base.nodes = plan.node_count;
    reply.base.samples = plan.sample_count;
    reply.base.buffer_bytes = plan.buffer_bytes + row_bytes + choice.gpu_bytes;
    if (send_reply(version, &reply) != 1)
        return 1;
    struct row_reader reader = {(uint32_t)request.options.dims, malloc((size_t)row_bytes)};
    if (!reader.values)
        return fail(version, &reply, VXVECTOR_DATA_MEMORY, "Could not allocate the worker input row");
    struct vxvector_cpu_result result;
    if (choice.backend == VXVECTOR_BACKEND_GPU)
        gpu.reset_timings();
    reply.base.status = vxvector_run_v2(&request.options, &choice, &gpu.trainer, next_row, &reader,
                                        &result, &error);
    free(reader.values);
    if (choice.backend == VXVECTOR_BACKEND_GPU) {
        struct gpu_timings t;
        gpu.timings(&t);
        reply.gpu_allocation_seconds = t.allocation_seconds;
        reply.gpu_copy_in_seconds = t.copy_in_seconds;
        reply.gpu_training_seconds = t.training_seconds;
        reply.gpu_copy_out_seconds = t.copy_out_seconds;
    }
    if (reply.base.status != 0)
        return fail(version, &reply, reply.base.status, error.message);
    reply.base.read_seconds = result.read_seconds;
    reply.base.sample_seconds = result.sample_seconds;
    reply.base.clustering_seconds = result.clustering_seconds;
    int ok = send_reply(version, &reply) == 1 &&
             transfer(STDOUT_FILENO, result.nodes,
                      (size_t)reply.base.nodes * (size_t)reply.base.dims * sizeof(float), 1) == 1 &&
             transfer(STDOUT_FILENO, result.parents, (size_t)reply.base.nodes * sizeof(int32_t), 1) == 1;
    vxvector_free_cpu_result_v1(&result);
    return ok ? 0 : 1;
}
