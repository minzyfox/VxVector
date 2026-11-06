#include "vxvector_protocol.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    struct vxvector_worker_request request;
    struct vxvector_worker_reply reply = {0};
    struct vxvector_data_error error = {0};
    struct vxvector_cpu_plan plan;
    if (transfer(STDIN_FILENO, &request, sizeof(request), 0) != 1 ||
        request.magic != VXVECTOR_WORKER_MAGIC) {
        reply.status = VXVECTOR_DATA_INVALID;
        snprintf(reply.message, sizeof(reply.message), "Invalid or truncated worker request");
        transfer(STDOUT_FILENO, &reply, sizeof(reply), 1);
        return 1;
    }
    uint64_t row_bytes = request.options.dims > 0 ? (uint64_t)request.options.dims * sizeof(float) : 0;
    if (request.options.memory_budget_bytes <= row_bytes) {
        reply.status = VXVECTOR_DATA_BUDGET;
        snprintf(reply.message, sizeof(reply.message), "Memory budget cannot hold the worker input row");
    } else {
        request.options.memory_budget_bytes -= row_bytes;
        reply.status = vxvector_plan_cpu_v1(&request.options, &plan, &error);
        snprintf(reply.message, sizeof(reply.message), "%s", error.message);
    }
    if (reply.status != 0) {
        transfer(STDOUT_FILENO, &reply, sizeof(reply), 1);
        return 1;
    }
    reply.dims = request.options.dims;
    reply.nodes = plan.node_count;
    reply.samples = plan.sample_count;
    reply.buffer_bytes = plan.buffer_bytes + row_bytes;
    if (transfer(STDOUT_FILENO, &reply, sizeof(reply), 1) != 1)
        return 1;
    struct row_reader reader = {(uint32_t)request.options.dims, malloc((size_t)row_bytes)};
    if (!reader.values) {
        reply.status = VXVECTOR_DATA_MEMORY;
        snprintf(reply.message, sizeof(reply.message), "Could not allocate the worker input row");
        transfer(STDOUT_FILENO, &reply, sizeof(reply), 1);
        return 1;
    }
    struct vxvector_cpu_result result;
    reply.status = vxvector_run_cpu_v1(&request.options, next_row, &reader, &result, &error);
    free(reader.values);
    snprintf(reply.message, sizeof(reply.message), "%s", error.message);
    if (reply.status != 0) {
        transfer(STDOUT_FILENO, &reply, sizeof(reply), 1);
        return 1;
    }
    reply.read_seconds = result.read_seconds;
    reply.sample_seconds = result.sample_seconds;
    reply.clustering_seconds = result.clustering_seconds;
    int ok = transfer(STDOUT_FILENO, &reply, sizeof(reply), 1) == 1 &&
             transfer(STDOUT_FILENO, result.nodes,
                      (size_t)reply.nodes * (size_t)reply.dims * sizeof(float), 1) == 1 &&
             transfer(STDOUT_FILENO, result.parents, (size_t)reply.nodes * sizeof(int32_t), 1) == 1;
    vxvector_free_cpu_result_v1(&result);
    return ok ? 0 : 1;
}
