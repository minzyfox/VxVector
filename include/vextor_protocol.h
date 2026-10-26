#ifndef VEXTOR_PROTOCOL_H
#define VEXTOR_PROTOCOL_H

#include "vextor_data_path.h"

/* Versioned local IPC. Backend and worker use the same host ABI and byte order.
 * No pointers cross the socket. A row is a u32 dimension followed by f32 values.
 * EOF ends input. A successful reply precedes nodes and parent IDs. The worker
 * first sends a planning reply, then a completion reply after reading all rows.
 */
#define VEXTOR_WORKER_MAGIC UINT64_C(0x31524b5758544556)
struct vextor_worker_request {
    uint64_t magic;
    struct vextor_cpu_options options;
};

struct vextor_worker_reply {
    int32_t status, dims;
    uint64_t nodes, samples, buffer_bytes;
    double read_seconds, sample_seconds, clustering_seconds;
    char message[256];
};

_Static_assert(sizeof(struct vextor_cpu_options) == 56, "CPU options ABI changed");
_Static_assert(sizeof(struct vextor_worker_request) == 64, "Worker request ABI changed");
_Static_assert(sizeof(struct vextor_worker_reply) == 312, "Worker reply ABI changed");
#endif
