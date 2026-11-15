#ifndef VXVECTOR_PROTOCOL_H
#define VXVECTOR_PROTOCOL_H

#include "vxvector_data_path.h"

/* Versioned local IPC. Backend and worker use the same host ABI and byte order.
 * No pointers cross the socket. A row is a u32 dimension followed by f32 values.
 * EOF ends input. A successful reply precedes nodes and parent IDs. The worker
 * first sends a planning reply, then a completion reply after reading all rows.
 */
#define VXVECTOR_WORKER_MAGIC UINT64_C(0x31524b5758544556)
struct vxvector_worker_request {
    uint64_t magic;
    struct vxvector_cpu_options options;
};

struct vxvector_worker_reply {
    int32_t status, dims;
    uint64_t nodes, samples, buffer_bytes;
    double read_seconds, sample_seconds, clustering_seconds;
    char message[256];
};

/* Version 2 adds backend selection (Step 4). The request names a mode from
 * enum vxvector_backend_mode. Each reply adds the backend that was chosen,
 * why, the GPU memory it needs, and the device name. The worker answers a
 * version 1 request as before, on the CPU. */
#define VXVECTOR_WORKER_MAGIC_V2 UINT64_C(0x32524b5758544556)
struct vxvector_worker_request_v2 {
    uint64_t magic;
    struct vxvector_cpu_options options;
    int32_t backend_mode, reserved;
};

struct vxvector_worker_reply_v2 {
    struct vxvector_worker_reply base;
    int32_t backend, reason;
    uint64_t gpu_bytes;
    char device[64];
    char reason_text[128];
};

_Static_assert(sizeof(struct vxvector_cpu_options) == 56, "CPU options ABI changed");
_Static_assert(sizeof(struct vxvector_worker_request_v2) == 72, "Worker request v2 ABI changed");
_Static_assert(sizeof(struct vxvector_worker_reply_v2) == 520, "Worker reply v2 ABI changed");
_Static_assert(sizeof(struct vxvector_worker_request) == 64, "Worker request ABI changed");
_Static_assert(sizeof(struct vxvector_worker_reply) == 312, "Worker reply ABI changed");
#endif
