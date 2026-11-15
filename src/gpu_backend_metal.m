/* src/gpu_backend.h for Apple GPUs, with the Metal runtime of Vx.
 *
 * vx_metal_alloc returns memory that the CPU and the GPU share, and Metal uses
 * it in place. The allocation also proves that a Metal device is present and
 * enabled, because the Vx runtime would otherwise run the kernels on the CPU
 * without saying so. vx_metal_free waits for the GPU work that uses the memory
 * before it frees it. The copies are plain memcpy, because the memory is shared.
 *
 * Objective-C only for the device name, because Metal's device API is.
 */
#import <Metal/Metal.h>

#include "gpu_backend.h"

#include <string.h>

void *vx_metal_alloc(size_t bytes);
int vx_metal_free(void *memory);
void vx_metal_wait(void);

const char vxvector_gpu_backend_api[] = "Metal";

/* At most about 2 ms of GPU time per launch on an M2 Pro. */
const int64_t vxvector_gpu_backend_launch_work = (int64_t)1 << 26;

void *vxvector_gpu_backend_alloc(size_t bytes) { return vx_metal_alloc(bytes); }

void vxvector_gpu_backend_free(void *memory)
{
    if (memory)
        vx_metal_free(memory);
}

int vxvector_gpu_backend_copy_in(void *device, const void *host, size_t bytes)
{
    memcpy(device, host, bytes);
    return 0;
}

int vxvector_gpu_backend_copy_out(void *host, const void *device, size_t bytes)
{
    memcpy(host, device, bytes);
    return 0;
}

int vxvector_gpu_backend_wait(void)
{
    vx_metal_wait();
    return 0;
}

int vxvector_gpu_backend_device_name(char *name, size_t size)
{
    if (!name || size == 0)
        return 0;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            name[0] = '\0';
            return 0;
        }
        strlcpy(name, device.name.UTF8String, size);
        return 1;
    }
}
