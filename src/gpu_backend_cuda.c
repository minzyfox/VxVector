/* src/gpu_backend.h for NVIDIA GPUs, with the CUDA runtime of Vx.
 *
 * Memory is device memory from cudaMalloc. The Vx CUDA runtime asks CUDA
 * whether each kernel argument is device memory, and uses device memory in
 * place, so every tensor stays on the GPU for the whole run.
 *
 * The Vx runtime launches on a stream of its own choosing. So that a copy
 * never overtakes a launch, each copy first waits for the whole device. The
 * waits cost little: one restart copies the samples in once, the k centers out
 * and back once per iteration only for spherical centers, and the results out
 * once.
 */
#include "gpu_backend.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>

const char vxvector_gpu_backend_api[] = "CUDA";

/* No watchdog stops a long launch on Linux, so launches are not split.
 * Splitting would change no result. */
const int64_t vxvector_gpu_backend_launch_work = (int64_t)1 << 62;

static int failed(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return 0;
    fprintf(stderr, "[VxVector CUDA] %s failed: %s: %s\n", operation, cudaGetErrorName(error),
            cudaGetErrorString(error));
    return 1;
}

void *vxvector_gpu_backend_alloc(size_t bytes)
{
    void *memory = NULL;
    /* cudaMalloc fails without a usable device, which makes the probe fail. */
    if (failed(cudaMalloc(&memory, bytes ? bytes : 1), "cudaMalloc"))
        return NULL;
    return memory;
}

void vxvector_gpu_backend_free(void *memory)
{
    /* cudaFree waits for the work on the device before it frees. */
    if (memory)
        failed(cudaFree(memory), "cudaFree");
}

int vxvector_gpu_backend_wait(void)
{
    return failed(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

int vxvector_gpu_backend_copy_in(void *device, const void *host, size_t bytes)
{
    return vxvector_gpu_backend_wait() ||
           failed(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy to device");
}

int vxvector_gpu_backend_copy_out(void *host, const void *device, size_t bytes)
{
    return vxvector_gpu_backend_wait() ||
           failed(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy to host");
}

int vxvector_gpu_backend_device_name(char *name, size_t size)
{
    if (!name || size == 0)
        return 0;
    name[0] = '\0';
    int device = 0;
    struct cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return 0;
    snprintf(name, size, "%s", properties.name);
    return 1;
}
