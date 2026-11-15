/* What src/gpu_l2.c needs from a GPU: memory, copies, a wait, and names.
 *
 * src/gpu_kmeans.vx holds the kernels and the training loop for every GPU.
 * Each GPU library links src/gpu_l2.c with one file that implements these
 * functions: src/gpu_backend_metal.m for Apple GPUs, or src/gpu_backend_cuda.c
 * for NVIDIA GPUs. The Vx runtime of that GPU then uses the memory in place.
 */
#ifndef VXVECTOR_GPU_BACKEND_H
#define VXVECTOR_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

/* GPU memory of bytes bytes, or NULL when there is no usable GPU. */
void *vxvector_gpu_backend_alloc(size_t bytes);

/* Frees memory from vxvector_gpu_backend_alloc, after the GPU work that uses it. */
void vxvector_gpu_backend_free(void *memory);

/* Copies between host and GPU memory. Each returns 0, or nonzero on failure. */
int vxvector_gpu_backend_copy_in(void *device, const void *host, size_t bytes);
int vxvector_gpu_backend_copy_out(void *host, const void *device, size_t bytes);

/* Waits for every launch so far. Returns 0, or nonzero on failure. */
int vxvector_gpu_backend_wait(void);

/* Writes the device name to name. Returns 1, or 0 when there is no device. */
int vxvector_gpu_backend_device_name(char *name, size_t size);

/* The name of the GPU API, such as "Metal", for the version report. */
extern const char vxvector_gpu_backend_api[];

/* The default multiply-adds per launch. On macOS a long launch can be stopped,
 * so Metal keeps launches short. Splitting a launch changes no result. */
extern const int64_t vxvector_gpu_backend_launch_work;

#endif
