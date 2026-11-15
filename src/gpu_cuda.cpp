#include "vxvector_gpu.h"
#include "gpu_lloyd_ptx.h"

#include <cuda.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

#ifndef VXVECTOR_VX_REVISION
#define VXVECTOR_VX_REVISION "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Timings {
  double allocation_seconds = 0.0;
  double copy_in_seconds = 0.0;
  double training_seconds = 0.0;
  double copy_out_seconds = 0.0;
  uint64_t calls = 0;
};

thread_local Timings g_timings;
std::mutex g_cuda_mutex;

struct CudaState {
  bool ready = false;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUdevice device = 0;
  CUfunction assign = nullptr;
  CUfunction update = nullptr;
  CUfunction copy = nullptr;
  char device_name[256] = {};
};

CudaState &state() {
  static CudaState value;
  return value;
}

bool cuda_ok(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "[VxVector CUDA] %s failed: %s: %s\n", operation,
               name ? name : "unknown CUDA error",
               message ? message : "no description");
  return false;
}

bool ensure_cuda() {
  CudaState &cuda = state();
  if (cuda.ready) {
    return cuda_ok(cuCtxSetCurrent(cuda.context), "cuCtxSetCurrent");
  }

  int count = 0;
  if (!cuda_ok(cuInit(0), "cuInit") ||
      !cuda_ok(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
    return false;
  }
  if (!cuda_ok(cuDeviceGet(&cuda.device, 0), "cuDeviceGet")) {
    return false;
  }
  int major = 0;
  int minor = 0;
  if (!cuda_ok(cuDeviceComputeCapability(&major, &minor, cuda.device),
               "cuDeviceComputeCapability")) {
    return false;
  }
  if (major < 8) {
    std::fprintf(stderr,
                 "[VxVector CUDA] GPU compute capability %d.%d is below the "
                 "PTX target requirement 8.0\n",
                 major, minor);
    return false;
  }
  if (!cuda_ok(cuDeviceGetName(cuda.device_name, sizeof(cuda.device_name),
                               cuda.device),
               "cuDeviceGetName") ||
      !cuda_ok(cuDevicePrimaryCtxRetain(&cuda.context, cuda.device),
               "cuDevicePrimaryCtxRetain") ||
      !cuda_ok(cuCtxSetCurrent(cuda.context), "cuCtxSetCurrent") ||
      !cuda_ok(cuModuleLoadData(&cuda.module, vxvector_gpu_lloyd_ptx),
               "cuModuleLoadData") ||
      !cuda_ok(cuModuleGetFunction(&cuda.assign, cuda.module,
                                   "vxvector_gpu_assign_v1"),
               "load assignment kernel") ||
      !cuda_ok(cuModuleGetFunction(&cuda.update, cuda.module,
                                   "vxvector_gpu_update_v1"),
               "load update kernel") ||
      !cuda_ok(cuModuleGetFunction(&cuda.copy, cuda.module,
                                   "vxvector_gpu_copy_v1"),
               "load copy kernel")) {
    return false;
  }
  cuda.ready = true;
  return true;
}

struct DeviceBuffer {
  CUdeviceptr pointer = 0;

  bool allocate(size_t bytes) {
    return cuda_ok(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
  }

  ~DeviceBuffer() {
    if (pointer != 0) {
      cuMemFree(pointer);
    }
  }
};

struct Tensor2DArgs {
  uint64_t allocated;
  uint64_t aligned;
  uint64_t offset;
  uint64_t rows;
  uint64_t columns;
  uint64_t row_stride;
  uint64_t column_stride;
};

struct Tensor1DArgs {
  uint64_t allocated;
  uint64_t aligned;
  uint64_t offset;
  uint64_t size;
  uint64_t stride;
};

Tensor2DArgs tensor2d(CUdeviceptr pointer, int32_t rows, int32_t columns) {
  const uint64_t address = static_cast<uint64_t>(pointer);
  return {address, address, 0, static_cast<uint64_t>(rows),
          static_cast<uint64_t>(columns), static_cast<uint64_t>(columns), 1};
}

Tensor1DArgs tensor1d(CUdeviceptr pointer, int32_t size) {
  const uint64_t address = static_cast<uint64_t>(pointer);
  return {address, address, 0, static_cast<uint64_t>(size), 1};
}

bool launch(CUfunction function, unsigned grid, void **arguments,
            const char *operation) {
  if (!cuda_ok(cuLaunchKernel(function, grid, 1, 1, 1, 1, 1, 0, nullptr,
                              arguments, nullptr),
               operation)) {
    return false;
  }
  return cuda_ok(cuCtxSynchronize(), "cuCtxSynchronize");
}

bool launch_assign(CUdeviceptr points, CUdeviceptr centers, CUdeviceptr labels,
                   int32_t rows, int32_t dims, int32_t k, int32_t metric) {
  Tensor2DArgs point_args = tensor2d(points, rows, dims);
  Tensor2DArgs center_args = tensor2d(centers, k, dims);
  Tensor1DArgs label_args = tensor1d(labels, rows);
  void *arguments[] = {
      &point_args.allocated, &point_args.aligned, &point_args.offset,
      &point_args.rows,      &point_args.columns, &point_args.row_stride,
      &point_args.column_stride, &rows, &dims, &k, &metric,
      &center_args.allocated, &center_args.aligned, &center_args.offset,
      &center_args.rows, &center_args.columns, &center_args.row_stride,
      &center_args.column_stride, &label_args.allocated, &label_args.aligned,
      &label_args.offset, &label_args.size, &label_args.stride};
  return launch(state().assign, static_cast<unsigned>(rows), arguments,
                "launch assignment kernel");
}

bool launch_update(CUdeviceptr points, CUdeviceptr labels,
                   CUdeviceptr updated, CUdeviceptr errors, int32_t rows,
                   int32_t dims, int32_t k, int32_t iteration,
                   int32_t spherical) {
  Tensor2DArgs point_args = tensor2d(points, rows, dims);
  Tensor1DArgs label_args = tensor1d(labels, rows);
  Tensor2DArgs updated_args = tensor2d(updated, k, dims);
  Tensor1DArgs error_args = tensor1d(errors, k);
  void *arguments[] = {
      &point_args.allocated, &point_args.aligned, &point_args.offset,
      &point_args.rows, &point_args.columns, &point_args.row_stride,
      &point_args.column_stride, &rows, &dims, &k, &iteration, &spherical,
      &label_args.allocated, &label_args.aligned, &label_args.offset,
      &label_args.size, &label_args.stride, &updated_args.allocated,
      &updated_args.aligned, &updated_args.offset, &updated_args.rows,
      &updated_args.columns, &updated_args.row_stride,
      &updated_args.column_stride, &error_args.allocated, &error_args.aligned,
      &error_args.offset, &error_args.size, &error_args.stride};
  return launch(state().update, static_cast<unsigned>(k), arguments,
                "launch centroid update kernel");
}

bool launch_copy(CUdeviceptr centers, CUdeviceptr updated, int32_t k,
                 int32_t dims) {
  Tensor2DArgs center_args = tensor2d(centers, k, dims);
  Tensor2DArgs updated_args = tensor2d(updated, k, dims);
  void *arguments[] = {
      &k, &dims, &center_args.allocated, &center_args.aligned,
      &center_args.offset, &center_args.rows, &center_args.columns,
      &center_args.row_stride, &center_args.column_stride,
      &updated_args.allocated, &updated_args.aligned, &updated_args.offset,
      &updated_args.rows, &updated_args.columns, &updated_args.row_stride,
      &updated_args.column_stride};
  return launch(state().copy, static_cast<unsigned>(k), arguments,
                "launch centroid copy kernel");
}

bool valid_shape(const float *points, int32_t rows, int32_t dims, int32_t k,
                 int32_t metric) {
  return points && rows > 0 && dims > 0 && k > 0 && k <= rows &&
         metric >= 0 && metric <= 2 &&
         static_cast<int64_t>(rows) * dims <
             std::numeric_limits<int32_t>::max();
}

void normalize(float *center, int32_t dims) {
  double squared = 0.0;
  for (int32_t d = 0; d < dims; ++d) {
    const double value = static_cast<double>(center[d]);
    squared += value * value;
  }
  if (squared > 0.0) {
    const double norm = std::sqrt(squared);
    for (int32_t d = 0; d < dims; ++d) {
      center[d] = static_cast<float>(static_cast<double>(center[d]) / norm);
    }
  }
}

int32_t train(const float *points, int32_t rows, int32_t dims, int32_t k,
              int32_t iterations, int32_t metric, int32_t spherical,
              int32_t restart, float *centers, int32_t *labels) {
  if (!valid_shape(points, rows, dims, k, metric) || iterations <= 0 ||
      (spherical != 0 && spherical != 1) || restart < 0 || !centers ||
      !labels) {
    return VXVECTOR_GPU_INVALID;
  }
  if (!ensure_cuda()) {
    return VXVECTOR_GPU_UNAVAILABLE;
  }

  const size_t values = static_cast<size_t>(rows) * dims;
  const size_t point_bytes = values * sizeof(float);
  const size_t center_bytes = static_cast<size_t>(k) * dims * sizeof(float);
  std::vector<float> initial(static_cast<size_t>(k) * dims);
  std::vector<int32_t> error_flags(static_cast<size_t>(k));
  for (int32_t center = 0; center < k; ++center) {
    const int64_t spread = static_cast<int64_t>(center) * rows / k;
    const int64_t shift = static_cast<int64_t>(restart) * 104729;
    const int32_t position = static_cast<int32_t>((spread + shift) % rows);
    std::memcpy(initial.data() + static_cast<size_t>(center) * dims,
                points + static_cast<size_t>(position) * dims,
                static_cast<size_t>(dims) * sizeof(float));
    if (spherical) {
      normalize(initial.data() + static_cast<size_t>(center) * dims, dims);
    }
  }

  const auto allocation_start = Clock::now();
  DeviceBuffer device_points;
  DeviceBuffer device_centers;
  DeviceBuffer device_updated;
  DeviceBuffer device_labels;
  DeviceBuffer device_errors;
  if (!device_points.allocate(point_bytes) ||
      !device_centers.allocate(center_bytes) ||
      !device_updated.allocate(center_bytes) ||
      !device_labels.allocate(static_cast<size_t>(rows) * sizeof(int32_t)) ||
      !device_errors.allocate(static_cast<size_t>(k) * sizeof(int32_t))) {
    g_timings.allocation_seconds +=
        std::chrono::duration<double>(Clock::now() - allocation_start).count();
    return VXVECTOR_GPU_UNAVAILABLE;
  }
  g_timings.allocation_seconds +=
      std::chrono::duration<double>(Clock::now() - allocation_start).count();

  auto copy_start = Clock::now();
  bool ok = cuda_ok(cuMemcpyHtoD(device_points.pointer, points, point_bytes),
                    "copy points to device") &&
            cuda_ok(cuMemcpyHtoD(device_centers.pointer, initial.data(),
                                 center_bytes),
                    "copy initial centers to device");
  g_timings.copy_in_seconds +=
      std::chrono::duration<double>(Clock::now() - copy_start).count();
  if (!ok) {
    return VXVECTOR_GPU_UNAVAILABLE;
  }

  const auto training_start = Clock::now();
  for (int32_t iteration = 0; iteration < iterations; ++iteration) {
    if (!launch_assign(device_points.pointer, device_centers.pointer,
                       device_labels.pointer, rows, dims, k, metric) ||
        !launch_update(device_points.pointer, device_labels.pointer,
                       device_updated.pointer, device_errors.pointer, rows,
                       dims, k, iteration, spherical) ||
        !cuda_ok(cuMemcpyDtoH(error_flags.data(), device_errors.pointer,
                              static_cast<size_t>(k) * sizeof(int32_t)),
                 "copy centroid update status")) {
      g_timings.training_seconds +=
          std::chrono::duration<double>(Clock::now() - training_start).count();
      return VXVECTOR_GPU_UNAVAILABLE;
    }
    for (int32_t error : error_flags) {
      if (error != 0) {
        g_timings.training_seconds +=
            std::chrono::duration<double>(Clock::now() - training_start).count();
        return VXVECTOR_GPU_OVERFLOW;
      }
    }
    if (!launch_copy(device_centers.pointer, device_updated.pointer, k, dims)) {
      g_timings.training_seconds +=
          std::chrono::duration<double>(Clock::now() - training_start).count();
      return VXVECTOR_GPU_UNAVAILABLE;
    }
  }
  if (!launch_assign(device_points.pointer, device_centers.pointer,
                     device_labels.pointer, rows, dims, k, metric)) {
    g_timings.training_seconds +=
        std::chrono::duration<double>(Clock::now() - training_start).count();
    return VXVECTOR_GPU_UNAVAILABLE;
  }
  g_timings.training_seconds +=
      std::chrono::duration<double>(Clock::now() - training_start).count();

  copy_start = Clock::now();
  ok = cuda_ok(cuMemcpyDtoH(centers, device_centers.pointer, center_bytes),
               "copy trained centers to host") &&
       cuda_ok(cuMemcpyDtoH(labels, device_labels.pointer,
                            static_cast<size_t>(rows) * sizeof(int32_t)),
               "copy trained labels to host");
  g_timings.copy_out_seconds +=
      std::chrono::duration<double>(Clock::now() - copy_start).count();
  if (!ok) {
    return VXVECTOR_GPU_UNAVAILABLE;
  }
  for (size_t i = 0; i < static_cast<size_t>(k) * dims; ++i) {
    if (!std::isfinite(centers[i])) {
      return VXVECTOR_GPU_OVERFLOW;
    }
  }
  return VXVECTOR_GPU_OK;
}

int32_t callback_status(int32_t status) {
  if (status == VXVECTOR_GPU_OK) {
    return 0;
  }
  if (status == VXVECTOR_GPU_INVALID || status == VXVECTOR_GPU_OVERFLOW) {
    return 1;
  }
  return 2;
}

} // namespace

extern "C" {

int32_t vxvector_gpu_probe_v1(char *device, size_t size) {
  if (device && size) {
    device[0] = '\0';
  }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  if (!ensure_cuda()) {
    return VXVECTOR_GPU_UNAVAILABLE;
  }
  if (device && size) {
    std::snprintf(device, size, "%s", state().device_name);
  }
  return VXVECTOR_GPU_OK;
}

int32_t vxvector_gpu_restart_v1(const float *points, int32_t rows, int32_t dims,
                                int32_t leaf_count, int32_t iterations,
                                int32_t metric, int32_t spherical,
                                int32_t restart, float *centers,
                                int32_t *labels) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ++g_timings.calls;
  try {
    return callback_status(train(points, rows, dims, leaf_count, iterations,
                                 metric, spherical, restart, centers, labels));
  } catch (const std::bad_alloc &error) {
    std::fprintf(stderr, "[VxVector CUDA] host allocation failed: %s\n",
                 error.what());
    return 2;
  }
}

int32_t vxvector_gpu_assign_v1(const float *points, int32_t rows, int32_t dims,
                               const float *centers, int32_t k, int32_t metric,
                               int32_t *labels) {
  if (!valid_shape(points, rows, dims, k, metric) || !centers || !labels) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ++g_timings.calls;
  if (!ensure_cuda()) {
    return 2;
  }
  const size_t point_bytes =
      static_cast<size_t>(rows) * dims * sizeof(float);
  const size_t center_bytes = static_cast<size_t>(k) * dims * sizeof(float);
  const auto allocation_start = Clock::now();
  DeviceBuffer device_points;
  DeviceBuffer device_centers;
  DeviceBuffer device_labels;
  if (!device_points.allocate(point_bytes) ||
      !device_centers.allocate(center_bytes) ||
      !device_labels.allocate(static_cast<size_t>(rows) * sizeof(int32_t))) {
    g_timings.allocation_seconds +=
        std::chrono::duration<double>(Clock::now() - allocation_start).count();
    return 2;
  }
  g_timings.allocation_seconds +=
        std::chrono::duration<double>(Clock::now() - allocation_start).count();
  auto copy_start = Clock::now();
  bool ok = cuda_ok(cuMemcpyHtoD(device_points.pointer, points, point_bytes),
                    "copy points to device") &&
            cuda_ok(cuMemcpyHtoD(device_centers.pointer, centers, center_bytes),
                    "copy centers to device");
  g_timings.copy_in_seconds +=
      std::chrono::duration<double>(Clock::now() - copy_start).count();
  if (!ok) {
    return 2;
  }
  const auto training_start = Clock::now();
  ok = launch_assign(device_points.pointer, device_centers.pointer,
                     device_labels.pointer, rows, dims, k, metric);
  g_timings.training_seconds +=
      std::chrono::duration<double>(Clock::now() - training_start).count();
  if (!ok) {
    return 2;
  }
  copy_start = Clock::now();
  ok = cuda_ok(cuMemcpyDtoH(labels, device_labels.pointer,
                            static_cast<size_t>(rows) * sizeof(int32_t)),
               "copy labels to host");
  g_timings.copy_out_seconds +=
      std::chrono::duration<double>(Clock::now() - copy_start).count();
  return ok ? 0 : 2;
}

void vxvector_gpu_reset_timings_v1(void) { g_timings = {}; }

void vxvector_gpu_timings_v1(struct vxvector_gpu_timings *out) {
  if (!out) {
    return;
  }
  out->allocation_seconds = g_timings.allocation_seconds;
  out->copy_in_seconds = g_timings.copy_in_seconds;
  out->training_seconds = g_timings.training_seconds;
  out->copy_out_seconds = g_timings.copy_out_seconds;
  out->calls = g_timings.calls;
}

const char *vxvector_gpu_version_v1(void) {
  return "CUDA PTX, Vx " VXVECTOR_VX_REVISION;
}

} // extern "C"
