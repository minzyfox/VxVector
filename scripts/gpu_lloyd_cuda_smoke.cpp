#include <cuda.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "%s failed: %s: %s\n", operation,
                 name ? name : "unknown CUDA error",
                 message ? message : "no description");
    std::exit(EXIT_FAILURE);
  }
}

struct Tensor2D {
  uint64_t pointer;
  uint64_t aligned;
  uint64_t offset = 0;
  uint64_t rows;
  uint64_t columns;
  uint64_t row_stride;
  uint64_t column_stride = 1;
};

struct Tensor1D {
  uint64_t pointer;
  uint64_t aligned;
  uint64_t offset = 0;
  uint64_t size;
  uint64_t stride = 1;
};

void append(std::vector<void *> &args, Tensor2D &tensor) {
  args.insert(args.end(), {&tensor.pointer, &tensor.aligned, &tensor.offset,
                           &tensor.rows, &tensor.columns, &tensor.row_stride,
                           &tensor.column_stride});
}

void append(std::vector<void *> &args, Tensor1D &tensor) {
  args.insert(args.end(), {&tensor.pointer, &tensor.aligned, &tensor.offset,
                           &tensor.size, &tensor.stride});
}

CUdeviceptr allocate(size_t bytes) {
  CUdeviceptr pointer = 0;
  check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
  return pointer;
}

void launch(CUfunction function, unsigned grid_x, void **args,
            const char *name) {
  check(cuLaunchKernel(function, grid_x, 1, 1, 1, 1, 1, 0, nullptr, args,
                       nullptr),
        name);
  check(cuCtxSynchronize(), "cuCtxSynchronize");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <gpu_lloyd.ptx>\n", argv[0]);
    return EXIT_FAILURE;
  }

  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  check(cuCtxCreate(&context, 0, device), "cuCtxCreate");

  CUmodule module;
  check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
  CUfunction assign;
  CUfunction update;
  CUfunction copy;
  check(cuModuleGetFunction(&assign, module, "vxvector_gpu_assign_v1"),
        "cuModuleGetFunction(assign)");
  check(cuModuleGetFunction(&update, module, "vxvector_gpu_update_v1"),
        "cuModuleGetFunction(update)");
  check(cuModuleGetFunction(&copy, module, "vxvector_gpu_copy_v1"),
        "cuModuleGetFunction(copy)");

  constexpr int32_t row_count = 6;
  constexpr int32_t dimension_count = 2;
  constexpr int32_t cluster_count = 2;
  int32_t rows = row_count;
  int32_t dims = dimension_count;
  int32_t k = cluster_count;
  int32_t metric_l2 = 0;
  int32_t spherical = 0;
  constexpr size_t point_count = row_count * dimension_count;
  constexpr size_t center_count = cluster_count * dimension_count;

  const std::array<float, point_count> host_points{
      0.0F, 0.0F, 2.0F, 0.0F, 1.0F, 1.0F,
      10.0F, 10.0F, 12.0F, 10.0F, 11.0F, 12.0F};
  const std::array<float, center_count> host_centers{0.0F, 0.0F, 12.0F,
                                                      12.0F};

  const CUdeviceptr points_device = allocate(sizeof(host_points));
  const CUdeviceptr centers_device = allocate(sizeof(host_centers));
  const CUdeviceptr updated_device = allocate(sizeof(host_centers));
  const CUdeviceptr labels_device = allocate(rows * sizeof(int32_t));
  check(cuMemcpyHtoD(points_device, host_points.data(), sizeof(host_points)),
        "copy points to device");
  check(cuMemcpyHtoD(centers_device, host_centers.data(),
                     sizeof(host_centers)),
        "copy centers to device");

  Tensor2D points{static_cast<uint64_t>(points_device),
                  static_cast<uint64_t>(points_device), 0, rows, dims, dims};
  Tensor2D centers{static_cast<uint64_t>(centers_device),
                   static_cast<uint64_t>(centers_device), 0, k, dims, dims};
  Tensor2D updated{static_cast<uint64_t>(updated_device),
                   static_cast<uint64_t>(updated_device), 0, k, dims, dims};
  Tensor1D labels{static_cast<uint64_t>(labels_device),
                  static_cast<uint64_t>(labels_device), 0, rows};

  std::vector<void *> assign_args;
  append(assign_args, points);
  assign_args.insert(assign_args.end(),
                     {&rows, &dims, &k, &metric_l2});
  append(assign_args, centers);
  append(assign_args, labels);
  launch(assign, rows, assign_args.data(), "launch assignment kernel");

  std::array<int32_t, row_count> host_labels{};
  check(cuMemcpyDtoH(host_labels.data(), labels_device, sizeof(host_labels)),
        "copy labels to host");
  const std::array<int32_t, row_count> expected_labels{0, 0, 0, 1, 1, 1};
  if (host_labels != expected_labels) {
    std::fprintf(stderr, "assignment mismatch:");
    for (int32_t label : host_labels) {
      std::fprintf(stderr, " %d", label);
    }
    std::fprintf(stderr, "\n");
    return EXIT_FAILURE;
  }

  std::vector<void *> update_args;
  append(update_args, points);
  update_args.insert(update_args.end(),
                     {&rows, &dims, &k, &spherical});
  append(update_args, centers);
  append(update_args, labels);
  append(update_args, updated);
  launch(update, k, update_args.data(), "launch update kernel");

  std::vector<void *> copy_args;
  copy_args.insert(copy_args.end(), {&k, &dims});
  append(copy_args, centers);
  append(copy_args, updated);
  launch(copy, k, copy_args.data(), "launch copy kernel");

  std::array<float, center_count> result{};
  check(cuMemcpyDtoH(result.data(), centers_device, sizeof(result)),
        "copy centers to host");
  constexpr std::array<float, center_count> expected_centers{
      1.0F, 1.0F / 3.0F, 11.0F, 32.0F / 3.0F};
  for (size_t i = 0; i < result.size(); ++i) {
    if (std::fabs(result[i] - expected_centers[i]) > 1e-4F) {
      std::fprintf(stderr, "center[%zu] mismatch: got %.7g, expected %.7g\n",
                   i, result[i], expected_centers[i]);
      return EXIT_FAILURE;
    }
  }

  std::puts("NVIDIA PTX smoke test passed (assignment, update, copy).");
  return EXIT_SUCCESS;
}
