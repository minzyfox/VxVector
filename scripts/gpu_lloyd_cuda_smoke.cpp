#include "vxvector_gpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-5F;
}

} // namespace

int main() {
  constexpr int32_t rows = 6;
  constexpr int32_t dims = 2;
  constexpr int32_t k = 2;
  const std::array<float, rows * dims> points{
      0.0F, 0.0F, 2.0F, 0.0F, 1.0F, 1.0F,
      10.0F, 10.0F, 12.0F, 10.0F, 11.0F, 12.0F};
  std::array<float, k * dims> centers{};
  std::array<int32_t, rows> labels{};
  char device[256] = {};
  const int32_t probe = vxvector_gpu_probe_v1(device, sizeof(device));
  if (probe != VXVECTOR_GPU_OK) {
    std::fprintf(stderr, "No supported NVIDIA GPU is available for the CUDA smoke test.\n");
    return 1;
  }

  const int32_t trained = vxvector_gpu_restart_v1(
      points.data(), rows, dims, k, 2, 0, 0, 0, centers.data(), labels.data());
  if (trained != VXVECTOR_GPU_OK) {
    std::fprintf(stderr, "GPU restart failed with status %d.\n", trained);
    return 1;
  }
  const std::array<float, k * dims> expected_centers{
      1.0F, 1.0F / 3.0F, 11.0F, 32.0F / 3.0F};
  const std::array<int32_t, rows> expected_labels{0, 0, 0, 1, 1, 1};
  if (labels != expected_labels) {
    std::fprintf(stderr, "GPU restart returned unexpected labels.\n");
    return 1;
  }
  for (size_t i = 0; i < centers.size(); ++i) {
    if (!close(centers[i], expected_centers[i])) {
      std::fprintf(stderr, "GPU center %zu mismatch: got %.7g, expected %.7g\n",
                   i, centers[i], expected_centers[i]);
      return 1;
    }
  }

  labels.fill(-1);
  const int32_t assigned = vxvector_gpu_assign_v1(
      points.data(), rows, dims, centers.data(), k, 0, labels.data());
  if (assigned != VXVECTOR_GPU_OK || labels != expected_labels) {
    std::fprintf(stderr, "GPU assignment callback failed or returned wrong labels.\n");
    return 1;
  }

  std::printf("CUDA worker-library smoke test passed on %s.\n", device);
  return 0;
}
