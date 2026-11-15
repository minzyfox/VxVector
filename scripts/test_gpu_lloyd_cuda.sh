#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/.." && pwd)/Vx}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cxx="${CXX:-c++}"

VX_ROOT="$vx_root" CUDA_HOME="$cuda_home" \
  "$project_root/scripts/build_gpu_lloyd_cuda.sh"

"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$project_root/include" \
  "$project_root/scripts/gpu_lloyd_cuda_smoke.cpp" \
  -L"$project_root/build/linux" -Wl,-rpath,"$project_root/build/linux" \
  -lvxvector_gpu_lloyd \
  -o "$project_root/build/gpu_lloyd_cuda_smoke"

"$project_root/build/gpu_lloyd_cuda_smoke"
