#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cxx="${CXX:-c++}"

if [[ ! -f "$cuda_home/include/cuda.h" ]]; then
  echo "CUDA driver headers not found under $cuda_home (set CUDA_HOME)." >&2
  exit 1
fi

VX_CHECK_GPU=1 "$project_root/scripts/build.sh"
python3 "$project_root/scripts/check_gpu_lloyd_manifest.py" \
  "$project_root/build/gpu_lloyd.json"

"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$project_root/scripts/gpu_lloyd_cuda_smoke.cpp" \
  -I"$cuda_home/include" -L"$cuda_home/lib64" -L"$cuda_home/lib64/stubs" \
  -Wl,-rpath,"$cuda_home/lib64" -lcuda \
  -o "$project_root/build/gpu_lloyd_cuda_smoke"

"$project_root/build/gpu_lloyd_cuda_smoke" "$project_root/build/gpu_lloyd.ptx"
