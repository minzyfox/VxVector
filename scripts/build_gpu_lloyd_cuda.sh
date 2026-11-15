#!/usr/bin/env bash
# The CUDA Driver API prototype (src/gpu_cuda.cpp, src/gpu_lloyd.vx). It is
# replaced by scripts/build_gpu_cuda.sh and goes once that build is validated
# on an NVIDIA GPU.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/.." && pwd)/Vx}"
profile="${VX_PROFILE:-debug}"
revision="$(cat "$project_root/vx-revision.txt")"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cxx="${CXX:-c++}"
out="$project_root/build/linux"

if [[ "$(uname -s)" != Linux ]]; then
  echo "The CUDA worker library build requires Linux." >&2
  exit 1
fi
if [[ ! -f "$cuda_home/include/cuda.h" || ! -f "$cuda_home/lib64/stubs/libcuda.so" ]]; then
  echo "CUDA driver headers and stubs not found under $cuda_home (set CUDA_HOME)." >&2
  exit 1
fi
if [[ "$(git -C "$vx_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "VX_ROOT must be checked out at the commit in vx-revision.txt: $revision" >&2
  exit 1
fi
if ! git -C "$vx_root" diff --quiet HEAD --; then
  echo "Build Vx from a clean checkout so its recorded revision identifies its sources." >&2
  exit 1
fi

compiler="$vx_root/target/$profile/vxc"
if [[ ! -x "$compiler" || "$("$compiler" --version)" != *"(${revision:0:8})"* ]]; then
  echo "Build Vx at vx-revision.txt with VX_VERSION_WITH_COMMIT=1 first." >&2
  exit 1
fi

mkdir -p "$out"
"$compiler" --machine "$vx_root/fleet/a100-40.vx" \
  "$project_root/src/gpu_lloyd.vx" --action emit-ptx \
  -o "$out/gpu_lloyd.ptx" > "$out/gpu_lloyd-build.log" 2>&1 || {
    cat "$out/gpu_lloyd-build.log" >&2
    exit 1
  }
python3 "$project_root/scripts/check_gpu_lloyd_manifest.py" \
  "$out/gpu_lloyd.json"
python3 "$project_root/scripts/embed_ptx.py" \
  "$out/gpu_lloyd.ptx" "$out/gpu_lloyd_ptx.h"

"$cxx" -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared \
  -I"$project_root/include" -I"$out" -I"$cuda_home/include" \
  -DVXVECTOR_VX_REVISION="\"${revision:0:8}\"" \
  "$project_root/src/gpu_cuda.cpp" \
  -L"$cuda_home/lib64/stubs" -Wl,-z,defs \
  -Wl,-soname,libvxvector_gpu_lloyd.so -lcuda \
  -o "$out/libvxvector_gpu_lloyd.so"

symbols="$(nm -D --defined-only "$out/libvxvector_gpu_lloyd.so")"
for symbol in vxvector_gpu_probe_v1 vxvector_gpu_restart_v1 \
  vxvector_gpu_assign_v1 vxvector_gpu_reset_timings_v1 \
  vxvector_gpu_timings_v1 vxvector_gpu_version_v1; do
  if ! grep -Eq " $symbol$" <<<"$symbols"; then
    echo "The CUDA library is missing required worker symbol $symbol." >&2
    exit 1
  fi
done

printf 'Built %s\n' "$out/libvxvector_gpu_lloyd.so"
