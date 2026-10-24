#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/../.." && pwd)/Vx}"
profile="${VX_PROFILE:-debug}"
mkdir -p "$project_root/build"

(
  cd "$vx_root"
  source config.local
  "$vx_root/target/$profile/vxc" "$project_root/src/kmeans.vx" \
    --action emit-obj -o "$project_root/build/kmeans.o"

  if [[ "$(uname -s)" == "Darwin" ]]; then
    clang -dynamiclib "$project_root/build/kmeans.o" \
      "$vx_root/target/$profile/libvx_std_core.dylib" \
      -Wl,-rpath,"$vx_root/target/$profile" \
      -o "$project_root/build/libvextor_kmeans.dylib"
  else
    clang -shared "$project_root/build/kmeans.o" \
      "$vx_root/target/$profile/libvx_std_core.so" \
      -Wl,-rpath,"$vx_root/target/$profile" -lm \
      -o "$project_root/build/libvextor_kmeans.so"
  fi

  if [[ "${VX_CHECK_GPU:-0}" == "1" ]]; then
    "$vx_root/target/$profile/vxc" --machine "$vx_root/fleet/a100-40.vx" \
      "$project_root/src/gpu_lloyd.vx" --action emit-llvm \
      > "$project_root/build/gpu_lloyd.mlir"
  fi
)
