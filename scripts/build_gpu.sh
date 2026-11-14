#!/usr/bin/env bash
# Builds build/gpu/libvxvector_gpu.dylib for Step 3: the Apple GPU kernels of
# src/gpu_kmeans.vx and the C wrapper src/gpu_l2.c. macOS on Apple silicon only.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/../.." && pwd)/Vx}"
profile="${VX_PROFILE:-debug}"
revision="$(cat "$project_root/vx-revision.txt")"
machine="${VXVECTOR_GPU_MACHINE:-$vx_root/fleet/m4-uma.vx}"
out="$project_root/build/gpu"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "The Metal build needs macOS on arm64. Under Rosetta, run it with arch -arm64." >&2
  exit 1
fi
if [[ "$(git -C "$vx_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "VX_ROOT must be checked out at the commit in vx-revision.txt: $revision" >&2
  exit 1
fi
compiler="$vx_root/target/$profile/vxc"
if [[ "$("$compiler" --version)" != *"(${revision:0:8})"* ]]; then
  echo "Rebuild Vx with VX_VERSION_WITH_COMMIT=1 before building VxVector." >&2
  exit 1
fi
# The Metal runtime that this compiler was built with. vxc records its path.
dispatch="$(grep -a -o '/[^"[:cntrl:]]*libvx_npu_dispatch\.dylib' "$compiler" | sort -u | head -1)"
if [[ -z "$dispatch" || ! -f "$dispatch" ]]; then
  echo "Cannot find the libvx_npu_dispatch.dylib of $compiler." >&2
  exit 1
fi
mkdir -p "$out"

if ! "$compiler" --host default --machine "$machine" "$project_root/src/gpu_kmeans.vx" \
  --action emit-obj -o "$out/gpu_kmeans.o" > "$out/gpu_kmeans-build.log" 2>&1; then
  cat "$out/gpu_kmeans-build.log" >&2
  exit 1
fi
"$compiler" --host default --machine "$machine" "$project_root/src/gpu_kmeans.vx" \
  --action emit-msl -o "$out/gpu_kmeans.metal" > /dev/null 2>&1 || true
clang -std=c11 -O2 -Wall -Wextra -Werror -fPIC -I "$project_root/include" \
  -c "$project_root/src/gpu_l2.c" -o "$out/gpu_l2.o"
clang -shared "$out/gpu_kmeans.o" "$out/gpu_l2.o" "$dispatch" \
  -Wl,-rpath,"$(dirname "$dispatch")" -o "$out/libvxvector_gpu.dylib"
printf '%s\n' "$dispatch" > "$out/dispatch-path.txt"
