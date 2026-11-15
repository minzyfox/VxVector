#!/usr/bin/env bash
# Builds build/linux/libvxvector_gpu.so: the GPU kernels of src/gpu_kmeans.vx
# for NVIDIA GPUs (A100 PTX inside a host object), the C wrapper src/gpu_l2.c,
# and its CUDA backend src/gpu_backend_cuda.c. Linux only. It needs the CUDA
# toolkit, but no GPU.
#
# The Vx compiler must itself be built where the toolkit is installed. Only
# then does it build libvx_cuda_dispatch.so, which launches the kernels on the
# GPU. Without the toolkit, Vx builds a CPU dispatch library instead, which
# would run the kernels on the CPU without saying so, so this script refuses it.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/.." && pwd)/Vx}"
profile="${VX_PROFILE:-debug}"
revision="$(cat "$project_root/vx-revision.txt")"
machine="${VXVECTOR_GPU_MACHINE:-$vx_root/fleet/a100-40.vx}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cc="${CC:-cc}"
out="$project_root/build/linux"

if [[ "$(uname -s)" != Linux ]]; then
  echo "The CUDA GPU library build requires Linux." >&2
  exit 1
fi
if [[ ! -f "$cuda_home/include/cuda_runtime_api.h" ]]; then
  echo "CUDA toolkit headers not found under $cuda_home (set CUDA_HOME)." >&2
  exit 1
fi
cuda_lib="$cuda_home/lib64"
[[ -d "$cuda_lib" ]] || cuda_lib="$cuda_home/lib"
if [[ "$(git -C "$vx_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "VX_ROOT must be checked out at the commit in vx-revision.txt: $revision" >&2
  exit 1
fi
compiler="$vx_root/target/$profile/vxc"
if [[ ! -x "$compiler" || "$("$compiler" --version)" != *"(${revision:0:8})"* ]]; then
  echo "Build Vx at vx-revision.txt with VX_VERSION_WITH_COMMIT=1 first." >&2
  exit 1
fi
# The dispatch library that this compiler was built with. vxc records its path.
dispatch="$(grep -a -o '/[^"[:cntrl:]]*libvx_[a-z0-9]*_dispatch\.so' "$compiler" | sort -u | head -1)"
if [[ "$(basename "$dispatch")" != libvx_cuda_dispatch.so || ! -f "$dispatch" ]]; then
  echo "$compiler was built with ${dispatch:-no dispatch library}, not libvx_cuda_dispatch.so." >&2
  echo "Rebuild Vx where the CUDA toolkit is installed, so that the kernels run on the GPU." >&2
  exit 1
fi
mkdir -p "$out"

if ! "$compiler" --host default --machine "$machine" "$project_root/src/gpu_kmeans.vx" \
  --action emit-obj -o "$out/gpu_kmeans_cuda.o" > "$out/gpu_kmeans_cuda-build.log" 2>&1; then
  cat "$out/gpu_kmeans_cuda-build.log" >&2
  exit 1
fi
# The PTX of the same source, to check that every kernel compiled for the GPU.
"$compiler" --host default --machine "$machine" "$project_root/src/gpu_kmeans.vx" \
  --action emit-ptx -o "$out/gpu_kmeans.ptx" >> "$out/gpu_kmeans_cuda-build.log" 2>&1
for kernel in vxvector_gpu_seed_v2 vxvector_gpu_assign_v2 vxvector_gpu_assign_angular_v1 \
  vxvector_gpu_zero_v1 vxvector_gpu_accumulate_v1 vxvector_gpu_finish_update_v1; do
  if ! grep -Fq ".visible .entry $kernel(" "$out/gpu_kmeans.ptx"; then
    echo "The PTX of src/gpu_kmeans.vx is missing kernel $kernel." >&2
    exit 1
  fi
done

# -ffp-contract=off: the float64 normalization must round as the CPU kernel's.
"$cc" -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -fPIC \
  -I"$project_root/include" -I"$project_root/src" -I"$cuda_home/include" \
  -DVXVECTOR_VX_REVISION="\"${revision:0:8}\"" \
  -c "$project_root/src/gpu_l2.c" -o "$out/gpu_l2.o"
"$cc" -std=c11 -O2 -Wall -Wextra -Werror -fPIC \
  -I"$project_root/src" -I"$cuda_home/include" \
  -c "$project_root/src/gpu_backend_cuda.c" -o "$out/gpu_backend_cuda.o"
fused="$(objdump -d --no-show-raw-insn "$out/gpu_l2.o" |
  awk '/<vxvector_gpu_normalize_rows>:/{f=1;next} /^$/{f=0} f' |
  grep -ciE '[[:space:]]v?fn?m(add|sub)' || true)"
if [[ "$fused" != 0 ]]; then
  echo "vxvector_gpu_normalize_rows has a fused multiply-add; build with -ffp-contract=off." >&2
  exit 1
fi

"$cc" -shared "$out/gpu_kmeans_cuda.o" "$out/gpu_l2.o" "$out/gpu_backend_cuda.o" \
  "$dispatch" -L"$cuda_lib" -lcudart -lm \
  -Wl,-rpath,"$(dirname "$dispatch")" -Wl,-rpath,"$cuda_lib" -Wl,-z,defs \
  -Wl,-soname,libvxvector_gpu.so -o "$out/libvxvector_gpu.so"

symbols="$(nm -D --defined-only "$out/libvxvector_gpu.so")"
for symbol in vxvector_gpu_probe_v1 vxvector_gpu_restart_v1 \
  vxvector_gpu_assign_v1 vxvector_gpu_reset_timings_v1 \
  vxvector_gpu_timings_v1 vxvector_gpu_version_v1 vxvector_gpu_train_v1; do
  if ! grep -Eq " $symbol$" <<<"$symbols"; then
    echo "The CUDA library is missing required worker symbol $symbol." >&2
    exit 1
  fi
done
printf '%s\n' "$dispatch" > "$out/dispatch-path.txt"
printf 'Built %s\n' "$out/libvxvector_gpu.so"
