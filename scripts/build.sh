#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${VX_ROOT:-}" ]]; then
  vx_root="$VX_ROOT"
elif [[ -d "$project_root/../Vx" ]]; then
  vx_root="$project_root/../Vx"
elif [[ -d "$project_root/../../Vx" ]]; then
  vx_root="$project_root/../../Vx"
else
  echo "VX_ROOT is unset and no Vx checkout was found next to this repo." >&2
  echo "Expected one of: $project_root/../Vx or $project_root/../../Vx" >&2
  exit 1
fi
profile="${VX_PROFILE:-release}"
revision="$(cat "$project_root/vx-revision.txt")"
if [[ "$(git -C "$vx_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "VX_ROOT must be checked out at the commit in vx-revision.txt: $revision" >&2
  exit 1
fi
if ! git -C "$vx_root" diff --quiet HEAD --; then
  echo "Build Vx from a clean checkout so its recorded revision identifies its sources." >&2
  exit 1
fi
mkdir -p "$project_root/build"

(
  cd "$vx_root"
  source config.local
  compiler="$vx_root/target/$profile/vxc"
  if [[ "$("$compiler" --version)" != *"(${revision:0:8})"* ]]; then
    echo "Rebuild Vx with VX_VERSION_WITH_COMMIT=1 before building VxVector." >&2
    exit 1
  fi
  if ! VX_FLAT_DBG=1 "$compiler" "$project_root/src/kmeans.vx" \
    --action emit-obj -o "$project_root/build/kmeans.o" > "$project_root/build/kmeans-build.log" 2>&1; then
    cat "$project_root/build/kmeans-build.log" >&2
    exit 1
  fi
  if ! grep -Fq '[flat-codegen] emitted module via the flat path' "$project_root/build/kmeans-build.log"; then
    cat "$project_root/build/kmeans-build.log" >&2
    echo "The k-means build must use the flat code generator." >&2
    exit 1
  fi

  clang -std=c11 -Wall -Wextra -Werror -fPIC -I "$project_root/include" \
    -c "$project_root/src/data_path.c" -o "$project_root/build/data_path.o"

  if [[ "$(uname -s)" == "Darwin" ]]; then
    clang -dynamiclib "$project_root/build/kmeans.o" "$project_root/build/data_path.o" \
      "$vx_root/target/$profile/libvx_std_core.dylib" \
      -Wl,-rpath,"$vx_root/target/$profile" \
      -o "$project_root/build/libvxvector_kmeans.dylib"
  else
    clang -shared "$project_root/build/kmeans.o" "$project_root/build/data_path.o" \
      "$vx_root/target/$profile/libvx_std_core.so" \
      -Wl,-rpath,"$vx_root/target/$profile" -lm \
      -o "$project_root/build/libvxvector_kmeans.so"
  fi

  clang -std=c11 -Wall -Wextra -Werror -I "$project_root/include" \
    "$project_root/scripts/benchmark_data_path.c" -L "$project_root/build" \
    -lvxvector_kmeans -Wl,-rpath,"$project_root/build" \
    -o "$project_root/build/benchmark_data_path"

  python3 "$project_root/scripts/write_build_info.py" --compiler "$compiler" \
    --source "$project_root/src/kmeans.vx" --object "$project_root/build/kmeans.o" \
    --revision "$revision" --output "$project_root/build/vx-build-info.json"

  if [[ "${VX_CHECK_GPU:-0}" == "1" ]]; then
    "$vx_root/target/$profile/vxc" --machine "$vx_root/fleet/a100-40.vx" \
      "$project_root/src/gpu_lloyd.vx" --action emit-llvm \
      > "$project_root/build/gpu_lloyd.mlir"
  fi
)
