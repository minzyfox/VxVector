#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/../.." && pwd)/Vx}"
revision="$(cat "$project_root/vx-revision.txt")"
builder="${VXVECTOR_VX_BUILD_CONTAINER:-vxvector-vx-build}"
source_dir="/workspace/vxvector-vx/$revision"
mkdir -p "$project_root/build/linux"

if [[ "$(git -C "$vx_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "VX_ROOT must be checked out at the commit in vx-revision.txt: $revision" >&2
  exit 1
fi

if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --dns 8.8.8.8 --dns 1.1.1.1 \
    --mount type=volume,source=vxvector_vx_build,target=/workspace \
    "${VXVECTOR_VX_BUILD_IMAGE:-vx-build:llvm22}" sleep infinity
else
  docker start "$builder" >/dev/null
fi

# Each revision has its own source tree. Cargo artifacts share the existing cache.
docker exec "$builder" mkdir -p "$source_dir"
git -C "$vx_root" archive "$revision" | docker exec -i "$builder" tar -xf - -C "$source_dir"
docker exec "$builder" ln -sfn /workspace/target "$source_dir/target"
docker exec -w "$source_dir" "$builder" bash -lc \
  './setup.sh && source config.local && export CARGO_TARGET_DIR=/workspace/target VX_VERSION_WITH_COMMIT=1 && cargo build --locked -j 2 --bin vxc -p vxc && cargo build --locked -j 2 -p vx_std_core' \
  > "$project_root/build/linux/vx-build.log" 2>&1
docker cp "$project_root/src/kmeans.vx" "$builder:$source_dir/vxvector_kmeans.vx"
docker exec -w "$source_dir" "$builder" bash -lc \
  'source config.local && VX_FLAT_DBG=1 target/debug/vxc vxvector_kmeans.vx --action emit-obj -o vxvector_kmeans.o' \
  > "$project_root/build/linux/kmeans-build.log" 2>&1
if ! grep -Fq '[flat-codegen] emitted module via the flat path' "$project_root/build/linux/kmeans-build.log"; then
  cat "$project_root/build/linux/kmeans-build.log" >&2
  echo "The k-means build must use the flat code generator." >&2
  exit 1
fi
docker cp "$builder:$source_dir/vxvector_kmeans.o" "$project_root/build/linux/kmeans.o"
docker cp "$project_root/scripts/write_build_info.py" "$builder:/tmp/vxvector-write-build-info.py"
docker exec "$builder" python3 /tmp/vxvector-write-build-info.py \
  --compiler "$source_dir/target/debug/vxc" --source "$source_dir/vxvector_kmeans.vx" \
  --object "$source_dir/vxvector_kmeans.o" --revision "$revision" --output /tmp/vxvector-build-info.json
docker cp "$builder:/tmp/vxvector-build-info.json" "$project_root/build/linux/vx-build-info.json"
printf '%s\n' "$revision" > "$project_root/build/linux/vx-revision.txt"
printf '%s\n' "$source_dir" > "$project_root/build/linux/vx-source-path.txt"
