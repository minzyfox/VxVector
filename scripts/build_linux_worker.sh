#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vx_root="${VX_ROOT:-$(cd "$project_root/../.." && pwd)/Vx}"
builder="${VEXTOR_VX_BUILD_CONTAINER:-vextor-vx-build}"
mkdir -p "$project_root/build/linux"

if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --dns 8.8.8.8 --dns 1.1.1.1 \
    --mount type=volume,source=vextor_vx_build,target=/workspace \
    "${VEXTOR_VX_BUILD_IMAGE:-vx-build:llvm22}" sleep infinity
else
  docker start "$builder" >/dev/null
fi

git -C "$vx_root" archive HEAD | docker exec -i "$builder" tar -xf - -C /workspace
docker exec "$builder" bash -lc \
  './setup.sh && source config.local && cargo build --locked -j 2 --bin vxc -p vxc && cargo build --locked -j 2 -p vx_std_core' \
  > "$project_root/build/linux/vx-build.log" 2>&1
docker cp "$project_root/src/kmeans.vx" "$builder:/workspace/vextor_kmeans.vx"
docker exec "$builder" bash -lc \
  'source config.local && target/debug/vxc vextor_kmeans.vx --action emit-obj -o /workspace/vextor_kmeans.o' \
  > "$project_root/build/linux/kmeans-build.log" 2>&1
docker cp "$builder:/workspace/vextor_kmeans.o" "$project_root/build/linux/kmeans.o"
git -C "$vx_root" rev-parse HEAD > "$project_root/build/linux/vx-revision.txt"
