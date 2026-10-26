#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VEXTOR_POSTGRES_IMAGE:-vextor-postgres:17-vextor}"
builder="${VEXTOR_EXTENSION_BUILD_CONTAINER:-vextor-extension-build}"
test -f "$project_root/build/linux/kmeans.o" || {
  echo "Run scripts/build_linux_worker.sh first." >&2
  exit 1
}
if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --dns 8.8.8.8 --dns 1.1.1.1 \
    --mount type=volume,source=vextor_extension_build,target=/opt/vextor-src \
    --mount type=volume,source=vextor_extension_cargo,target=/usr/local/cargo \
    vextor-postgres:17-dev sleep infinity
else
  docker start "$builder" >/dev/null
fi
COPYFILE_DISABLE=1 tar -cf - -C "$project_root" pg_vextor src include | \
  docker exec -i "$builder" tar -xf - -C /opt/vextor-src
docker exec -w /opt/vextor-src/pg_vextor "$builder" sh -c \
  'PGRX_PG_CONFIG_PATH=/usr/lib/postgresql/17/bin/pg_config cargo build --locked -j 2' \
  > "$project_root/build/linux/extension-build.log" 2>&1
docker cp "$builder:/opt/vextor-src/pg_vextor/target/debug/libpg_vextor.so" \
  "$project_root/build/linux/libpg_vextor.so"
if docker buildx version >/dev/null 2>&1; then
  docker build -f "$project_root/docker/Dockerfile.extension" -t "$image" "$project_root"
else
  DOCKER_BUILDKIT=0 docker build -f "$project_root/docker/Dockerfile.extension" -t "$image" "$project_root"
fi
