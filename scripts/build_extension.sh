#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VXVECTOR_POSTGRES_IMAGE:-vxvector-postgres:17-vxvector}"
builder="${VXVECTOR_EXTENSION_BUILD_CONTAINER:-vxvector-extension-build}"
test -f "$project_root/build/linux/kmeans.o" || {
  echo "Run scripts/build_linux_worker.sh first." >&2
  exit 1
}
python3 - "$project_root" <<'PY_CHECK'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
info_path = root / 'build/linux/vx-build-info.json'
if not info_path.exists():
    sys.exit('Run scripts/build_linux_worker.sh to record the compiler and kernel provenance first.')
info = json.loads(info_path.read_text())
if info['vx_revision'] != (root / 'vx-revision.txt').read_text().strip():
    sys.exit('The Linux object uses a different Vx pin. Rebuild it with scripts/build_linux_worker.sh.')
for key, path in [('kernel_sha256', root / 'src/kmeans.vx'),
                  ('object_sha256', root / 'build/linux/kmeans.o')]:
    if info[key] != hashlib.sha256(path.read_bytes()).hexdigest():
        sys.exit('The Linux kernel changed after its build. Run scripts/build_linux_worker.sh again.')
PY_CHECK

if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --dns 8.8.8.8 --dns 1.1.1.1 \
    --mount type=volume,source=vxvector_extension_build,target=/opt/vxvector-src \
    --mount type=volume,source=vxvector_extension_cargo,target=/usr/local/cargo \
    vxvector-postgres:17-dev sleep infinity
else
  docker start "$builder" >/dev/null
fi
COPYFILE_DISABLE=1 tar -cf - -C "$project_root" pg_vxvector src include build/linux/kmeans.o | \
  docker exec -i "$builder" tar -xf - -C /opt/vxvector-src
docker exec -w /opt/vxvector-src/pg_vxvector "$builder" sh -c \
  'PGRX_PG_CONFIG_PATH=/usr/lib/postgresql/17/bin/pg_config cargo build --locked -j 2' \
  > "$project_root/build/linux/extension-build.log" 2>&1
docker cp "$builder:/opt/vxvector-src/pg_vxvector/target/debug/libpg_vxvector.so" \
  "$project_root/build/linux/libpg_vxvector.so"
# Build for the platform of the local base image. BuildKit otherwise looks for
# a host-platform variant, which an amd64 image on an arm64 host lacks.
platform="$(docker image inspect vxvector-postgres:17-dev --format '{{.Os}}/{{.Architecture}}')"
if docker buildx version >/dev/null 2>&1; then
  docker build --platform "$platform" -f "$project_root/docker/Dockerfile.extension" -t "$image" "$project_root"
else
  DOCKER_BUILDKIT=0 docker build --platform "$platform" -f "$project_root/docker/Dockerfile.extension" -t "$image" "$project_root"
fi
