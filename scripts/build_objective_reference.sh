#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$project_root/build/objectives"
archive="${VECTORCHORD_SOURCE_ARCHIVE:-$build/VectorChord-1.1.1.tar.gz}"
builder="${VEXTOR_EXTENSION_BUILD_CONTAINER:-vextor-extension-build}"
base_image="${VEXTOR_OBJECTIVE_BASE_IMAGE:-vextor-postgres:17-vextor}"
image="${VEXTOR_OBJECTIVE_IMAGE:-vextor-postgres:17-objectives}"
source_root="$build/source/VectorChord-1.1.1"
reference_rust="1.93.0"
mkdir -p "$build/source"

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 \
    https://codeload.github.com/tensorchord/VectorChord/tar.gz/refs/tags/1.1.1 -o "$archive"
fi
python3 - "$archive" <<'PY'
import hashlib
from pathlib import Path
import sys
expected = 'd70b5595bfc852f1f24c05c0a40272e7deecbb0ddf8ffdddec5afa42c2392b1e'
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    sys.exit('VectorChord archive does not match the recorded 1.1.1 source')
PY
tar -xzf "$archive" -C "$build/source"
python3 "$project_root/scripts/instrument_vectorchord.py" "$source_root" > "$build/instrumentation.json"

if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --dns 8.8.8.8 --dns 1.1.1.1 \
    --mount type=volume,source=vextor_extension_build,target=/opt/vextor-src \
    --mount type=volume,source=vextor_extension_cargo,target=/usr/local/cargo \
    vextor-postgres:17-dev sleep infinity
else
  docker start "$builder" >/dev/null
fi
docker exec "$builder" mkdir -p /opt/vextor-src/vectorchord-objectives
docker cp "$source_root/." "$builder:/opt/vextor-src/vectorchord-objectives/"
docker exec "$builder" rustup toolchain install "$reference_rust" --profile minimal > "$build/rust-install.log" 2>&1
docker exec -w /opt/vextor-src/vectorchord-objectives "$builder" env \
  PGRX_PG_CONFIG_PATH=/usr/lib/postgresql/17/bin/pg_config \
  CARGO_TARGET_DIR=/opt/vextor-src/pg_vextor/target \
  CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo "+$reference_rust" build --locked --lib --features pg17 -j 2 \
  > "$build/vectorchord-build.log" 2>&1
docker cp "$builder:/opt/vextor-src/pg_vextor/target/debug/libvchord.so" "$build/libvchord.so"
docker exec "$builder" rustc "+$reference_rust" --version > "$build/rust-version.txt"
docker image inspect "$base_image" --format '{{.Id}}' > "$build/base-image.txt"
python3 - "$build" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
metadata = dict(vectorchord_version='1.1.1',
                archive_sha256='d70b5595bfc852f1f24c05c0a40272e7deecbb0ddf8ffdddec5afa42c2392b1e',
                instrumentation=json.loads((root / 'instrumentation.json').read_text()),
                library_sha256=hashlib.sha256((root / 'libvchord.so').read_bytes()).hexdigest(),
                rust_version=(root / 'rust-version.txt').read_text().strip(),
                base_image=(root / 'base-image.txt').read_text().strip(),
                profile=dict(opt_level=0, debug=0, features=['pg17']),
                capture_phase='after_internal_training_before_rotation',
                test_only=True)
(root / 'reference-build.json').write_text(json.dumps(metadata, indent=2) + '\n')
PY
if docker buildx version >/dev/null 2>&1; then
  docker build -f "$project_root/docker/Dockerfile.objectives" \
    --build-arg "VEXTOR_POSTGRES_IMAGE=$base_image" -t "$image" "$project_root"
else
  DOCKER_BUILDKIT=0 docker build -f "$project_root/docker/Dockerfile.objectives" \
    --build-arg "VEXTOR_POSTGRES_IMAGE=$base_image" -t "$image" "$project_root"
fi
echo "Built test-only centroid capture image: $image"
