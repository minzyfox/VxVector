#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VEXTOR_POSTGRES_DEV_IMAGE:-vextor-postgres:17-dev}"

if docker buildx version >/dev/null 2>&1; then
  docker build --target development -t "$image" "$project_root/docker/postgres"
else
  DOCKER_BUILDKIT=0 docker build --target development -t "$image" "$project_root/docker/postgres"
fi
