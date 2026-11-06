#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose --project-directory "$project_root" -f "$project_root/docker-compose.yml" "$@"
  else
    docker-compose --project-directory "$project_root" -f "$project_root/docker-compose.yml" "$@"
  fi
}

postgres_exec() {
  if [[ -n "${VXVECTOR_POSTGRES_CONTAINER:-}" ]]; then
    docker exec -i "$VXVECTOR_POSTGRES_CONTAINER" "$@"
  else
    compose exec -T postgres "$@"
  fi
}

postgres_exec psql -X -v ON_ERROR_STOP=1 -U vxvector -d vxvector \
  < "$project_root/tests/postgres_image.sql"
postgres_exec sh -c 'cc --version && clang --version && cargo --version && rustc --version && test -f "$(pg_config --includedir-server)/postgres.h"'

postgres_exec sh -c 'cc -I "$(pg_config --includedir-server)" -x c -fsyntax-only -' <<'C'
#include "postgres.h"
#include "fmgr.h"
PG_MODULE_MAGIC;
C
