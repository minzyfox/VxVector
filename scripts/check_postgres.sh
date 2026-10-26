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

compose exec -T postgres psql -X -v ON_ERROR_STOP=1 -U vextor -d vextor \
  < "$project_root/tests/postgres_image.sql"
compose exec -T postgres sh -c 'cc --version && clang --version && cargo --version && rustc --version && test -f "$(pg_config --includedir-server)/postgres.h"'

compose exec -T postgres sh -c 'cc -I "$(pg_config --includedir-server)" -x c -fsyntax-only -' <<'C'
#include "postgres.h"
#include "fmgr.h"
PG_MODULE_MAGIC;
C
