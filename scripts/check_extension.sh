#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container="${VXVECTOR_POSTGRES_CONTAINER:-vxvector-postgres-1}"

docker exec -i "$container" sh -c 'cat > /tmp/vxvector-sleep-worker; chmod 755 /tmp/vxvector-sleep-worker' <<'SH'
#!/bin/sh
echo $$ > /tmp/vxvector-sleep-worker.pid
exec sleep 30
SH
docker exec -i "$container" sh -c 'cat > /tmp/vxvector-crash-worker; chmod 755 /tmp/vxvector-crash-worker' <<'SH'
#!/bin/sh
ulimit -c 0
kill -ABRT "$$"
SH
output="$(mktemp)"
trap 'rm -f "$output"; docker exec "$container" rm -f /tmp/vxvector-sleep-worker /tmp/vxvector-sleep-worker.pid /tmp/vxvector-crash-worker >/dev/null 2>&1 || true' EXIT
docker exec -i "$container" psql -X -v ON_ERROR_STOP=1 -U vxvector -d vxvector \
  < "$project_root/tests/postgres_extension.sql" 2>&1 | tee "$output"
grep -Fq "GPU acceleration is unavailable; running CPU k-means" "$output"
docker exec "$container" sh -c 'if test -f /tmp/vxvector-sleep-worker.pid && test -d "/proc/$(cat /tmp/vxvector-sleep-worker.pid)"; then echo "Cancelled worker is still alive" >&2; exit 1; fi'
