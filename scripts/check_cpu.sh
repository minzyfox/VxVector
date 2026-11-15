#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${VXVECTOR_PYTHON:-$project_root/build/correctness-venv/bin/python}"
test -x "$python" || { echo "Create build/correctness-venv and install tests/requirements-correctness.txt first." >&2; exit 1; }
test -f "$project_root/build/libvxvector_kmeans.so" || test -f "$project_root/build/libvxvector_kmeans.dylib"

clang -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror \
  -I "$project_root/include" "$project_root/src/worker.c" -L "$project_root/build" \
  -lvxvector_kmeans -Wl,-rpath,"$project_root/build" -o "$project_root/build/vxvector-worker"
# A stand-in GPU library for the backend-selection tests, and the real Metal
# library when scripts/build_gpu.sh has built it.
suffix="$([[ "$(uname -s)" == Darwin ]] && echo dylib || echo so)"
clang -std=c11 -Wall -Wextra -Werror -shared -fPIC "$project_root/tests/fake_gpu.c" \
  -o "$project_root/build/fake_gpu.$suffix"
real_gpu="$project_root/build/gpu/libvxvector_gpu.dylib"
VXVECTOR_WORKER="$project_root/build/vxvector-worker" \
  VXVECTOR_FAKE_GPU="$project_root/build/fake_gpu.$suffix" \
  VXVECTOR_REAL_GPU="$([[ -f "$real_gpu" ]] && echo "$real_gpu")" \
  "$python" -B -m unittest discover -s "$project_root/tests" -v
"$python" -B "$project_root/scripts/check_cpu_correctness.py"
"$project_root/scripts/check_data_path.sh"

if [[ "${VXVECTOR_CHECK_POSTGRES:-0}" == "1" ]]; then
  "$project_root/scripts/check_postgres.sh"
  "$project_root/scripts/check_extension.sh"
  "$python" -B "$project_root/scripts/check_snapshot.py"
  "$python" -B "$project_root/scripts/check_recall.py"
  "$python" -B "$project_root/scripts/check_real_recall.py" --datasets "$project_root/build/datasets"
fi
