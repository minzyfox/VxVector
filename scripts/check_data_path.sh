#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT

clang -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I "$project_root/include" "$project_root/tests/data_path_memory.c" \
  "$project_root/build/kmeans.o" -lm -o "$test_dir/data_path_memory"
"$test_dir/data_path_memory"
