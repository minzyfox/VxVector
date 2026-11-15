#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "This script builds the native macOS PostgreSQL extension." >&2
  exit 1
fi

pg_config="${PGRX_PG_CONFIG_PATH:-$(command -v pg_config || true)}"
if [[ -z "$pg_config" || ! -x "$pg_config" ]]; then
  echo "Set PGRX_PG_CONFIG_PATH to the pg_config for native PostgreSQL 17." >&2
  exit 1
fi
if [[ "$("$pg_config" --version)" != *" 17."* ]]; then
  echo "The pg_vxvector extension currently targets PostgreSQL 17." >&2
  exit 1
fi
test -f "$project_root/build/kmeans.o" || {
  echo "Run scripts/build.sh first to build the native Vx kernel." >&2
  exit 1
}

developer_dir="$(xcode-select -p)"
sdk="$(xcrun --show-sdk-path)"
clang_library="$developer_dir/usr/lib"
if [[ ! -f "$clang_library/libclang.dylib" ]]; then
  clang_library="$developer_dir/Toolchains/XcodeDefault.xctoolchain/usr/lib"
fi
if [[ ! -f "$clang_library/libclang.dylib" ]]; then
  echo "Could not find libclang under the active Xcode or Command Line Tools." >&2
  exit 1
fi

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PGRX_PG_CONFIG_PATH="$pg_config"
export VXVECTOR_KERNEL_OBJECT="$project_root/build/kmeans.o"
export SDKROOT="$sdk"
export LIBCLANG_PATH="$clang_library"
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:+$BINDGEN_EXTRA_CLANG_ARGS }-isysroot $sdk"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-undefined -C link-arg=dynamic_lookup"

cargo build --locked --manifest-path "$project_root/pg_vxvector/Cargo.toml"
