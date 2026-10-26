ARG RUST_IMAGE=rust:1.90.0-slim-bookworm
ARG VECTORCHORD_IMAGE=ghcr.io/tensorchord/vchord-postgres:pg17-v1.1.1@sha256:22fb63e5a505601bf80975c858591558ca1b48952c1cbb3e9124c51b68f8aa8d

FROM ${RUST_IMAGE} AS rust_toolchain

FROM ${VECTORCHORD_IMAGE} AS runtime
COPY init/ /docker-entrypoint-initdb.d/
CMD ["postgres", "-c", "shared_preload_libraries=vchord"]

FROM runtime AS development
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates clang curl git libclang-dev \
       libssl-dev llvm-dev pkg-config postgresql-server-dev-17 python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rust_toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust_toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    CARGO_BUILD_JOBS=2
RUN rustc --version \
    && cargo --version \
    && pg_config --version \
    && test -f "$(pg_config --includedir-server)/postgres.h"
