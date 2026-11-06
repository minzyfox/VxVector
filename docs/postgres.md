# PostgreSQL development image

The VxVector image uses PostgreSQL 17 with pgvector and VectorChord 1.1.1.
The upstream VectorChord image is pinned by manifest digest in `docker/postgres/Dockerfile`.
The development stage adds C/Clang, libclang, PostgreSQL server headers, Python, and Rust 1.90.0.
The base image provides build tools. The complete image adds the pgrx extension and separate Linux Vx CPU worker.
GPU acceleration will come later. This image requires no GPU or CUDA libraries.

## Build and start

Use the Docker context containing the development VM. On this machine it is `colima`.
This setting applies only to commands in the current shell.

```sh
export DOCKER_CONTEXT=colima
```

Create `.env` from `.env.example` when `.env` does not already exist.
Set `VXVECTOR_POSTGRES_PASSWORD` to a local password. Git ignores `.env`.
The initial setup generated a password in this machine's `.env`.

```sh
./scripts/build_postgres_image.sh
./scripts/build_linux_worker.sh
./scripts/build_extension.sh
docker-compose up -d --no-build --wait
./scripts/check_postgres.sh
```

Use `docker compose` instead of `docker-compose` when the Compose plugin is installed.
The scripts support either form. The build script also supports Docker without Buildx.
The Compose service is named `postgres` and uses `vxvector-postgres:17-vxvector`.
It initializes a separate `vxvector_postgres_data` volume and a database/user named `vxvector`.
The server preloads `vchord`, and first initialization creates `vector`, `vchord`, and `pg_vxvector` extensions.
Initialization scripts run only on a fresh database volume.

Connect at `127.0.0.1:15432`, or use the client inside the container:

```sh
docker-compose exec postgres psql -U vxvector -d vxvector
```

The existing database on port 5432 uses a separate container and volume.
The VxVector service does not mount that volume or change that database.
The default port can be changed with `VXVECTOR_POSTGRES_PORT` in `.env`.

Stop the VxVector service while retaining its database:

```sh
docker-compose down
```

## Validation

`scripts/check_postgres.sh` checks PostgreSQL 17, installed extensions, and the preload setting.
It builds indexes from hand-written flat and two-level centroid trees.
Both trees use a zero-vector root with `parent = NULL` and declared centroid constraints.
The indexed column is named `features`, rather than `embedding`.
The checks set probes to `4` for the flat tree and `2,4` for the two-level tree.
The checks assert index selection and execute nearest-neighbor queries.
M3 must document probe settings matching the generated tree depth.
The checks roll back all smoke tables and indexes after success.
The script also verifies the C/Clang and Rust tools and compiles a PostgreSQL extension-header probe.
These checks validate the database image and centroid-table contract.
See [M3 checks](milestone_3_cpu_integration.md) for the CPU worker and SQL integration.

Validated on 2026-10-26 with linux/amd64 Colima: PostgreSQL 17.11, pgvector 0.8.2,
VectorChord 1.1.1, and Rust/Cargo 1.90.0. The development package installation updates the base server from 17.9 to 17.11.
See [raw validation evidence](evidence/postgres_image_2026-10-26.txt).
The complete image includes the VxVector worker and `pg_vxvector` 0.1.0.
The Linux Vx compiler remains in its build container. The `cargo-pgrx` CLI is not required by the current build scripts.

## Known Colima DNS issue

Restart the intended profile with explicit resolvers if image pulls fail with DNS errors.
Keep other Colima profiles running.

```sh
colima stop --profile default
colima start --profile default --activate=false --dns 8.8.8.8 --dns 1.1.1.1
```

On this machine, the restart persisted the DNS settings, but the resolver proxy still refused queries.
Using those resolvers directly in the default VM restored registry access:

```sh
colima ssh --profile default -- sudo sh -c 'printf "nameserver 8.8.8.8\nnameserver 1.1.1.1\n" > /etc/resolv.conf'
```

Colima may regenerate that resolver file on a later restart.

## Sources

- [VectorChord 1.1.1 Docker quick start](https://github.com/tensorchord/VectorChord/tree/1.1.1#quick-start)
- [VectorChord external build contract](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)
- [pgrx build requirements](https://github.com/pgcentralfoundation/pgrx/tree/v0.16.1#system-requirements)
