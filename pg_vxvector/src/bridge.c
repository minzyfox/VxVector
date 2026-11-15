#define _GNU_SOURCE
#include "postgres.h"
#include "access/table.h"
#include "access/tableam.h"
#include "catalog/namespace.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "commands/defrem.h"
#include "executor/spi.h"
#include "executor/tuptable.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "storage/lmgr.h"
#include "storage/ipc.h"
#include "utils/acl.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/json.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/rls.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
#include "vxvector_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;
static char *worker_path;
static int backend_mode = VXVECTOR_MODE_AUTO;
static pid_t active_worker_pid;
static int active_worker_fd = -1;

static const struct config_enum_entry backend_modes[] = {
    {"auto", VXVECTOR_MODE_AUTO, false},
    {"cpu", VXVECTOR_MODE_CPU, false},
    {"gpu", VXVECTOR_MODE_GPU, false},
    {NULL, 0, false}
};

static void backend_exit(int code, Datum argument) {
    (void)code;
    (void)argument;
    if (active_worker_fd >= 0)
        close(active_worker_fd);
    if (active_worker_pid > 0) {
        kill(active_worker_pid, SIGKILL);
        while (waitpid(active_worker_pid, NULL, 0) < 0 && errno == EINTR) {}
    }
}

void vxvector_pg_init(void) {
    on_proc_exit(backend_exit, (Datum)0);
    DefineCustomStringVariable("vxvector.worker_path", "VxVector worker executable", NULL,
                               &worker_path, "/usr/local/libexec/vxvector/vxvector-worker",
                               PGC_SUSET, 0, NULL, NULL, NULL);
    DefineCustomEnumVariable("vxvector.backend_mode", "GPU backend selection mode", NULL,
                             &backend_mode, VXVECTOR_MODE_AUTO, backend_modes,
                             PGC_SUSET, 0, NULL, NULL, NULL);
}

struct build_state {
    Relation relation;
    TableScanDesc scan;
    TupleTableSlot *slot;
    Snapshot snapshot;
    int fd;
    pid_t pid;
    double deadline;
    bool spi;
};

static double now_seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
        ereport(ERROR, (errmsg("Could not read the monotonic clock")));
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void cleanup(struct build_state *s) {
    active_worker_pid = 0;
    active_worker_fd = -1;
    if (s->fd >= 0) {
        close(s->fd);
        s->fd = -1;
    }
    if (s->pid > 0) {
        kill(s->pid, SIGKILL);
        while (waitpid(s->pid, NULL, 0) < 0 && errno == EINTR) {}
        s->pid = 0;
    }
    if (s->scan) {
        table_endscan(s->scan);
        s->scan = NULL;
    }
    if (s->slot) {
        ExecDropSingleTupleTableSlot(s->slot);
        s->slot = NULL;
    }
    if (s->snapshot) {
        UnregisterSnapshot(s->snapshot);
        s->snapshot = NULL;
    }
    if (s->relation) {
        table_close(s->relation, NoLock);
        s->relation = NULL;
    }
    if (s->spi) {
        SPI_finish();
        s->spi = false;
    }
}

static void check_deadline(struct build_state *s) {
    CHECK_FOR_INTERRUPTS();
    if (now_seconds() >= s->deadline)
        ereport(ERROR, (errcode(ERRCODE_QUERY_CANCELED), errmsg("VxVector worker timed out")));
}

static void transfer(struct build_state *s, void *data, size_t bytes, bool writing) {
    char *position = data;
    while (bytes) {
        check_deadline(s);
        ssize_t count = writing ? send(s->fd, position, bytes, MSG_NOSIGNAL) : recv(s->fd, position, bytes, 0);
        if (count > 0) {
            position += count;
            bytes -= (size_t)count;
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        if (count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK))
            ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION), errmsg("VxVector worker closed the connection or failed")));
        struct pollfd descriptor = {s->fd, writing ? POLLOUT : POLLIN, 0};
        int ready = poll(&descriptor, 1, 50);
        if (ready < 0 && errno != EINTR)
            ereport(ERROR, (errmsg("Could not wait for the VxVector worker: %m")));
    }
}

static void spawn_worker(struct build_state *s) {
    int sockets[2];
    int socket_type = SOCK_STREAM;
#ifdef SOCK_CLOEXEC
    socket_type |= SOCK_CLOEXEC;
#endif
    if (socketpair(AF_UNIX, socket_type, 0, sockets) != 0)
        ereport(ERROR, (errmsg("Could not create the VxVector worker socket: %m")));
#ifndef SOCK_CLOEXEC
    for (int i = 0; i < 2; ++i) {
        if (fcntl(sockets[i], F_SETFD, FD_CLOEXEC) < 0) {
            int saved_errno = errno;
            close(sockets[0]);
            close(sockets[1]);
            errno = saved_errno;
            ereport(ERROR, (errmsg("Could not mark the VxVector worker socket close-on-exec: %m")));
        }
    }
#endif
    s->fd = sockets[0];
    active_worker_fd = s->fd;
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (code) {
        close(sockets[1]);
        ereport(ERROR, (errmsg("Could not initialize the worker spawn actions: %s", strerror(code))));
    }
    if (!code) code = posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, sockets[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, sockets[1]);
    char *arguments[] = {worker_path, NULL};
    if (!code) code = posix_spawn(&s->pid, worker_path, &actions, NULL, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sockets[1]);
    if (code)
        ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION), errmsg("Could not start the VxVector worker: %s", strerror(code))));
    active_worker_pid = s->pid;
    if (fcntl(s->fd, F_SETFL, O_NONBLOCK) < 0)
        ereport(ERROR, (errmsg("Could not configure the VxVector worker socket: %m")));
}

static void read_reply(struct build_state *s, struct vxvector_worker_reply_v2 *reply,
                        struct vxvector_cpu_options *options, struct vxvector_cpu_plan *plan) {
    transfer(s, reply, sizeof(*reply), false);
    reply->base.message[sizeof(reply->base.message) - 1] = 0;
    reply->reason_text[sizeof(reply->reason_text) - 1] = 0;
    reply->device[sizeof(reply->device) - 1] = 0;
    reply->gpu_version[sizeof(reply->gpu_version) - 1] = 0;
    if (reply->base.status)
        ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                        errmsg("VxVector worker: %s", reply->base.message)));
    uint64 expected_buffer_bytes = plan->buffer_bytes + (uint64_t)options->dims * 4;
    if (reply->backend != VXVECTOR_BACKEND_CPU && reply->backend != VXVECTOR_BACKEND_GPU)
        ereport(ERROR, (errmsg("VxVector worker returned an invalid backend")));
    if (reply->backend == VXVECTOR_BACKEND_CPU && reply->gpu_bytes != 0)
        ereport(ERROR, (errmsg("VxVector CPU worker returned GPU buffer usage")));
    if (reply->gpu_bytes > UINT64_MAX - expected_buffer_bytes)
        ereport(ERROR, (errmsg("VxVector worker returned overflowing buffer usage")));
    expected_buffer_bytes += reply->gpu_bytes;
    if (reply->base.dims != options->dims || reply->base.nodes != plan->node_count ||
        reply->base.samples != plan->sample_count ||
        reply->base.buffer_bytes != expected_buffer_bytes)
        ereport(ERROR, (errmsg("VxVector worker returned an incompatible response")));
}

static void finish_worker(struct build_state *s) {
    close(s->fd);
    s->fd = -1;
    active_worker_fd = -1;
    for (;;) {
        check_deadline(s);
        int status;
        pid_t waited = waitpid(s->pid, &status, WNOHANG);
        if (waited == s->pid) {
            s->pid = 0;
            active_worker_pid = 0;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                ereport(ERROR, (errmsg("VxVector worker exited unsuccessfully")));
            return;
        }
        if (waited < 0 && errno != EINTR)
            ereport(ERROR, (errmsg("Could not reap the VxVector worker: %m")));
        pg_usleep(10000);
    }
}

static int vector_values(Datum datum, Oid send_function, int expected_dims, float *values) {
    bytea *binary = OidSendFunctionCall(send_function, datum);
    const unsigned char *bytes = (const unsigned char *)VARDATA_ANY(binary);
    size_t length = VARSIZE_ANY_EXHDR(binary);
    if (length < 4)
        ereport(ERROR, (errmsg("Invalid pgvector binary representation")));
    int dims = ((int)bytes[0] << 8) | bytes[1];
    if (dims <= 0 || dims > 16000 || length != 4 + (size_t)dims * 4 || bytes[2] || bytes[3])
        ereport(ERROR, (errmsg("Invalid pgvector dimensions or binary representation")));
    if (expected_dims > 0 && dims != expected_dims)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Source contains mixed vector dimensions")));
    if (values) {
        for (int d = 0; d < dims; ++d) {
            const unsigned char *component = bytes + 4 + d * 4;
            uint32_t bits = ((uint32_t)component[0] << 24) | ((uint32_t)component[1] << 16) |
                            ((uint32_t)component[2] << 8) | component[3];
            memcpy(values + d, &bits, 4);
            if (!isfinite(values[d]))
                ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Source contains a nonfinite vector component")));
        }
    }
    pfree(binary);
    return dims;
}

static char *output_name(const char *provided, const char *fallback, const char *schema) {
    const char *name = provided ? provided : fallback;
    /* Output names are identifiers in the source schema, never SQL fragments. */
    if (!*name || strlen(name) >= NAMEDATALEN)
        ereport(ERROR, (errcode(ERRCODE_INVALID_NAME), errmsg("Output names must contain 1 through 63 bytes")));
    return quote_qualified_identifier(schema, name);
}

static void sql(const char *statement) {
    CHECK_FOR_INTERRUPTS();
    if (SPI_execute(statement, false, 0) < 0)
        ereport(ERROR, (errmsg("VxVector could not execute a database statement")));
}

char *vxvector_pg_run(Oid relation_id, const char *column, const int32_t *lists, int list_count,
                    int sampling, int batch, int iterations, int restarts, const char *metric,
                    bool skip_index, bool spherical, bool residual, int seed, int64 budget,
                    const char *centroid_name, const char *index_name, int timeout_ms) {
    if (list_count < 1 || list_count > 2 || budget <= 0 || timeout_ms <= 0)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Use one or two positive list counts, a positive budget, and a positive worker timeout")));
    struct build_state *s = palloc0(sizeof(*s));
    s->fd = -1;
    char *result = NULL;
    PG_TRY();
    {
        double count_start = now_seconds();
        s->relation = table_open(relation_id, AccessShareLock);
        if (s->relation->rd_rel->relkind != RELKIND_RELATION || s->relation->rd_rel->relhassubclass ||
            s->relation->rd_rel->relam != get_table_am_oid("heap", false))
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("VxVector currently accepts ordinary heap tables")));
        if (check_enable_rls(relation_id, InvalidOid, false) == RLS_ENABLED)
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("VxVector does not yet support tables with active row-level security")));
        AttrNumber attribute = get_attnum(relation_id, column);
        if (attribute <= 0)
            ereport(ERROR, (errcode(ERRCODE_UNDEFINED_COLUMN), errmsg("Source vector column does not exist")));
        if (pg_class_aclcheck(relation_id, GetUserId(), ACL_SELECT) != ACLCHECK_OK &&
            pg_attribute_aclcheck(relation_id, attribute, GetUserId(), ACL_SELECT) != ACLCHECK_OK)
            aclcheck_error(ACLCHECK_NO_PRIV, OBJECT_COLUMN, column);
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("Could not connect VxVector to SPI")));
        s->spi = true;
        if (SPI_execute("SELECT extnamespace, (SELECT extnamespace FROM pg_catalog.pg_extension WHERE extname = 'vchord') FROM pg_catalog.pg_extension WHERE extname = 'vector'", true, 1) != SPI_OK_SELECT || SPI_processed != 1)
            ereport(ERROR, (errmsg("pgvector must be installed")));
        bool is_null;
        Oid vector_namespace = DatumGetObjectId(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &is_null));
        Oid index_namespace = DatumGetObjectId(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &is_null));
        if (is_null)
            ereport(ERROR, (errmsg("VectorChord must be installed")));
        Oid vector_type = GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid, CStringGetDatum("vector"), ObjectIdGetDatum(vector_namespace));
        if (TupleDescAttr(RelationGetDescr(s->relation), attribute - 1)->atttypid != vector_type)
            ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("Source column must have the pgvector vector type")));
        Oid send_function;
        bool variable_length;
        getTypeBinaryOutputInfo(vector_type, &send_function, &variable_length);
        s->snapshot = RegisterSnapshot(GetActiveSnapshot());
        s->slot = table_slot_create(s->relation, NULL);
        s->scan = table_beginscan_strat(s->relation, s->snapshot, 0, NULL, true, false);
        uint64 rows = 0;
        int dims = 0;
        while (table_scan_getnextslot(s->scan, ForwardScanDirection, s->slot)) {
            CHECK_FOR_INTERRUPTS();
            Datum datum = slot_getattr(s->slot, attribute, &is_null);
            if (is_null)
                ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), errmsg("Source contains a NULL vector")));
            if (!dims)
                dims = vector_values(datum, send_function, 0, NULL);
            ++rows;
            if (rows > INT64_MAX)
                ereport(ERROR, (errmsg("Source row count exceeds the supported range")));
            ExecClearTuple(s->slot);
        }
        table_endscan(s->scan);
        s->scan = NULL;
        double count_seconds = now_seconds() - count_start;

        int metric_code = strcmp(metric, "l2") == 0 ? 0 : strcmp(metric, "ip") == 0 ? 1 : strcmp(metric, "cos") == 0 ? 2 : -1;
        struct vxvector_cpu_options options = {
            .row_count = rows, .dims = dims, .root_count = list_count == 2 ? lists[0] : 0,
            .leaf_count = lists[list_count - 1], .sampling_factor = sampling, .batch_size = batch,
            .iterations = iterations, .restarts = restarts, .metric = metric_code,
            .spherical = spherical, .seed = seed, .memory_budget_bytes = (uint64)budget
        };
        /* Reserve backend output buffers, float/Datum row conversions, and IPC storage.
         * PostgreSQL executor, relation buffers, and allocator overhead are separate.
         */
        uint64 node_count = 1 + (uint64)(options.root_count > 0 ? options.root_count : 0) +
                              (uint64)(options.leaf_count > 0 ? options.leaf_count : 0);
        uint64 reserve = node_count * ((uint64)dims * 4 + 4) + (uint64)dims * 32 + 65536;
        if (reserve >= (uint64)budget)
            ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("VxVector buffers exceed the configured memory budget")));
        options.memory_budget_bytes -= reserve;
        struct vxvector_cpu_options planned_options = options;
        if (planned_options.memory_budget_bytes <= (uint64)dims * 4)
            ereport(ERROR, (errmsg("Memory budget cannot hold a worker input row")));
        planned_options.memory_budget_bytes -= (uint64)dims * 4;
        struct vxvector_cpu_plan plan;
        struct vxvector_data_error error;
        if (vxvector_plan_cpu_v1(&planned_options, &plan, &error) != 0)
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s", error.message)));
        uint64 node_bytes = plan.node_count * (uint64)dims * 4;
        if (node_bytes > MaxAllocSize || plan.node_count * 4 > MaxAllocSize)
            ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("Centroid output exceeds PostgreSQL allocation limits")));
        float *nodes = palloc((Size)node_bytes);
        int32_t *parents = palloc((Size)plan.node_count * 4);
        float *values = palloc((Size)dims * 4);
        const char *schema = get_namespace_name(RelationGetNamespace(s->relation));
        const char *source_name = pstrdup(RelationGetRelationName(s->relation));
        char *centroids = output_name(centroid_name, psprintf("%s_vxvector_centroids", source_name), schema);
        char *index = output_name(index_name, psprintf("%s_vxvector_idx", source_name), schema);
        char *source = quote_qualified_identifier(schema, source_name);
        char *vector_sql_type = quote_qualified_identifier(get_namespace_name(vector_namespace), "vector");

        s->deadline = now_seconds() + (double)timeout_ms / 1000.0;
        spawn_worker(s);
        struct vxvector_worker_request_v2 request = {
            .magic = VXVECTOR_WORKER_MAGIC_V2,
            .options = options,
            .backend_mode = backend_mode
        };
        transfer(s, &request, sizeof(request), true);
        struct vxvector_worker_reply_v2 reply;
        read_reply(s, &reply, &options, &plan);
        double transfer_start = now_seconds();
        s->scan = table_beginscan_strat(s->relation, s->snapshot, 0, NULL, true, false);
        uint64 sent = 0;
        while (table_scan_getnextslot(s->scan, ForwardScanDirection, s->slot)) {
            check_deadline(s);
            Datum datum = slot_getattr(s->slot, attribute, &is_null);
            if (is_null)
                ereport(ERROR, (errmsg("Source contains a NULL vector")));
            vector_values(datum, send_function, dims, values);
            uint32 dimension = (uint32)dims;
            transfer(s, &dimension, sizeof(dimension), true);
            transfer(s, values, (size_t)dims * 4, true);
            ++sent;
            ExecClearTuple(s->slot);
        }
        if (sent != rows)
            ereport(ERROR, (errmsg("Source count changed within the registered snapshot")));
        if (shutdown(s->fd, SHUT_WR) != 0)
            ereport(ERROR, (errmsg("Could not finish the worker input stream: %m")));
        double transfer_seconds = now_seconds() - transfer_start;
        read_reply(s, &reply, &options, &plan);
        transfer(s, nodes, (size_t)node_bytes, false);
        transfer(s, parents, (size_t)plan.node_count * 4, false);
        finish_worker(s);
        table_endscan(s->scan);
        s->scan = NULL;
        ExecDropSingleTupleTableSlot(s->slot);
        s->slot = NULL;
        UnregisterSnapshot(s->snapshot);
        s->snapshot = NULL;
        table_close(s->relation, NoLock);
        s->relation = NULL;

        double write_start = now_seconds();
        sql(psprintf("CREATE TABLE %s (id integer NOT NULL UNIQUE, parent integer, vector %s(%d) NOT NULL)", centroids, vector_sql_type, dims));
        char *insert = psprintf("INSERT INTO %s VALUES ($1,$2,$3::real[]::%s)", centroids, vector_sql_type);
        Datum *components = palloc((Size)dims * sizeof(Datum));
        for (uint64 node = 0; node < plan.node_count; ++node) {
            CHECK_FOR_INTERRUPTS();
            int expected_parent_min = node > (uint64)options.root_count ? 1 : 0;
            if (node == 0 ? parents[node] != -1 :
                (options.root_count == 0 || node <= (uint64)options.root_count) ? parents[node] != 0 :
                (parents[node] < expected_parent_min || parents[node] > options.root_count))
                ereport(ERROR, (errmsg("Worker returned an invalid centroid tree")));
            for (int d = 0; d < dims; ++d) {
                float value = nodes[node * dims + d];
                if (!isfinite(value))
                    ereport(ERROR, (errmsg("Worker returned a nonfinite centroid")));
                components[d] = Float4GetDatum(value);
            }
            ArrayType *array = construct_array(components, dims, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);
            Oid types[] = {INT4OID, INT4OID, FLOAT4ARRAYOID};
            Datum arguments[] = {Int32GetDatum((int32)node), Int32GetDatum(parents[node]), PointerGetDatum(array)};
            char nulls[] = {' ', node == 0 ? 'n' : ' ', ' '};
            if (SPI_execute_with_args(insert, 3, types, arguments, nulls, false, 0) != SPI_OK_INSERT)
                ereport(ERROR, (errmsg("Could not write VxVector centroids")));
            pfree(array);
        }
        double write_seconds = now_seconds() - write_start;
        double index_start = now_seconds();
        if (!skip_index) {
            const char *opclass = metric_code == 0 ? "vector_l2_ops" : metric_code == 1 ? "vector_ip_ops" : "vector_cosine_ops";
            /* JSON quoted strings are valid TOML basic strings for SQL identifiers. */
            StringInfoData table_literal;
            initStringInfo(&table_literal);
            escape_json(&table_literal, centroids);
            char *options_text = psprintf("residual_quantization = %s\n[build.external]\ntable = %s\n", residual ? "true" : "false", table_literal.data);
            sql(psprintf("CREATE INDEX %s ON %s USING vchordrq (%s %s) WITH (options = %s)",
                            quote_identifier(index_name ? index_name : psprintf("%s_vxvector_idx", source_name)),
                            source, quote_identifier(column), quote_qualified_identifier(get_namespace_name(index_namespace), opclass),
                            quote_literal_cstr(options_text)));
        }
        double index_seconds = now_seconds() - index_start;
        StringInfoData json;
        initStringInfo(&json);
        appendStringInfo(&json, "{\"backend\":\"%s\",\"backend_reason\":",
                         reply.backend == VXVECTOR_BACKEND_GPU ? "gpu" : "cpu");
        escape_json(&json, reply.reason_text);
        appendStringInfoString(&json, ",\"device\":");
        if (reply.device[0]) escape_json(&json, reply.device); else appendStringInfoString(&json, "null");
        appendStringInfoString(&json, ",\"gpu_version\":");
        if (reply.gpu_version[0]) escape_json(&json, reply.gpu_version); else appendStringInfoString(&json, "null");
        appendStringInfoString(&json, ",\"centroid_table\":");
        escape_json(&json, centroids);
        appendStringInfoString(&json, ",\"index\":");
        if (skip_index) appendStringInfoString(&json, "null"); else escape_json(&json, index);
        appendStringInfo(&json, ",\"rows\":%llu,\"samples\":%llu,\"dims\":%d,\"nodes\":%llu,\"seed\":%d,\"buffer_bytes\":%llu,"
                         "\"timings\":{\"count_seconds\":%.9f,\"transfer_seconds\":%.9f,\"worker_read_seconds\":%.9f,"
                         "\"sample_seconds\":%.9f,\"clustering_seconds\":%.9f,\"gpu_allocation_seconds\":%.9f,"
                         "\"gpu_copy_in_seconds\":%.9f,\"gpu_training_seconds\":%.9f,\"gpu_copy_out_seconds\":%.9f,"
                         "\"centroid_write_seconds\":%.9f,\"index_seconds\":%.9f}}",
                         (unsigned long long)rows, (unsigned long long)plan.sample_count, dims,
                         (unsigned long long)plan.node_count, seed,
                         (unsigned long long)(reserve + reply.base.buffer_bytes),
                         count_seconds, transfer_seconds, reply.base.read_seconds, reply.base.sample_seconds,
                         reply.base.clustering_seconds, reply.gpu_allocation_seconds,
                         reply.gpu_copy_in_seconds, reply.gpu_training_seconds, reply.gpu_copy_out_seconds,
                         write_seconds, index_seconds);
        /* SPI_finish releases its context. Keep the result in the caller's context. */
        result = SPI_palloc(json.len + 1);
        memcpy(result, json.data, json.len + 1);
        cleanup(s);
    }
    PG_CATCH();
    {
        cleanup(s);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return result;
}
