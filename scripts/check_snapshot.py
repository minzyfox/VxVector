"""Change a source after counting and prove the worker sees the original snapshot."""

import json
import os
import subprocess
import time


CONTAINER = os.environ.get("VXVECTOR_POSTGRES_CONTAINER", "vxvector-postgres-1")


def docker(*arguments, **kwargs):
    return subprocess.run(["docker", "exec", "-i", CONTAINER, *arguments],
                          check=True, text=True, capture_output=True, **kwargs)


def sql(statement):
    return docker("psql", "-X", "-v", "ON_ERROR_STOP=1", "-At", "-U", "vxvector",
                  "-d", "vxvector", "-c", statement).stdout.strip()


def main():
    created = False
    running = None
    try:
        sql("""
            CREATE SCHEMA vxvector_snapshot_probe;
            CREATE TABLE vxvector_snapshot_probe.source (id integer, features vector(2));
            INSERT INTO vxvector_snapshot_probe.source
              SELECT i, ARRAY[(i % 4) * 10, i % 3]::vector FROM generate_series(1,64) i;
            SELECT vxvector.create_vector_index('vxvector_snapshot_probe.source','features',ARRAY[4],
              skip_index_build=>true,centroid_table=>'baseline');
        """)
        created = True
        docker("sh", "-c", "cat > /tmp/vxvector-delayed-worker; chmod 755 /tmp/vxvector-delayed-worker",
               input="""#!/bin/sh
touch /tmp/vxvector-snapshot-ready
sleep 5
exec /usr/local/libexec/vxvector/vxvector-worker
""")
        docker("rm", "-f", "/tmp/vxvector-snapshot-ready")
        running = subprocess.Popen(
            ["docker", "exec", CONTAINER, "psql", "-X", "-v", "ON_ERROR_STOP=1", "-At",
             "-U", "vxvector", "-d", "vxvector", "-c", """
                 SET statement_timeout='20s';
                 SET vxvector.worker_path='/tmp/vxvector-delayed-worker';
                 SELECT vxvector.create_vector_index('vxvector_snapshot_probe.source','features',ARRAY[4],
                   skip_index_build=>true,centroid_table=>'concurrent');
             """], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 10
        while subprocess.run(["docker", "exec", CONTAINER, "test", "-f", "/tmp/vxvector-snapshot-ready"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            if time.monotonic() > deadline or running.poll() is not None:
                raise RuntimeError("The delayed worker did not reach the snapshot checkpoint")
            time.sleep(0.05)
        sql("""
            INSERT INTO vxvector_snapshot_probe.source
              SELECT i, '[1000,1000]'::vector FROM generate_series(65,128) i;
            UPDATE vxvector_snapshot_probe.source SET features='[500,500]' WHERE id <= 4;
        """)
        output, errors = running.communicate(timeout=25)
        if running.returncode:
            raise RuntimeError(errors)
        result = json.loads(next(line for line in output.splitlines() if line.startswith("{")))
        if result["rows"] != 64 or result["samples"] != 64:
            raise AssertionError(f"Concurrent changes entered the registered snapshot: {result}")
        if sql("""SELECT count(*) FROM (
            (SELECT * FROM vxvector_snapshot_probe.baseline EXCEPT SELECT * FROM vxvector_snapshot_probe.concurrent)
            UNION ALL
            (SELECT * FROM vxvector_snapshot_probe.concurrent EXCEPT SELECT * FROM vxvector_snapshot_probe.baseline)
        ) changes""") != "0":
            raise AssertionError("Concurrent insert/update changed centroids from the original snapshot")
        if sql("SELECT count(*) FROM vxvector_snapshot_probe.source") != "128":
            raise AssertionError("The concurrent insertion did not commit")
        print("Snapshot check passed: concurrent inserts and updates were excluded; centroids matched the original 64 rows.")
    finally:
        if running is not None and running.poll() is None:
            running.communicate(timeout=25)
        if created:
            sql("DROP SCHEMA vxvector_snapshot_probe CASCADE")
        docker("rm", "-f", "/tmp/vxvector-delayed-worker", "/tmp/vxvector-snapshot-ready")


if __name__ == "__main__":
    main()
