"""Verify opt-in, unrotated centroid export with a known-answer SQL fixture."""

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import uuid


def parse_captures(text):
    # PostgreSQL json_agg can pretty-print array elements onto multiple lines.
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(text[match.end():])[0]
            for match in re.finditer(r'(?m)^(?:NOTICE:\s*)?VEXTOR_CENTROIDS ', text)]


def reference_build(container):
    command = ['docker', 'exec', container]
    result = subprocess.run(command + ['cat', '/usr/local/share/vextor-objectives/reference-build.json'],
                            capture_output=True, text=True)
    if result.returncode:
        raise AssertionError('Centroid capture requires the isolated objective-reference image')
    metadata = json.loads(result.stdout)
    if metadata.get('test_only') is not True or metadata.get('vectorchord_version') != '1.1.1':
        raise AssertionError('Unexpected centroid-reference build metadata')
    actual = subprocess.check_output(command + ['sha256sum', '/usr/lib/postgresql/17/lib/vchord.so'], text=True).split()[0]
    if actual != metadata['library_sha256']:
        raise AssertionError('Instrumented VectorChord library differs from its build manifest')
    worker = json.loads(subprocess.check_output(command + ['cat', '/usr/local/libexec/vextor/vx-build-info.json'], text=True))
    expected_pin = (Path(__file__).resolve().parents[1] / 'vx-revision.txt').read_text().strip()
    if worker['vx_revision'] != expected_pin:
        raise AssertionError('Reference image does not use the current Vextor worker pin')
    metadata['vx_build'] = worker
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default=os.environ.get('VEXTOR_POSTGRES_CONTAINER', 'vextor-objectives-postgres-1'))
    args = parser.parse_args()
    metadata = reference_build(args.container)
    schema = 'vextor_capture_' + uuid.uuid4().hex[:12]
    build = f"""CREATE INDEX candidate ON {schema}.items USING vchordrq(features vector_l2_ops)
        WITH(options=$options$[build.internal]
        lists = [1]
        sampling_factor = 40
        kmeans_iterations = 1
        build_threads = 1
        spherical_centroids = false
        $options$);"""
    sql = f"""BEGIN;
        SET LOCAL statement_timeout = '60s';
        SET LOCAL client_min_messages = notice;
        CREATE SCHEMA {schema};
        CREATE TABLE {schema}.items(features vector(2));
        INSERT INTO {schema}.items VALUES ('[0,0]'),('[2,0]');
        ANALYZE {schema}.items;
        SELECT set_config('vextor_test.case','known_mean',true);
        {build}
        DROP INDEX {schema}.candidate;
        SET LOCAL vextor_test.capture_centroids = 'off';
        {build}
        DROP INDEX {schema}.candidate;
        SET LOCAL vextor_test.capture_centroids = 'on';
        {build}
        ROLLBACK;
    """
    result = subprocess.run(['docker', 'exec', '-i', args.container, 'psql', '-X', '-At',
                             '-v', 'ON_ERROR_STOP=1', '-U', 'vextor', '-d', 'vextor'],
                            input=sql, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    captures = parse_captures(result.stderr)
    if len(captures) != 1:
        raise AssertionError(f'Expected one opt-in capture, found {len(captures)}')
    capture = captures[0]
    if capture != dict(trainer='internal', case='known_mean', levels=[[[1.0, 0.0]], [[1.0, 0.0]]]):
        raise AssertionError(f'Expected unrotated mean [1,0] at both levels: {capture}')
    print(json.dumps(dict(result='passed', default_builds=1, opt_out_builds=1, opt_in_builds=1,
                          captured_centroids=capture, reference_build=metadata)), flush=True)


if __name__ == '__main__':
    main()
