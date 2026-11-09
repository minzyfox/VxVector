"""Check pointer contracts in generated LLVM and verify their failure checks."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def check_copy(llvm):
    match = re.search(r'  llvm.func[^\n]* @copy_vector\b.*?(?=\n  llvm.func|\Z)', llvm, re.S)
    if match is None:
        raise AssertionError('Generated LLVM is missing copy_vector')
    body = match.group(0)
    if body.count('llvm.nonnull') != 2 or body.count('dereferenceable') != 2:
        raise AssertionError('Copy pointer validity contracts are missing from LLVM')
    if body.count('noalias_scopes') != 2:
        raise AssertionError('Copy load/store alias contracts are missing from LLVM')


# The only trusted clauses outside the C boundary: two copies from data into
# current. Vx's !overlaps takes one length for both buffers, so it cannot state
# that two buffers of different lengths are disjoint.
KNOWN_TRUSTED = {('run_restart', '!overlaps(destination, source, length)'): 2}


def trusted_clauses(compiler, vx_root, path):
    """Map (function, clause) to the number of trusted clauses that `--audit` lists."""
    result = subprocess.run([str(compiler), str(path), '--audit'], cwd=vx_root,
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(f'Audit of {path.name} failed: {result.stderr}')
    lines = path.read_text().splitlines()
    starts = [(number, match.group(1)) for number, line in enumerate(lines, 1)
              if (match := re.match(r'(?:pub |unsafe )*fn (\w+)', line))]
    counts = {}
    for entry in re.finditer(r':(\d+):\d+: a call in `unsafe` code trusts `(.+?)` of', result.stdout + result.stderr):
        line = int(entry.group(1))
        function = [name for start, name in starts if start <= line][-1]
        key = (function, entry.group(2))
        counts[key] = counts.get(key, 0) + 1
    return counts


def check_audit(compiler, vx_root, path):
    """Raise unless each trusted clause is at the C boundary or in KNOWN_TRUSTED."""
    counts = trusted_clauses(compiler, vx_root, path)
    inside = {key: n for key, n in counts.items() if key[0] != 'vxvector_kmeans_v1'}
    if inside != KNOWN_TRUSTED:
        raise AssertionError(f'Trusted clauses outside the C boundary: {inside}')


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx-root', type=Path, default=Path(os.environ.get('VX_ROOT', project.parent.parent / 'Vx')))
    args = parser.parse_args()
    compiler = args.vx_root / 'target/debug/vxc'
    source = (project / 'src/kmeans.vx').read_text()
    # Without its validity clauses, copy_vector can no longer prove that it
    # initializes its destination, so the checker rejects it before code generation.
    rejected = [('missing_validity', source.replace('requires valid(destination, length)\n', '')
                 .replace('requires valid(source, length)\n', ''), 'E8012')]
    cases = [('kernel', source, None),
             ('missing_alias_contract', source.replace('requires !overlaps(destination, source, length)\n', ''),
              'alias contracts')]
    # Numeric contracts are checked at unsafe calls too. Removing nearest's bound must
    # fail at the real accumulation caller, before code generation.
    proof_mutations = [
        ('nearest_result', source.replace('return best;', 'return k;'), 'E8001'),
        ('nearest_caller', source.replace('ensures return >= 0 && return < k\n', ''), 'E8006'),
        ('buffer_offset', source.replace('return row * dims + d;',
                                        'return rows * dims;'), 'E8001'),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        # Pointer clauses inside the kernel are proved, not trusted. A read one
        # row past a sampled row must appear in the audit.
        path = Path(temporary) / 'audit.vx'
        path.write_text(source)
        check_audit(compiler, args.vx_root, path)
        print(json.dumps(dict(case='audit', result='passed', expected_failure=None)), flush=True)
        past_end = source.replace('let point: *const f32 = offset(data, row * width);',
                                  'let point: *const f32 = offset(data, (row + 1i64) * width);')
        if past_end == source:
            raise AssertionError('past_end mutation did not change the source')
        path.write_text(past_end)
        try:
            check_audit(compiler, args.vx_root, path)
        except AssertionError as error:
            if 'run_restart' not in str(error):
                raise
        else:
            raise AssertionError('past_end must add a trusted clause in run_restart')
        print(json.dumps(dict(case='audit_past_end', result='passed',
                              expected_failure='trusted clause')), flush=True)

        for name, text, expected in proof_mutations + rejected:
            if text == source:
                raise AssertionError(f'{name} mutation did not change the source')
            path = Path(temporary) / (name + '.vx')
            path.write_text(text)
            result = subprocess.run([str(compiler), str(path), '--action', 'emit-mlir'],
                                    cwd=args.vx_root, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 or expected not in result.stderr:
                raise AssertionError(f'{name} must fail with {expected}: {result.stderr}')
            print(json.dumps(dict(case=name, result='passed', expected_failure=expected)), flush=True)
        for name, text, expected in cases:
            path = Path(temporary) / (name + '.vx')
            path.write_text(text)
            result = subprocess.run([str(compiler), str(path), '--action', 'emit-llvm'], cwd=args.vx_root,
                                    env=dict(os.environ, VX_FLAT_DBG='1'), capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise AssertionError(f'{name} did not compile: {result.stderr}')
            if '[flat-codegen] emitted module via the flat path' not in result.stderr:
                raise AssertionError('Kernel did not use the flat code generator')
            try:
                check_copy(result.stdout)
            except AssertionError as error:
                if expected is None or expected not in str(error):
                    raise
            else:
                if expected is not None:
                    raise AssertionError(f'{name} did not fail its intended check')
            print(json.dumps(dict(case=name, result='passed', expected_failure=expected)), flush=True)


if __name__ == '__main__':
    main()
