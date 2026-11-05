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


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx-root', type=Path, default=Path(os.environ.get('VX_ROOT', project.parent.parent / 'Vx')))
    args = parser.parse_args()
    compiler = args.vx_root / 'target/debug/vxc'
    source = (project / 'src/kmeans.vx').read_text()
    cases = [('kernel', source, None),
             ('missing_validity', source.replace('requires valid(destination, length)\n', '')
              .replace('requires valid(source, length)\n', ''), 'pointer validity'),
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
        for name, text, expected in proof_mutations:
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
