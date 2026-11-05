"""Check small Vx contracts and record the current verifier's limitations.

These examples establish helper properties. They do not prove the complete
k-means loop, foreign buffer capacities, or floating-point accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx-root', type=Path, default=Path(os.environ.get('VX_ROOT', project.parent.parent / 'Vx')))
    args = parser.parse_args()
    compiler = args.vx_root / 'target/debug/vxc'
    source = (project / 'tests/contracts/helpers.vx').read_text()
    mutations = [
        ('index_bound', source.replace('return candidate;', 'return k;'), 'E8001'),
        ('offset_bound', source.replace('return row * 128 + d;', 'return row * 128 + d + 274877906816;'), 'E8001'),
        ('positive_count', source.replace('return count;', 'return 0;'), 'E8001'),
        ('caller_bound', source.replace('choose_index(0, 2, 4, true)', 'choose_index(0, 4, 4, true)'), 'E8006'),
    ]
    print(json.dumps(dict(compiler_sha256=hashlib.file_digest(compiler.open('rb'), 'sha256').hexdigest())), flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        nearest = (project / 'tests/contracts/nearest_loop.vx').read_text()
        variable = (project / 'tests/contracts/variable_offset.vx').read_text()
        cases = [('helpers', source, None)] + mutations
        cases += [('nearest_loop', nearest, None),
                  ('nearest_invalid_update', nearest.replace('best = center;', 'best = k;'), 'E8009'),
                  ('variable_offset', variable, None),
                  ('variable_offset_overflow', variable.replace('return row * dims + d;',
                     'return row * dims + d + 4611686014132420609;'), 'E8001')]
        cases += [(path.stem, path.read_text(), 'E8001')
                  for path in sorted((project / 'tests/contracts/gaps').glob('*.vx'))]
        for name, text, expected in cases:
            path = root / (name + '.vx')
            path.write_text(text)
            started = time.monotonic()
            result = subprocess.run([str(compiler), str(path), '--action', 'emit-mlir'],
                                    cwd=args.vx_root, capture_output=True, text=True, timeout=30)
            diagnostics = result.stderr
            if expected is None:
                if result.returncode != 0 or any(x in diagnostics for x in ('E800', 'ignores a condition', 'Could not')):
                    raise AssertionError(f'Helper proof failed: {diagnostics}')
            elif result.returncode == 0 or expected not in diagnostics:
                raise AssertionError(f'{name} must fail with {expected}: {diagnostics}')
            print(json.dumps(dict(case=name, expected_diagnostic=expected,
                                  seconds=time.monotonic()-started, diagnostics=diagnostics)), flush=True)
    print(json.dumps(dict(result='passed', helper_proofs=5, mutations=6, known_gaps=len(cases)-9)), flush=True)


if __name__ == '__main__':
    main()
