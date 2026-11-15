"""Record the pinned Vx source and the exact compiler and kernel bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--object', type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opt-level', required=True, help='the -O level of the object')
    args = parser.parse_args()
    version = subprocess.check_output([str(args.compiler), '--version'], text=True).strip()
    args.output.write_text(json.dumps(dict(vx_revision=args.revision, compiler_version=version,
                                          compiler_sha256=digest(args.compiler), opt_level=f'-O{args.opt_level}',
                                          kernel_sha256=digest(args.source), object_sha256=digest(args.object or args.source.with_suffix('.o')),
                                          platform=platform.platform()), indent=2) + '\n')


if __name__ == '__main__':
    main()
