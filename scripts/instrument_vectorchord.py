"""Add an opt-in centroid export to the exact VectorChord 1.1.1 source.

The hook runs after internal clustering and before centroid rotation. It emits
leaf-first centroid levels as JSON in a NOTICE, without changing their values.
Use the patched library only in an isolated test database.
"""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = 'ffc14c00a0c3733bba9a65dd047b6fdf7e8bdb0e4829f9e67d7bc8e88969595c'
ANCHOR = '''            let result =
                make_internal_build(vector_options, opfamily, internal_build, sampler, &reporter);
'''
HOOK = r'''            // Test-only export of unrotated centroids. The normal index path is unchanged.
            if pgrx::Spi::get_one::<String>(
                "SELECT current_setting('vextor_test.capture_centroids', true)",
            )
            .expect("read centroid capture setting")
            .as_deref()
                == Some("on")
            {
                let case = pgrx::Spi::get_one::<String>(
                    "SELECT current_setting('vextor_test.case', true)",
                )
                .expect("read centroid capture case")
                .expect("centroid capture requires a case label");
                let levels = result.iter().map(|level| &level.centroids).collect::<Vec<_>>();
                pgrx::notice!(
                    "VEXTOR_CENTROIDS {{\"trainer\":\"internal\",\"case\":{:?},\"levels\":{:?}}}",
                    case,
                    levels,
                );
            }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    path = args.source / 'src/index/vchordrq/am/am_build.rs'
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise SystemExit('VectorChord source does not match the recorded 1.1.1 file')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise SystemExit('Expected exactly one internal-training capture point')
    path.write_text(text.replace(ANCHOR, ANCHOR + HOOK))
    print(json.dumps(dict(original_sha256=SOURCE_SHA256,
                          instrumented_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
