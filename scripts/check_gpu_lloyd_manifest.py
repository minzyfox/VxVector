#!/usr/bin/env python3
import json
import sys


EXPECTED = {
    "vxvector_gpu_assign_v1": (
        [("points", 7), ("rows", 1), ("dims", 1), ("k", 1), ("metric", 1),
         ("centers", 7), ("labels", 5)],
        {"block": 1, "grid": [{"param": "rows"}], "grid_loops": True},
    ),
    "vxvector_gpu_update_v1": (
        [("points", 7), ("rows", 1), ("dims", 1), ("k", 1), ("spherical", 1),
         ("centers", 7), ("labels", 5), ("updated", 7)],
        {"block": 1, "grid": [{"param": "k"}], "grid_loops": True},
    ),
    "vxvector_gpu_copy_v1": (
        [("k", 1), ("dims", 1), ("centers", 7), ("updated", 7)],
        {"block": 1, "grid": [{"param": "k"}], "grid_loops": True},
    ),
}


def main(path):
    with open(path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("target") != "sm_80":
        raise ValueError(f"expected A100 target sm_80, got {manifest.get('target')!r}")

    kernels = {kernel["name"]: kernel for kernel in manifest.get("kernels", [])}
    if set(kernels) != set(EXPECTED):
        raise ValueError(f"unexpected kernel names: {sorted(kernels)}")
    for name, (params, launch) in EXPECTED.items():
        kernel = kernels[name]
        actual_params = [(param["name"], param["words"])
                         for param in kernel["params"]]
        if actual_params != params:
            raise ValueError(f"{name} parameter ABI mismatch: {actual_params}")
        if kernel.get("launch") != launch:
            raise ValueError(f"{name} launch shape mismatch: {kernel.get('launch')}")
    print("GPU Lloyd PTX manifest matches the smoke-test ABI.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <gpu_lloyd.json>")
    try:
        main(sys.argv[1])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"GPU Lloyd manifest check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
