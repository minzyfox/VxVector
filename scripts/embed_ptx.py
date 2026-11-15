#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("header", type=Path)
    args = parser.parse_args()

    text = args.ptx.read_text(encoding="ascii")
    chunks = [text[index:index + 100] for index in range(0, len(text), 100)]
    output = [
        "#pragma once",
        "static constexpr char vxvector_gpu_lloyd_ptx[] =",
        *(f"    {json.dumps(chunk)}" for chunk in chunks),
        "    ;",
        "",
    ]
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text("\n".join(output), encoding="ascii")


if __name__ == "__main__":
    main()
