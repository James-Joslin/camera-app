#!/usr/bin/env python3
"""Download and verify the pinned official CityPersons annotations."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "839c22fb05a16c150cb77f9b73a5c0e9642af21e"
FILES = {
    "README.txt": (
        "annotations/README.txt", "a45c8470e862bcbb7d270bddb9304e47e6620df19e2c55cc7270324cb2c6e7a7"
    ),
    "anno_train.mat": (
        "annotations/anno_train.mat", "1e9675594e2d30772b40c046a9e3e9b24604d43c20dd93121f35d864e63bdd06"
    ),
    "anno_val.mat": (
        "annotations/anno_val.mat", "e234f73b47d7e9c0dfa23a91a2bd8bd222b68a5f5fe7061e8bbfefa3435aac96"
    ),
    "evaluation/eval_script/coco.py": (
        "evaluation/eval_script/coco.py", "6958010a2e01881b139b23f652a1b2dd9efdb2ee13d3fa41e401946386499918"
    ),
    "evaluation/eval_script/eval_MR_multisetup.py": (
        "evaluation/eval_script/eval_MR_multisetup.py", "b91887ead3999b7616766e2093870647bf7ceb2cfbba0b6f43dad8f0887c89a4"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for relative_name, (upstream_name, expected_sha256) in FILES.items():
        destination = args.output / relative_name
        if destination.is_file() and sha256_file(destination) == expected_sha256:
            print(f"Using verified official source: {destination}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/cvgroup-njust/CityPersons/"
            f"{UPSTREAM_COMMIT}/{upstream_name}"
        )
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

        actual_sha256 = sha256_file(destination)
        if actual_sha256 != expected_sha256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {relative_name}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        print(f"Verified {relative_name}: sha256={actual_sha256}")


if __name__ == "__main__":
    main()
