#!/usr/bin/env python3
"""Download and verify the pinned official CityPersons annotations."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "839c22fb05a16c150cb77f9b73a5c0e9642af21e"
FILES = {
    "README.txt": "a45c8470e862bcbb7d270bddb9304e47e6620df19e2c55cc7270324cb2c6e7a7",
    "anno_train.mat": "1e9675594e2d30772b40c046a9e3e9b24604d43c20dd93121f35d864e63bdd06",
    "anno_val.mat": "e234f73b47d7e9c0dfa23a91a2bd8bd222b68a5f5fe7061e8bbfefa3435aac96",
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

    for filename, expected_sha256 in FILES.items():
        destination = args.output / filename
        if destination.is_file() and sha256_file(destination) == expected_sha256:
            print(f"Using verified annotation source: {destination}")
            continue

        url = (
            "https://raw.githubusercontent.com/cvgroup-njust/CityPersons/"
            f"{UPSTREAM_COMMIT}/annotations/{filename}"
        )
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

        actual_sha256 = sha256_file(destination)
        if actual_sha256 != expected_sha256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        print(f"Verified {filename}: sha256={actual_sha256}")


if __name__ == "__main__":
    main()
