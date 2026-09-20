#!/usr/bin/env python3
"""Download the CrowdHuman files used by the pooled training dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main"
FILES = {
    "annotation_train.odgt": "annotation_train.odgt",
    "CrowdHuman_train01.zip": "CrowdHuman_train01.zip",
    "CrowdHuman_train02.zip": "CrowdHuman_train02.zip",
    "CrowdHuman_train03.zip": "CrowdHuman_train03.zip",
}


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using existing CrowdHuman download: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "camera-software-dataset-ingest/1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading {url} -> {destination} (resume={existing})")
    try:
        with urllib.request.urlopen(request) as response, partial.open("ab" if existing else "wb") as output:
            if existing and response.status != 206:
                output.close()
                partial.unlink()
                return download(url, destination)
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        print(f"Partial download preserved for resume: {partial}", file=sys.stderr)
        raise
    os.replace(partial, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, remote_name in FILES.items():
        download(f"{BASE_URL}/{remote_name}?download=true", args.output / filename)


if __name__ == "__main__":
    main()
