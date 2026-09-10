"""Publish and verify an immutable person-detector release in Azurite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from azure.core.exceptions import ResourceExistsError

from scripts.data.upload_to_azurite import client


REQUIRED_ARTIFACTS = (
    "checkpoint/best_model_fp32.pth",
    "models/person_detector_fp32.xml",
    "models/person_detector_fp32.bin",
    "models/person_detector_fp16.xml",
    "models/person_detector_fp16.bin",
    "models/person_detector_int8.xml",
    "models/person_detector_int8.bin",
    "models/calibration_manifest.json",
    "models/optimization_report.json",
    "evaluation/fp32/evaluation_metrics.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(
    release_dir: Path,
    *,
    release_id: str,
    container_name: str,
    prefix: str,
) -> dict[str, Any]:
    missing = [
        relative_name
        for relative_name in REQUIRED_ARTIFACTS
        if not (release_dir / relative_name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Release is missing required artifacts: {missing}")

    optimization_report = json.loads(
        (release_dir / "models/optimization_report.json").read_text(encoding="utf-8")
    )
    accuracy_control = optimization_report.get("accuracyControl")
    if not isinstance(accuracy_control, dict) or accuracy_control.get("accepted") is not True:
        raise RuntimeError("INT8 optimization was not accepted; refusing to publish release")

    artifacts = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file() and path.name != "release_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(release_dir).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    return {
        "schemaVersion": 1,
        "releaseId": release_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "container": container_name,
            "prefix": prefix,
        },
        "checkpoint": optimization_report.get("checkpoint"),
        "dataset": optimization_report.get("dataset"),
        "preprocessing": optimization_report.get("preprocessing"),
        "accuracyControl": accuracy_control,
        "artifacts": artifacts,
    }


def remote_sha256(container, blob_name: str) -> str:
    digest = hashlib.sha256()
    for chunk in container.download_blob(blob_name).chunks():
        digest.update(chunk)
    return digest.hexdigest()


def publish_release(
    release_dir: Path,
    *,
    release_id: str,
    container_name: str,
    prefix: str,
) -> None:
    prefix_path = PurePosixPath(prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise ValueError("prefix must be a relative blob path without '..'")

    manifest = build_release_manifest(
        release_dir,
        release_id=release_id,
        container_name=container_name,
        prefix=prefix,
    )
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    service = client()
    container = service.get_container_client(container_name)
    try:
        container.create_container()
    except ResourceExistsError:
        pass

    guard_prefix = f"{prefix.rstrip('/')}/"
    existing = next(container.list_blobs(name_starts_with=guard_prefix), None)
    if existing is not None:
        raise RuntimeError(
            f"Refusing to overwrite immutable release prefix {prefix!r}; "
            f"existing blob: {existing.name}"
        )

    uploaded: list[tuple[Path, str]] = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_name = path.relative_to(release_dir).as_posix()
        blob_name = f"{prefix.rstrip('/')}/{relative_name}"
        with path.open("rb") as stream:
            container.upload_blob(name=blob_name, data=stream, overwrite=False)
        uploaded.append((path, blob_name))
        print(f"Uploaded {relative_name} -> {container_name}/{blob_name}")

    for path, blob_name in uploaded:
        expected = sha256_file(path)
        actual = remote_sha256(container, blob_name)
        if actual != expected:
            raise RuntimeError(f"Remote checksum mismatch for {container_name}/{blob_name}")

    print(f"Verified {len(uploaded)} uploaded release artifacts")
    print(f"Azurite release: {container_name}/{prefix.rstrip('/')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--container", default="computer-vision-models")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    if not args.release_dir.is_dir():
        parser.error(f"release directory does not exist: {args.release_dir}")
    publish_release(
        args.release_dir,
        release_id=args.release_id,
        container_name=args.container,
        prefix=args.prefix.strip("/"),
    )


if __name__ == "__main__":
    main()
