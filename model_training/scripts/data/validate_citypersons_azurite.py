#!/usr/bin/env python3
"""Validate a versioned CityPersons dataset, render previews, and optionally publish it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient, ContainerClient


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from person_detection.data.annotations import parse_canonical_annotation


LOADER_VERSION = 1
EXPECTED_SPLIT_COUNTS = {"train": 2975, "val": 500, "test": 1525}
TRAINABLE_STATUSES = {"positive", "verified_negative"}


def blob_bytes(container: ContainerClient, name: str) -> bytes:
    return container.download_blob(name).readall()


def load_json_blob(container: ContainerClient, name: str) -> dict[str, Any]:
    try:
        return json.loads(blob_bytes(container, name))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON blob: {name}") from exc


def version_blob(prefix: str, relative_name: str) -> str:
    return f"{prefix.rstrip('/')}/{relative_name.lstrip('/')}"


def resolve_version_prefix(container: ContainerClient, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix.strip("/")
    pointer = load_json_blob(container, "datasets/citypersons/current.json")
    prefix = pointer.get("versionPrefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("datasets/citypersons/current.json has no versionPrefix")
    return prefix.strip("/")


def parse_jsonl(content: bytes, name: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}:{line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{name}:{line_number} is not a JSON object")
        records.append(value)
    return records


def parse_yolo_label(name: str, content: bytes) -> list[tuple[float, float, float, float]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: label is not UTF-8 text") from exc

    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{name}:{line_number}: expected 5 YOLO fields")
        try:
            class_id, center_x, center_y, width, height = map(float, parts)
        except ValueError as exc:
            raise ValueError(f"{name}:{line_number}: contains a non-numeric field") from exc
        values = (class_id, center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name}:{line_number}: contains a non-finite value")
        if class_id != 0:
            raise ValueError(f"{name}:{line_number}: expected class 0")
        if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
            raise ValueError(f"{name}:{line_number}: center is outside [0, 1]")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError(f"{name}:{line_number}: size is outside (0, 1]")
        boxes.append((center_x, center_y, width, height))
    return boxes


def remote_sha256(container: ContainerClient, name: str) -> str:
    digest = hashlib.sha256()
    for chunk in container.download_blob(name).chunks():
        digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(
    container: ContainerClient,
    prefix: str,
    manifest: dict[str, Any],
    verify_checksums: bool,
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest.json has no artifacts")

    expected: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        relative_name = item.get("blob")
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError("manifest.json contains an artifact without a blob name")
        full_name = version_blob(prefix, relative_name)
        if full_name in expected:
            raise ValueError(f"manifest.json contains duplicate artifact {relative_name}")
        expected[full_name] = item

    manifest_name = version_blob(prefix, "manifest.json")
    actual = {
        blob.name: blob
        for blob in container.list_blobs(name_starts_with=f"{prefix}/")
    }
    expected_names = set(expected) | {manifest_name}
    missing = sorted(expected_names - set(actual))
    unexpected = sorted(set(actual) - expected_names)
    if missing or unexpected:
        raise RuntimeError(
            f"artifact set mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    for name, item in expected.items():
        expected_size = item.get("size")
        if expected_size != actual[name].size:
            raise RuntimeError(
                f"size mismatch for {name}: expected {expected_size}, got {actual[name].size}"
            )
    print(f"Validated presence and sizes for {len(expected)} version artifacts")

    if not verify_checksums:
        print("Remote checksum verification skipped")
        return

    def verify(item: tuple[str, dict[str, Any]]) -> None:
        name, metadata = item
        actual_sha256 = remote_sha256(container, name)
        if actual_sha256 != metadata.get("sha256"):
            raise RuntimeError(f"sha256 mismatch for {name}")

    with ThreadPoolExecutor(max_workers=12) as executor:
        for completed, _ in enumerate(executor.map(verify, expected.items()), start=1):
            if completed % 1000 == 0:
                print(f"Verified checksums: {completed}/{len(expected)}")
    print(f"Verified sha256 checksums for all {len(expected)} artifacts")


def validate_records(
    container: ContainerClient,
    prefix: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[tuple[float, float, float, float]]]]:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    boxes_by_label: dict[str, list[tuple[float, float, float, float]]] = {}
    status_counts: Counter[str] = Counter()
    seen_images: set[str] = set()

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        relative_manifest = manifest.get("splitManifests", {}).get(split)
        if not isinstance(relative_manifest, str):
            raise ValueError(f"manifest.json has no split manifest for {split}")
        name = version_blob(prefix, relative_manifest)
        records = parse_jsonl(blob_bytes(container, name), name)
        if len(records) != expected_count:
            raise RuntimeError(f"{name} has {len(records)} records; expected {expected_count}")
        records_by_split[split] = records
        print(f"Validated {split} split manifest: {len(records)} records")

        for record in records:
            image = record.get("image")
            if not isinstance(image, str) or not image.startswith(f"images/{split}/"):
                raise ValueError(f"Invalid {split} image path in {name}: {image!r}")
            if image in seen_images:
                raise ValueError(f"Image occurs in multiple split records: {image}")
            seen_images.add(image)

            status = record.get("labelStatus")
            status_counts[str(status)] += 1
            label = record.get("yoloLabel")
            annotation = record.get("annotation")
            if split == "test":
                if status != "unlabelled" or label is not None or annotation is not None:
                    raise ValueError(f"Test record is not explicitly unlabelled: {image}")
                continue

            if status not in TRAINABLE_STATUSES:
                raise ValueError(f"Non-trainable status in {split}: {status!r}")
            if not isinstance(label, str) or not isinstance(annotation, str):
                raise ValueError(f"Missing label or canonical annotation for {image}")

            full_label_name = version_blob(prefix, label)
            boxes = parse_yolo_label(full_label_name, blob_bytes(container, full_label_name))
            boxes_by_label[label] = boxes
            canonical_name = version_blob(prefix, annotation)
            checksums = record.get("checksums")
            if not isinstance(checksums, dict):
                raise ValueError(f"Missing checksums for {image}")
            image_sha256 = checksums.get("image")
            annotation_sha256 = checksums.get("annotation")
            if not isinstance(image_sha256, str) or len(image_sha256) != 64:
                raise ValueError(f"Invalid image checksum for {image}")
            if not isinstance(annotation_sha256, str) or len(annotation_sha256) != 64:
                raise ValueError(f"Invalid annotation checksum for {image}")
            person_count = record.get("personCount")
            ignored_count = record.get("ignoredCount")
            if not isinstance(person_count, int) or person_count < 0:
                raise ValueError(f"Invalid personCount for {image}")
            if not isinstance(ignored_count, int) or ignored_count < 0:
                raise ValueError(f"Invalid ignoredCount for {image}")
            parse_canonical_annotation(
                blob_bytes(container, canonical_name),
                expected_image_blob=image,
                expected_image_sha256=image_sha256,
                expected_status=status,
                expected_sidecar_sha256=annotation_sha256,
                expected_person_count=person_count,
                expected_ignored_count=ignored_count,
            )
            if len(boxes) != person_count:
                raise ValueError(f"YOLO box count differs from canonical sidecar for {image}")
            if status == "positive" and not boxes:
                raise ValueError(f"Positive record has an empty YOLO label: {image}")
            if status == "verified_negative" and boxes:
                raise ValueError(f"Verified negative has YOLO boxes: {image}")

    expected_status_counts = manifest.get("counts", {}).get("labelStatus")
    if dict(sorted(status_counts.items())) != expected_status_counts:
        raise RuntimeError(
            f"labelStatus counts differ from manifest: {dict(status_counts)} != "
            f"{expected_status_counts}"
        )
    print(
        f"Validated canonical and YOLO semantics: {sum(map(len, boxes_by_label.values()))} boxes"
    )
    return records_by_split, boxes_by_label


def draw_boxes(
    image: np.ndarray, boxes: list[tuple[float, float, float, float]]
) -> np.ndarray:
    height, width = image.shape[:2]
    for center_x, center_y, box_width, box_height in boxes:
        x1 = max(0, int((center_x - box_width / 2) * width))
        y1 = max(0, int((center_y - box_height / 2) * height))
        x2 = min(width - 1, int((center_x + box_width / 2) * width))
        y2 = min(height - 1, int((center_y + box_height / 2) * height))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image


def render_previews(
    container: ContainerClient,
    prefix: str,
    validation_records: list[dict[str, Any]],
    boxes_by_label: dict[str, list[tuple[float, float, float, float]]],
    output_dir: Path,
    preview_count: int,
    seed: int,
) -> None:
    candidates = [
        record
        for record in validation_records
        if record["labelStatus"] == "positive" and boxes_by_label[record["yoloLabel"]]
    ]
    selected = random.Random(seed).sample(candidates, min(preview_count, len(candidates)))
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails: list[np.ndarray] = []

    for index, record in enumerate(selected, start=1):
        image_name = version_blob(prefix, record["image"])
        encoded_image = blob_bytes(container, image_name)
        image = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {image_name}")
        boxes = boxes_by_label[record["yoloLabel"]]
        draw_boxes(image, boxes)
        filename = record["image"].removeprefix("images/").replace("/", "_")
        output_path = output_dir / f"{index:02d}_{filename}"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write preview {output_path}")

        thumbnail_width = 640
        thumbnail_height = round(image.shape[0] * thumbnail_width / image.shape[1])
        thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height))
        cv2.rectangle(thumbnail, (0, 0), (thumbnail_width, 32), (0, 0, 0), -1)
        cv2.putText(
            thumbnail,
            f"{filename} ({len(boxes)} boxes)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbnails.append(thumbnail)
        print(f"Preview: {output_path} ({len(boxes)} boxes)")

    if not thumbnails:
        raise RuntimeError("No positive validation images are available for previews")
    columns = 2
    rows = []
    blank = np.zeros_like(thumbnails[0])
    for start in range(0, len(thumbnails), columns):
        row = thumbnails[start : start + columns]
        row.extend(blank.copy() for _ in range(columns - len(row)))
        rows.append(np.hstack(row))
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet_path), np.vstack(rows)):
        raise RuntimeError(f"Could not write {contact_sheet_path}")
    print(f"Contact sheet: {contact_sheet_path}")


def publish_pointer(
    container: ContainerClient, prefix: str, manifest: dict[str, Any]
) -> None:
    pointer = {
        "schemaVersion": 1,
        "dataset": "citypersons",
        "version": manifest["version"],
        "versionPrefix": prefix,
        "manifest": version_blob(prefix, "manifest.json"),
        "publishedAt": datetime.now(timezone.utc).isoformat(),
    }
    container.upload_blob(
        name="datasets/citypersons/current.json",
        data=(json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        overwrite=True,
    )
    print(f"Published pointer: datasets/citypersons/current.json -> {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="computer-vision-data")
    parser.add_argument("--dataset-prefix")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "validation_preview",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--publish-current", action="store_true")
    args = parser.parse_args()
    if args.preview_count < 1:
        parser.error("preview-count must be at least 1")

    connection_string = os.environ.get("AZURITE_CONNECTION_STRING")
    if not connection_string:
        parser.error("AZURITE_CONNECTION_STRING is required")
    service = BlobServiceClient.from_connection_string(connection_string)
    container = service.get_container_client(args.container)

    try:
        prefix = resolve_version_prefix(container, args.dataset_prefix)
        manifest_name = version_blob(prefix, "manifest.json")
        manifest = load_json_blob(container, manifest_name)
        if manifest.get("versionPrefix") != prefix:
            raise ValueError(f"{manifest_name} versionPrefix does not match {prefix}")
        if manifest.get("minimumLoaderVersion", 0) > LOADER_VERSION:
            raise ValueError(
                f"Dataset requires loader {manifest['minimumLoaderVersion']}; "
                f"validator supports {LOADER_VERSION}"
            )
        validate_artifacts(container, prefix, manifest, args.verify_checksums)
        records, boxes = validate_records(container, prefix, manifest)
        render_previews(
            container,
            prefix,
            records["val"],
            boxes,
            args.preview_dir,
            args.preview_count,
            args.seed,
        )
        if args.publish_current:
            publish_pointer(container, prefix, manifest)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    print(f"CityPersons version validation passed: {prefix}")


if __name__ == "__main__":
    main()
