#!/usr/bin/env python3
"""Build canonical CityPersons metadata, split manifests, and generated YOLO labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from person_detection.data.annotations import parse_canonical_annotation


SCHEMA_VERSION = 1
LOADER_VERSION = 1
CONVERSION_VERSION = "citypersons-canonical-v1"
EXPECTED_IMAGE_COUNTS = {"train": 2975, "val": 500, "test": 1525}
SOURCE_LABELS = {
    0: "ignore",
    1: "pedestrian",
    2: "rider",
    3: "sitting person",
    4: "person (other)",
    5: "person group",
}
POSITIVE_CLASS_IDS = {1, 2, 3, 4}
IGNORE_REASONS = {0: "fake_human_or_reflection", 5: "group"}
ANNOTATION_COMMIT = "839c22fb05a16c150cb77f9b73a5c0e9642af21e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    return (
        json.dumps(value, indent=indent, separators=separators, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value, pretty=pretty))


def numeric(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def rows_for(entry: Any) -> np.ndarray:
    rows = np.asarray(entry.bbs)
    if rows.size == 0:
        return np.empty((0, 10))
    if rows.ndim == 1:
        return rows.reshape(1, -1)
    return rows


def load_annotations(path: Path, split: str) -> dict[tuple[str, str], np.ndarray]:
    variable = f"anno_{split}_aligned"
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    if variable not in data:
        raise RuntimeError(f"{path} does not contain {variable}")

    annotations: dict[tuple[str, str], np.ndarray] = {}
    for entry in data[variable]:
        key = (str(entry.cityname), str(entry.im_name))
        if key in annotations:
            raise RuntimeError(f"Duplicate source annotation: {key}")
        annotations[key] = rows_for(entry)
    return annotations


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Expected a PNG image: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def clipped_yolo_box(
    box: list[int | float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError(f"Non-finite source box: {box}")
    x1 = min(max(x, 0.0), image_width)
    y1 = min(max(y, 0.0), image_height)
    x2 = min(max(x + width, 0.0), image_width)
    y2 = min(max(y + height, 0.0), image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Source box is outside the image: {box}")
    return (
        (x1 + x2) / (2 * image_width),
        (y1 + y2) / (2 * image_height),
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def derived_attributes(
    source_class_id: int,
    full_box: list[int | float],
    visible_box: list[int | float],
) -> dict[str, Any]:
    full_area = float(full_box[2]) * float(full_box[3])
    visible_area = float(visible_box[2]) * float(visible_box[3])
    occlusion = None
    visibility = "unknown"
    if full_area > 0:
        visible_fraction = min(max(visible_area / full_area, 0.0), 1.0)
        occlusion = round(1.0 - visible_fraction, 6)
        if visible_fraction >= 0.9:
            visibility = "fully_visible"
        elif visible_fraction >= 0.35:
            visibility = "partially_occluded"
        else:
            visibility = "heavily_occluded"

    posture = None
    if source_class_id == 3:
        posture = "sitting"
    elif source_class_id == 4:
        posture = "unusual"

    return {
        "transport": "unknown_rider" if source_class_id == 2 else None,
        "posture": posture,
        "occlusion": occlusion,
        "visibility": visibility,
    }


def artifact(path: Path, blob: str, kind: str) -> dict[str, Any]:
    return {
        "blob": blob,
        "kind": kind,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def image_files(image_root: Path, split: str) -> list[Path]:
    split_root = image_root / split
    files = sorted(
        path
        for path in split_root.rglob("*.png")
        if path.is_file() and "(1)" not in path.name
    )
    expected = EXPECTED_IMAGE_COUNTS[split]
    if len(files) != expected:
        raise RuntimeError(f"{split_root} has {len(files)} PNGs; expected {expected}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--version-prefix", required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_dir = args.output / "annotations/source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_artifacts = []
    for filename in ("README.txt", "anno_train.mat", "anno_val.mat"):
        source_path = args.annotations / filename
        if not source_path.is_file():
            parser.error(f"missing official annotation source: {source_path}")
        destination = source_dir / filename
        shutil.copy2(source_path, destination)
        source_artifacts.append(
            artifact(destination, f"annotations/source/{filename}", "source_annotation")
        )

    annotations_by_split = {
        split: load_annotations(args.annotations / f"anno_{split}.mat", split)
        for split in ("train", "val")
    }
    split_records: dict[str, list[dict[str, Any]]] = {}
    artifacts: list[dict[str, Any]] = list(source_artifacts)
    source_label_counts: Counter[str] = Counter()
    attribute_counts: dict[str, Counter[str]] = {
        "transport": Counter(),
        "posture": Counter(),
        "visibility": Counter(),
    }
    status_counts: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        records = []
        images = image_files(args.images, split)
        image_keys = {(path.parent.name, path.name) for path in images}
        if split in annotations_by_split:
            annotation_keys = set(annotations_by_split[split])
            if image_keys != annotation_keys:
                missing = sorted(image_keys - annotation_keys)
                orphaned = sorted(annotation_keys - image_keys)
                raise RuntimeError(
                    f"{split} image/source mismatch: missing={missing[:1]}, orphaned={orphaned[:1]}"
                )

        for image_path in images:
            city = image_path.parent.name
            image_name = image_path.name
            image_blob = f"images/{split}/{city}/{image_name}"
            image_sha256 = sha256_file(image_path)
            image_width, image_height = png_dimensions(image_path)
            artifacts.append(
                {
                    "blob": image_blob,
                    "kind": "image",
                    "size": image_path.stat().st_size,
                    "sha256": image_sha256,
                }
            )

            if split == "test":
                status = "unlabelled"
                status_counts[status] += 1
                records.append(
                    {
                        "image": image_blob,
                        "yoloLabel": None,
                        "annotation": None,
                        "labelStatus": status,
                        "personCount": 0,
                        "ignoredCount": 0,
                        "checksums": {"image": image_sha256},
                    }
                )
                continue

            objects = []
            ignore_regions = []
            yolo_lines = []
            rows = annotations_by_split[split][(city, image_name)]
            for row_index, row in enumerate(rows):
                if len(row) != 10:
                    raise RuntimeError(f"Unexpected annotation width for {split}/{city}/{image_name}")
                source_class_id = int(row[0])
                if source_class_id not in SOURCE_LABELS:
                    raise RuntimeError(f"Unknown CityPersons class ID: {source_class_id}")
                source_label = SOURCE_LABELS[source_class_id]
                source_label_counts[source_label] += 1
                full_box = [numeric(value) for value in row[1:5]]
                source_instance_id = int(row[5])
                visible_box = [numeric(value) for value in row[6:10]]

                if source_class_id in POSITIVE_CLASS_IDS:
                    attributes = derived_attributes(source_class_id, full_box, visible_box)
                    for attribute_name in attribute_counts:
                        value = attributes[attribute_name]
                        attribute_counts[attribute_name]["null" if value is None else str(value)] += 1
                    objects.append(
                        {
                            "id": f"{image_path.stem}:instance-{source_instance_id}-{row_index}",
                            "detectionClass": "person",
                            "sourceClassId": source_class_id,
                            "sourceInstanceId": source_instance_id,
                            "sourceLabel": source_label,
                            "fullBoxXYWH": full_box,
                            "visibleBoxXYWH": visible_box,
                            "ignored": False,
                            "attributes": attributes,
                        }
                    )
                    yolo_box = clipped_yolo_box(full_box, image_width, image_height)
                    yolo_lines.append("0 " + " ".join(f"{value:.8f}" for value in yolo_box))
                else:
                    ignore_regions.append(
                        {
                            "id": f"{image_path.stem}:instance-{source_instance_id}-{row_index}",
                            "sourceClassId": source_class_id,
                            "sourceInstanceId": source_instance_id,
                            "sourceLabel": source_label,
                            "fullBoxXYWH": full_box,
                            "visibleBoxXYWH": visible_box,
                            "reason": IGNORE_REASONS[source_class_id],
                        }
                    )

            label_relative = Path("labels/yolo-person-v1") / split / city / f"{image_path.stem}.txt"
            label_path = args.output / label_relative
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("".join(f"{line}\n" for line in yolo_lines), encoding="utf-8")
            label_sha256 = sha256_file(label_path)
            artifacts.append(
                artifact(label_path, label_relative.as_posix(), "generated_yolo_label")
            )

            annotation_relative = Path("annotations/canonical") / split / city / f"{image_path.stem}.json"
            annotation_path = args.output / annotation_relative
            canonical = {
                "schemaVersion": SCHEMA_VERSION,
                "image": {
                    "blob": image_blob,
                    "width": image_width,
                    "height": image_height,
                    "sha256": image_sha256,
                },
                "objects": objects,
                "ignoreRegions": ignore_regions,
            }
            status = "positive" if objects else "verified_negative"
            annotation_data = json_bytes(canonical)
            parse_canonical_annotation(
                annotation_data,
                expected_image_blob=image_blob,
                expected_image_sha256=image_sha256,
                expected_status=status,
                expected_person_count=len(objects),
                expected_ignored_count=len(ignore_regions),
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_bytes(annotation_data)
            annotation_sha256 = sha256_file(annotation_path)
            artifacts.append(
                artifact(annotation_path, annotation_relative.as_posix(), "canonical_annotation")
            )

            status_counts[status] += 1
            records.append(
                {
                    "image": image_blob,
                    "yoloLabel": label_relative.as_posix(),
                    "annotation": annotation_relative.as_posix(),
                    "labelStatus": status,
                    "personCount": len(objects),
                    "ignoredCount": len(ignore_regions),
                    "checksums": {
                        "image": image_sha256,
                        "yoloLabel": label_sha256,
                        "annotation": annotation_sha256,
                    },
                }
            )

        split_records[split] = records
        split_path = args.output / f"splits/{split}.jsonl"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("wb") as stream:
            for record in records:
                stream.write(json_bytes(record, pretty=False))
        artifacts.append(artifact(split_path, f"splits/{split}.jsonl", "split_manifest"))

    archive_checksums = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in args.source_archive
        if path.is_file()
    ]
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "dataset": "citypersons",
        "version": args.version,
        "versionPrefix": args.version_prefix,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "minimumLoaderVersion": LOADER_VERSION,
        "checksumAlgorithm": "sha256",
        "source": {
            "kaggleDataset": "hakurei/citypersons",
            "archives": archive_checksums,
            "officialAnnotations": {
                "repository": "cvgroup-njust/CityPersons",
                "commit": ANNOTATION_COMMIT,
            },
        },
        "conversion": {
            "name": CONVERSION_VERSION,
            "codeSha256": sha256_file(Path(__file__)),
            "boxSource": "fullBoxXYWH",
            "coordinates": "absolute source pixels in canonical JSON; normalized cx cy w h in YOLO",
            "visibilityThresholds": {
                "fullyVisibleMinimum": 0.9,
                "partiallyOccludedMinimum": 0.35,
            },
        },
        "classMapping": {
            "ignore": {"sourceClassId": 0, "ignore": True},
            "pedestrian": {"sourceClassId": 1, "detectionClass": 0},
            "rider": {"sourceClassId": 2, "detectionClass": 0},
            "sitting person": {"sourceClassId": 3, "detectionClass": 0},
            "person (other)": {"sourceClassId": 4, "detectionClass": 0},
            "person group": {"sourceClassId": 5, "ignore": True},
        },
        "counts": {
            "splits": {split: len(records) for split, records in split_records.items()},
            "labelStatus": dict(sorted(status_counts.items())),
            "sourceLabels": dict(sorted(source_label_counts.items())),
            "attributes": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(attribute_counts.items())
            },
        },
        "splitManifests": {
            split: f"splits/{split}.jsonl" for split in split_records
        },
        "artifacts": sorted(artifacts, key=lambda item: item["blob"]),
    }
    write_json(args.output / "manifest.json", manifest)
    print(
        f"Built {args.version_prefix}: {len(artifacts)} artifacts, "
        f"{sum(source_label_counts.values())} source annotations"
    )
    print(f"Label statuses: {dict(sorted(status_counts.items()))}")


if __name__ == "__main__":
    main()
