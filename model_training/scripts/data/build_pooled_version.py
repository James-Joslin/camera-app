#!/usr/bin/env python3
"""Build one immutable CityPersons + CrowdHuman canonical dataset version."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from person_detection.data.annotations import parse_canonical_annotation
from scripts.data.build_citypersons_version import (
    ANNOTATION_COMMIT,
    CONVERSION_VERSION,
    EXPECTED_IMAGE_COUNTS,
    IGNORE_REASONS,
    LOADER_VERSION,
    POSITIVE_CLASS_IDS,
    SCHEMA_VERSION,
    SOURCE_LABELS,
    artifact,
    clipped_yolo_box,
    derived_attributes,
    image_files,
    json_bytes,
    load_annotations,
    numeric,
    png_dimensions,
    sha256_file,
    write_json,
)


def image_dimensions(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".png":
        return png_dimensions(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise RuntimeError(f"Could not decode image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def crowdhuman_image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.name in index:
            raise RuntimeError(f"Duplicate CrowdHuman image filename: {path.name}")
        index[path.name] = path
    if not index:
        raise RuntimeError(f"No CrowdHuman images found under {image_root}")
    return index


def load_odgt(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid CrowdHuman JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"CrowdHuman record at {path}:{line_number} is not an object")
            image_id = record.get("ID")
            if not isinstance(image_id, str) or not image_id:
                raise RuntimeError(f"CrowdHuman record at {path}:{line_number} has no ID")
            records.append(record)
    return records


def crowdhuman_box(
    value: Any, field: str, *, allow_missing_or_invalid: bool = False
) -> list[int | float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        raise RuntimeError(f"CrowdHuman {field} contains a non-finite coordinate")
    if allow_missing_or_invalid and any(item < 0 for item in (x, y, width, height)):
        return None
    if width < 0 or height < 0:
        if allow_missing_or_invalid:
            return None
        raise RuntimeError(f"CrowdHuman {field} contains a negative size")
    return [numeric(item) for item in (x, y, width, height)]


def crowdhuman_bool(value: Any) -> bool:
    return value in (1, True, "1", "true", "True")


def yolo_person_line(
    full_box: list[int | float], image_width: int, image_height: int
) -> str:
    """Use the identical class-0 YOLO body-label format for both sources."""
    box = clipped_yolo_box(full_box, image_width, image_height)
    return "0 " + " ".join(f"{value:.8f}" for value in box)


def build_crowdhuman_records(
    image_root: Path,
    annotation_path: Path,
    output: Path,
    artifacts: list[dict[str, Any]],
    source_label_counts: Counter[str],
    attribute_counts: dict[str, Counter[str]],
    status_counts: Counter[str],
) -> list[dict[str, Any]]:
    images = crowdhuman_image_index(image_root)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_record in load_odgt(annotation_path):
        image_id = str(source_record["ID"])
        image_name = Path(image_id).name
        if image_name in seen_ids:
            raise RuntimeError(f"Duplicate CrowdHuman annotation ID: {image_name}")
        seen_ids.add(image_name)
        image_path = images.get(image_name)
        if image_path is None and Path(image_name).suffix == "":
            image_path = images.get(image_name + ".jpg")
        if image_path is None:
            raise RuntimeError(f"CrowdHuman annotation has no matching image: {image_id}")

        image_blob = f"images/train/crowdhuman/{image_path.name}"
        image_sha256 = sha256_file(image_path)
        image_width, image_height = image_dimensions(image_path)
        artifacts.append({
            "blob": image_blob,
            "kind": "image",
            "size": image_path.stat().st_size,
            "sha256": image_sha256,
        })

        objects: list[dict[str, Any]] = []
        ignore_regions: list[dict[str, Any]] = []
        yolo_lines: list[str] = []
        gtboxes = source_record.get("gtboxes", [])
        if not isinstance(gtboxes, list):
            raise RuntimeError(f"CrowdHuman gtboxes is not an array for {image_id}")
        for box_index, raw_box in enumerate(gtboxes):
            if not isinstance(raw_box, dict):
                raise RuntimeError(f"CrowdHuman gtbox {image_id}:{box_index} is not an object")
            full_box = crowdhuman_box(raw_box.get("fbox"), "fbox")
            if full_box is None or full_box[2] <= 0 or full_box[3] <= 0:
                raise RuntimeError(f"CrowdHuman gtbox {image_id}:{box_index} has no valid fbox")
            extra = raw_box.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            ignored = str(raw_box.get("tag", "person")) == "mask" or crowdhuman_bool(extra.get("ignore"))
            # CrowdHuman uses negative sentinel coordinates for bodies with no
            # visible region. Keep the full-body target and represent missing
            # visible geometry as a zero-sized visible box.
            visible_box = crowdhuman_box(
                raw_box.get("vbox"), "vbox", allow_missing_or_invalid=True
            ) or [0, 0, 0, 0]
            box_id = extra.get("box_id", box_index)
            try:
                source_instance_id = int(box_id)
            except (TypeError, ValueError):
                source_instance_id = box_index

            if ignored:
                ignore_regions.append({
                    "id": f"{image_path.stem}:instance-{source_instance_id}-{box_index}",
                    "sourceClassId": 0,
                    "sourceInstanceId": source_instance_id,
                    "sourceLabel": "crowdhuman ignore",
                    "fullBoxXYWH": full_box,
                    "visibleBoxXYWH": visible_box,
                    "reason": "crowdhuman_ignore",
                })
                continue

            attributes = derived_attributes(1, full_box, visible_box)
            head_attr = raw_box.get("head_attr")
            if isinstance(head_attr, dict):
                attributes["headOccluded"] = crowdhuman_bool(head_attr.get("occ"))
                attributes["headIgnored"] = crowdhuman_bool(head_attr.get("ignore"))
            attributes["sourceDataset"] = "crowdhuman"
            for attribute_name in ("transport", "posture", "visibility"):
                value = attributes[attribute_name]
                attribute_counts[attribute_name]["null" if value is None else str(value)] += 1
            source_label_counts["crowdhuman person"] += 1
            objects.append({
                "id": f"{image_path.stem}:instance-{source_instance_id}-{box_index}",
                "detectionClass": "person",
                # CrowdHuman has one positive body class; map it to the existing pedestrian ID.
                "sourceClassId": 1,
                "sourceInstanceId": source_instance_id,
                "sourceLabel": "pedestrian",
                "fullBoxXYWH": full_box,
                "visibleBoxXYWH": visible_box,
                "ignored": False,
                "attributes": attributes,
            })
            yolo_lines.append(yolo_person_line(full_box, image_width, image_height))

        label_relative = Path("labels/yolo-person-v1") / "train" / "crowdhuman" / f"{image_path.stem}.txt"
        label_path = output / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(f"{line}\n" for line in yolo_lines), encoding="utf-8")
        label_sha256 = sha256_file(label_path)
        artifacts.append(artifact(label_path, label_relative.as_posix(), "generated_yolo_label"))

        annotation_relative = Path("annotations/canonical") / "train" / "crowdhuman" / f"{image_path.stem}.json"
        annotation_path_out = output / annotation_relative
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
        annotation_path_out.parent.mkdir(parents=True, exist_ok=True)
        annotation_path_out.write_bytes(annotation_data)
        annotation_sha256 = sha256_file(annotation_path_out)
        artifacts.append(artifact(annotation_path_out, annotation_relative.as_posix(), "canonical_annotation"))
        status_counts[status] += 1
        records.append({
            "image": image_blob,
            "yoloLabel": label_relative.as_posix(),
            "annotation": annotation_relative.as_posix(),
            "labelStatus": status,
            "personCount": len(objects),
            "ignoredCount": len(ignore_regions),
            "sourceDataset": "crowdhuman",
            "checksums": {
                "image": image_sha256,
                "yoloLabel": label_sha256,
                "annotation": annotation_sha256,
            },
        })
    if len(records) != len(seen_ids):
        raise AssertionError("CrowdHuman record accounting mismatch")
    return records


def build_citypersons_records(
    images: Path,
    annotations: Path,
    output: Path,
    artifacts: list[dict[str, Any]],
    source_label_counts: Counter[str],
    attribute_counts: dict[str, Counter[str]],
    status_counts: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    annotations_by_split = {
        split: load_annotations(annotations / f"anno_{split}.mat", split)
        for split in ("train", "val")
    }
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        records: list[dict[str, Any]] = []
        image_paths = image_files(images, split)
        image_keys = {(path.parent.name, path.name) for path in image_paths}
        if split in annotations_by_split and image_keys != set(annotations_by_split[split]):
            raise RuntimeError(f"{split} CityPersons image/source annotation mismatch")
        for image_path in image_paths:
            city = image_path.parent.name
            image_name = image_path.name
            image_blob = f"images/{split}/{city}/{image_name}"
            image_sha256 = sha256_file(image_path)
            image_width, image_height = png_dimensions(image_path)
            artifacts.append({"blob": image_blob, "kind": "image", "size": image_path.stat().st_size, "sha256": image_sha256})
            if split == "test":
                status_counts["unlabelled"] += 1
                records.append({
                    "image": image_blob, "yoloLabel": None, "annotation": None,
                    "labelStatus": "unlabelled", "personCount": 0, "ignoredCount": 0,
                    "sourceDataset": "citypersons", "checksums": {"image": image_sha256},
                })
                continue

            objects: list[dict[str, Any]] = []
            ignore_regions: list[dict[str, Any]] = []
            yolo_lines: list[str] = []
            for row_index, row in enumerate(annotations_by_split[split][(city, image_name)]):
                if len(row) != 10:
                    raise RuntimeError(f"Unexpected CityPersons annotation width for {split}/{city}/{image_name}")
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
                    for name in attribute_counts:
                        value = attributes[name]
                        attribute_counts[name]["null" if value is None else str(value)] += 1
                    objects.append({
                        "id": f"{image_path.stem}:instance-{source_instance_id}-{row_index}",
                        "detectionClass": "person", "sourceClassId": source_class_id,
                        "sourceInstanceId": source_instance_id, "sourceLabel": source_label,
                        "fullBoxXYWH": full_box, "visibleBoxXYWH": visible_box,
                        "ignored": False, "attributes": attributes,
                    })
                    yolo_lines.append(yolo_person_line(full_box, image_width, image_height))
                else:
                    ignore_regions.append({
                        "id": f"{image_path.stem}:instance-{source_instance_id}-{row_index}",
                        "sourceClassId": source_class_id, "sourceInstanceId": source_instance_id,
                        "sourceLabel": source_label, "fullBoxXYWH": full_box,
                        "visibleBoxXYWH": visible_box, "reason": IGNORE_REASONS[source_class_id],
                    })

            label_relative = Path("labels/yolo-person-v1") / split / city / f"{image_path.stem}.txt"
            label_path = output / label_relative
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("".join(f"{line}\n" for line in yolo_lines), encoding="utf-8")
            label_sha256 = sha256_file(label_path)
            artifacts.append(artifact(label_path, label_relative.as_posix(), "generated_yolo_label"))
            annotation_relative = Path("annotations/canonical") / split / city / f"{image_path.stem}.json"
            annotation_path = output / annotation_relative
            canonical = {
                "schemaVersion": SCHEMA_VERSION,
                "image": {"blob": image_blob, "width": image_width, "height": image_height, "sha256": image_sha256},
                "objects": objects, "ignoreRegions": ignore_regions,
            }
            status = "positive" if objects else "verified_negative"
            annotation_data = json_bytes(canonical)
            parse_canonical_annotation(annotation_data, expected_image_blob=image_blob,
                                       expected_image_sha256=image_sha256, expected_status=status,
                                       expected_person_count=len(objects), expected_ignored_count=len(ignore_regions))
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_bytes(annotation_data)
            annotation_sha256 = sha256_file(annotation_path)
            artifacts.append(artifact(annotation_path, annotation_relative.as_posix(), "canonical_annotation"))
            status_counts[status] += 1
            records.append({
                "image": image_blob, "yoloLabel": label_relative.as_posix(),
                "annotation": annotation_relative.as_posix(), "labelStatus": status,
                "personCount": len(objects), "ignoredCount": len(ignore_regions),
                "sourceDataset": "citypersons",
                "checksums": {"image": image_sha256, "yoloLabel": label_sha256, "annotation": annotation_sha256},
            })
        split_records[split] = records
    return split_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--crowdhuman-images", type=Path, required=True)
    parser.add_argument("--crowdhuman-annotations", type=Path, required=True)
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
    for source_path in (
        args.annotations / "README.txt", args.annotations / "anno_train.mat", args.annotations / "anno_val.mat",
        args.crowdhuman_annotations,
    ):
        if not source_path.is_file():
            parser.error(f"missing annotation source: {source_path}")
        destination = source_dir / source_path.name
        # Keep source names unambiguous if a future source uses the same filename.
        if destination.exists():
            destination = source_dir / f"crowdhuman-{source_path.name}"
        destination.write_bytes(source_path.read_bytes())
        source_artifacts.append(artifact(destination, f"annotations/source/{destination.name}", "source_annotation"))

    artifacts = list(source_artifacts)
    source_label_counts: Counter[str] = Counter()
    attribute_counts: dict[str, Counter[str]] = {"transport": Counter(), "posture": Counter(), "visibility": Counter()}
    status_counts: Counter[str] = Counter()
    split_records = build_citypersons_records(
        args.images, args.annotations, args.output, artifacts, source_label_counts, attribute_counts, status_counts
    )
    split_records["train"].extend(build_crowdhuman_records(
        args.crowdhuman_images, args.crowdhuman_annotations, args.output, artifacts,
        source_label_counts, attribute_counts, status_counts
    ))
    for split, records in split_records.items():
        records.sort(key=lambda record: record["image"])
        split_path = args.output / f"splits/{split}.jsonl"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("wb") as stream:
            for record in records:
                stream.write(json_bytes(record, pretty=False))
        artifacts.append(artifact(split_path, f"splits/{split}.jsonl", "split_manifest"))

    archive_checksums = [{"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
                         for path in args.source_archive if path.is_file()]
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "dataset": "citypersons-crowdhuman",
        "version": args.version,
        "versionPrefix": args.version_prefix,
        "createdAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "minimumLoaderVersion": LOADER_VERSION,
        "checksumAlgorithm": "sha256",
        "source": {
            "citypersons": {"kaggleDataset": "hakurei/citypersons", "archives": archive_checksums,
                            "officialAnnotations": {"repository": "cvgroup-njust/CityPersons", "commit": ANNOTATION_COMMIT}},
            "crowdhuman": {"huggingFaceDataset": "sshao0516/CrowdHuman", "annotation": args.crowdhuman_annotations.name,
                           "bodyBoxField": "fbox", "visibleBoxField": "vbox", "headBoxFieldExcluded": "hbox"},
        },
        "conversion": {"name": "citypersons-crowdhuman-canonical-v1", "codeSha256": sha256_file(Path(__file__)),
                       "boxSource": "fullBoxXYWH/fbox", "coordinates": "absolute source pixels in canonical JSON; normalized cx cy w h in YOLO",
                       "crowdhumanIgnored": ["tag=mask", "extra.ignore=1"]},
        "classMapping": {"person": {"detectionClass": 0, "sources": ["CityPersons positive classes", "CrowdHuman tag=person"]},
                         "ignore": {"detectionClass": None, "sources": ["CityPersons ignored classes", "CrowdHuman tag=mask or extra.ignore=1"]}},
        "counts": {"splits": {split: len(records) for split, records in split_records.items()},
                   "labelStatus": dict(sorted(status_counts.items())),
                   "sourceLabels": dict(sorted(source_label_counts.items())),
                   "attributes": {name: dict(sorted(counts.items())) for name, counts in sorted(attribute_counts.items())}},
        "splitManifests": {split: f"splits/{split}.jsonl" for split in split_records},
        "validation": {"officialCityPersonsSplit": "val", "crowdHumanSplit": "train"},
        "artifacts": sorted(artifacts, key=lambda item: item["blob"]),
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Built {args.version_prefix}: {len(artifacts)} artifacts")
    print(f"Split counts: {manifest['counts']['splits']}")
    print(f"Label statuses: {manifest['counts']['labelStatus']}")


if __name__ == "__main__":
    main()
