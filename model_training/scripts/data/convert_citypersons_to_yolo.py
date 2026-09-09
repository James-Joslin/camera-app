#!/usr/bin/env python3
"""Convert CityPersons per-image JSON annotations to YOLO text labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ANNOTATION_SUFFIX = "_gtBboxCityPersons.json"
IMAGE_SUFFIX = "_leftImg8bit"
POSITIVE_LABELS = {
    "pedestrian",
    "rider",
    "sitting person",
    "person (other)",
}


def yolo_box(
    bbox: Any, image_width: float, image_height: float
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    x_min = min(max(x, 0.0), image_width)
    y_min = min(max(y, 0.0), image_height)
    x_max = min(max(x + width, 0.0), image_width)
    y_max = min(max(y + height, 0.0), image_height)
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return (
        (x_min + x_max) / (2.0 * image_width),
        (y_min + y_max) / (2.0 * image_height),
        clipped_width / image_width,
        clipped_height / image_height,
    )


def convert_file(annotation_path: Path, output_path: Path) -> tuple[int, Counter[str]]:
    with annotation_path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)

    try:
        image_width = float(annotation["imgWidth"])
        image_height = float(annotation["imgHeight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid image dimensions in {annotation_path}") from exc

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"non-positive image dimensions in {annotation_path}")

    label_counts: Counter[str] = Counter()
    lines: list[str] = []
    for obj in annotation.get("objects", []):
        label = str(obj.get("label", ""))
        label_counts[label] += 1
        if label not in POSITIVE_LABELS:
            continue

        box = yolo_box(obj.get("bbox"), image_width, image_height)
        if box is None:
            continue
        lines.append("0 " + " ".join(f"{value:.8f}" for value in box))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(lines), label_counts


def output_path_for(annotation_path: Path, annotation_root: Path, output_root: Path) -> Path:
    relative_path = annotation_path.relative_to(annotation_root)
    if not relative_path.name.endswith(ANNOTATION_SUFFIX):
        raise ValueError(f"unexpected CityPersons annotation name: {annotation_path}")
    image_stem = relative_path.name[: -len(ANNOTATION_SUFFIX)] + IMAGE_SUFFIX
    return output_root / relative_path.parent / f"{image_stem}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert gtBboxCityPersons JSON annotations to binary-person YOLO labels"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.annotations.is_dir():
        parser.error(f"annotations is not a directory: {args.annotations}")

    annotation_paths = sorted(args.annotations.rglob(f"*{ANNOTATION_SUFFIX}"))
    if not annotation_paths:
        parser.error(f"no CityPersons JSON annotations found under {args.annotations}")

    total_boxes = 0
    all_label_counts: Counter[str] = Counter()
    for annotation_path in annotation_paths:
        output_path = output_path_for(annotation_path, args.annotations, args.output)
        box_count, label_counts = convert_file(annotation_path, output_path)
        total_boxes += box_count
        all_label_counts.update(label_counts)

    print(
        f"Converted {len(annotation_paths)} annotation files with "
        f"{total_boxes} person boxes to {args.output}"
    )
    print(
        "Source labels: "
        + ", ".join(f"{label or '<empty>'}={count}" for label, count in sorted(all_label_counts.items()))
    )


if __name__ == "__main__":
    main()
