"""Strict parsing helpers for canonical CityPersons annotation sidecars."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any


CANONICAL_SCHEMA_VERSION = 1
PERSON_SOURCE_LABELS = {
    1: "pedestrian",
    2: "rider",
    3: "sitting person",
    4: "person (other)",
}


@dataclass(frozen=True)
class CanonicalObject:
    object_id: str
    full_box: list[float]
    visible_box: list[float]
    source_class_id: int
    source_label: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class CanonicalAnnotation:
    image_blob: str
    width: int
    height: int
    image_sha256: str
    objects: list[CanonicalObject]
    ignore_regions: list[list[float]]


def _xywh_to_xyxy(
    value: Any,
    field: str,
    width: int,
    height: int,
    *,
    allow_zero_size: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be a four-element XYWH array")
    if any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError(f"{field} contains a non-finite or non-numeric coordinate")
    x, y, box_width, box_height = map(float, value)
    if box_width < 0 or box_height < 0:
        raise ValueError(f"{field} must not have negative width or height")
    if not allow_zero_size and (box_width == 0 or box_height == 0):
        raise ValueError(f"{field} must have positive width and height")
    return [x, y, x + box_width, y + box_height]


def parse_canonical_annotation(
    data: bytes | None,
    *,
    expected_image_blob: str,
    expected_image_sha256: str | None = None,
    expected_status: str | None = None,
    expected_sidecar_sha256: str | None = None,
    expected_person_count: int | None = None,
    expected_ignored_count: int | None = None,
) -> CanonicalAnnotation:
    """Parse a sidecar and reject schema, class, image, and status mismatches."""
    if data is None:
        raise RuntimeError(f"Canonical annotation is missing for {expected_image_blob}")
    if (expected_sidecar_sha256 is not None and
            hashlib.sha256(data).hexdigest() != expected_sidecar_sha256):
        raise ValueError(
            f"Canonical annotation checksum does not match split manifest for {expected_image_blob}"
        )
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Canonical annotation is not valid JSON for {expected_image_blob}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Canonical annotation must be a JSON object")
    if payload.get("schemaVersion") != CANONICAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported canonical schemaVersion {payload.get('schemaVersion')!r}; "
            f"expected {CANONICAL_SCHEMA_VERSION}"
        )

    image = payload.get("image")
    if not isinstance(image, dict) or image.get("blob") != expected_image_blob:
        raise ValueError(f"Canonical image blob does not match {expected_image_blob}")
    width, height, image_sha256 = image.get("width"), image.get("height"), image.get("sha256")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise ValueError("Canonical image width and height must be positive integers")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("Canonical image sha256 must be a 64-character string")
    if expected_image_sha256 is not None and image_sha256 != expected_image_sha256:
        raise ValueError(f"Canonical image checksum does not match split manifest for {expected_image_blob}")

    raw_objects = payload.get("objects")
    raw_ignore_regions = payload.get("ignoreRegions")
    if not isinstance(raw_objects, list) or not isinstance(raw_ignore_regions, list):
        raise ValueError("Canonical objects and ignoreRegions must be arrays")

    objects: list[CanonicalObject] = []
    for index, item in enumerate(raw_objects):
        field = f"objects[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be an object")
        if item.get("detectionClass") != "person" or item.get("ignored") is not False:
            raise ValueError(f"{field} is not a non-ignored person target")
        source_class_id = item.get("sourceClassId")
        if source_class_id not in PERSON_SOURCE_LABELS:
            raise ValueError(f"{field} has unsupported sourceClassId {source_class_id!r}")
        object_id, source_label, attributes = (
            item.get("id"), item.get("sourceLabel"), item.get("attributes")
        )
        if not isinstance(object_id, str) or not isinstance(source_label, str):
            raise ValueError(f"{field} must contain string id and sourceLabel fields")
        if not isinstance(attributes, dict):
            raise ValueError(f"{field}.attributes must be an object")
        if source_label != PERSON_SOURCE_LABELS[source_class_id]:
            raise ValueError(f"{field} sourceLabel does not match sourceClassId")
        objects.append(
            CanonicalObject(
                object_id=object_id,
                full_box=_xywh_to_xyxy(item.get("fullBoxXYWH"), f"{field}.fullBoxXYWH", width, height),
                visible_box=_xywh_to_xyxy(
                    item.get("visibleBoxXYWH"), f"{field}.visibleBoxXYWH", width, height,
                    allow_zero_size=True,
                ),
                source_class_id=source_class_id,
                source_label=source_label,
                attributes=dict(attributes),
            )
        )

    ignore_regions = []
    for index, item in enumerate(raw_ignore_regions):
        field = f"ignoreRegions[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be an object")
        ignore_regions.append(
            _xywh_to_xyxy(item.get("fullBoxXYWH"), f"{field}.fullBoxXYWH", width, height)
        )

    if expected_status == "positive" and not objects:
        raise ValueError(f"Positive split record has no person objects: {expected_image_blob}")
    if expected_status == "verified_negative" and objects:
        raise ValueError(f"Verified-negative split record has person objects: {expected_image_blob}")
    if expected_person_count is not None and len(objects) != expected_person_count:
        raise ValueError(f"Person count does not match split manifest for {expected_image_blob}")
    if expected_ignored_count is not None and len(ignore_regions) != expected_ignored_count:
        raise ValueError(f"Ignore-region count does not match split manifest for {expected_image_blob}")

    return CanonicalAnnotation(
        image_blob=expected_image_blob,
        width=width,
        height=height,
        image_sha256=image_sha256,
        objects=objects,
        ignore_regions=ignore_regions,
    )
