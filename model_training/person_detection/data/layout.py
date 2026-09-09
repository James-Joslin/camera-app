"""Shared helpers for resolving immutable dataset versions and split manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any


LOADER_VERSION = 1
CITYPERSONS_CURRENT_POINTER = "datasets/citypersons/current.json"
TRAINABLE_LABEL_STATUSES = {"positive", "verified_negative"}


def _load_json(read_blob: Callable[[str], bytes | None], name: str) -> dict[str, Any]:
    content = read_blob(name)
    if content is None:
        raise RuntimeError(f"Required dataset blob is missing: {name}")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Dataset blob is not valid JSON: {name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Dataset blob is not a JSON object: {name}")
    return value


def resolve_citypersons_prefix(read_blob: Callable[[str], bytes | None]) -> str:
    pointer = _load_json(read_blob, CITYPERSONS_CURRENT_POINTER)
    if pointer.get("dataset") != "citypersons":
        raise RuntimeError(f"Invalid dataset pointer: {CITYPERSONS_CURRENT_POINTER}")
    prefix = pointer.get("versionPrefix")
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError(f"Dataset pointer has no versionPrefix: {CITYPERSONS_CURRENT_POINTER}")
    prefix = prefix.strip("/")
    manifest_name = f"{prefix}/manifest.json"
    manifest = _load_json(read_blob, manifest_name)
    if manifest.get("versionPrefix") != prefix:
        raise RuntimeError(f"Dataset manifest does not match pointer: {manifest_name}")
    minimum_loader = manifest.get("minimumLoaderVersion", 0)
    if not isinstance(minimum_loader, int) or minimum_loader > LOADER_VERSION:
        raise RuntimeError(
            f"Dataset requires loader version {minimum_loader}; this loader is {LOADER_VERSION}"
        )
    return prefix


def load_citypersons_manifest(
    read_blob: Callable[[str], bytes | None], prefix: str
) -> tuple[dict[str, Any], str]:
    """Return the immutable manifest and checksum of its exact blob bytes."""
    name = f"{prefix.rstrip('/')}/manifest.json"
    content = read_blob(name)
    if content is None:
        raise RuntimeError(f"Required dataset blob is missing: {name}")
    manifest = _load_json(read_blob, name)
    if manifest.get("dataset") != "citypersons" or manifest.get("versionPrefix") != prefix:
        raise RuntimeError(f"Invalid CityPersons dataset manifest: {name}")
    return manifest, hashlib.sha256(content).hexdigest()


def load_citypersons_split(
    read_blob: Callable[[str], bytes | None], prefix: str, split: str
) -> list[dict[str, Any]]:
    name = f"{prefix}/splits/{split}.jsonl"
    content = read_blob(name)
    if content is None:
        raise RuntimeError(f"Required split manifest is missing: {name}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Split manifest is not UTF-8: {name}") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON at {name}:{line_number}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"Non-object record at {name}:{line_number}")
        records.append(record)
    return records


def version_blob(prefix: str, relative_name: str) -> str:
    return f"{prefix.rstrip('/')}/{relative_name.lstrip('/')}"
