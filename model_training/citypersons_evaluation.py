"""Adapters for the pinned official CityPersons miss-rate evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_EVALUATOR_COMMIT = "839c22fb05a16c150cb77f9b73a5c0e9642af21e"
OFFICIAL_EVALUATOR_SHA256 = {
    "coco.py": "6958010a2e01881b139b23f652a1b2dd9efdb2ee13d3fa41e401946386499918",
    "eval_MR_multisetup.py": "b91887ead3999b7616766e2093870647bf7ceb2cfbba0b6f43dad8f0887c89a4",
}
OFFICIAL_SETUP_NAMES = [
    "Reasonable",
    "Reasonable_small",
    "Reasonable_occ=heavy",
    "All",
]


def _xyxy_to_xywh(box) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def _visibility_ratio(full_box, visible_box) -> float:
    full = _xyxy_to_xywh(full_box)
    visible = _xyxy_to_xywh(visible_box)
    return min(max(visible[2] * visible[3] / max(full[2] * full[3], 1e-12), 0.0), 1.0)


class OfficialCityPersonsAccumulator:
    """Collect canonical annotations and detections in the official COCO dialect."""

    def __init__(self):
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []

    def add_image(self, image_id: int, image_blob: str, annotation, detections) -> None:
        self.images.append({
            "id": image_id,
            "file_name": image_blob,
            "width": annotation.width,
            "height": annotation.height,
        })
        for obj in annotation.objects:
            # Official CityPersons evaluates pedestrian (class 1). The additional
            # project-positive person classes remain neutral ignore regions here.
            self._add_annotation(
                image_id,
                obj.full_box,
                vis_ratio=_visibility_ratio(obj.full_box, obj.visible_box),
                ignored=obj.source_class_id != 1,
            )
        for box in annotation.ignore_regions:
            self._add_annotation(image_id, box, vis_ratio=1.0, ignored=True)
        for detection in detections:
            box = _xyxy_to_xywh(detection["box"])
            self.detections.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": box,
                "score": float(detection["score"]),
                "height": box[3],
            })

    def _add_annotation(
        self, image_id: int, box, *, vis_ratio: float, ignored: bool
    ) -> None:
        xywh = _xyxy_to_xywh(box)
        self.annotations.append({
            "id": len(self.annotations) + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": xywh,
            "area": xywh[2] * xywh[3],
            "height": xywh[3],
            "vis_ratio": vis_ratio,
            "ignore": int(ignored),
            "iscrowd": int(ignored),
        })

    def payloads(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ground_truth = {
            "info": {
                "description": "Canonical CityPersons export for official evaluation",
            },
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "pedestrian", "supercategory": "person"}],
        }
        return ground_truth, self.detections

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ground_truth, detections = self.payloads()
        gt_path = output_dir / "citypersons_official_ground_truth.json"
        detections_path = output_dir / "citypersons_official_detections.json"
        gt_path.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
        detections_path.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
        return gt_path, detections_path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _official_miss_rate(evaluator) -> float:
    """Extract the value printed (but not returned) by the pinned evaluator."""
    max_detection_indices = [
        index for index, value in enumerate(evaluator.params.maxDets) if value == 1000
    ]
    if not max_detection_indices:
        raise RuntimeError("Official evaluator has no 1000-detection summary setting")
    miss_rates = 1 - evaluator.eval["TP"][:, :, :, max_detection_indices]
    valid = miss_rates[miss_rates < 2]
    if not len(valid):
        return -1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.exp(np.mean(np.log(valid))))

def run_official_citypersons_evaluator(
    accumulator: OfficialCityPersonsAccumulator,
    evaluator_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the unmodified official COCO/CityPersons evaluator modules."""
    evaluator_dir = evaluator_dir.resolve()
    coco_path = evaluator_dir / "coco.py"
    mr_path = evaluator_dir / "eval_MR_multisetup.py"
    for path in (coco_path, mr_path):
        if not path.is_file():
            raise RuntimeError(f"Official CityPersons evaluator file is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != OFFICIAL_EVALUATOR_SHA256[path.name]:
            raise RuntimeError(
                f"Official evaluator checksum mismatch for {path}; expected pinned "
                f"commit {OFFICIAL_EVALUATOR_COMMIT}"
            )

    gt_path, detections_path = accumulator.write(output_dir)
    provenance = {
        "backend": "cvgroup-njust/CityPersons",
        "commit": OFFICIAL_EVALUATOR_COMMIT,
        "evaluatorDirectory": str(evaluator_dir),
        "cocoSha256": _sha256(coco_path),
        "missRateEvaluatorSha256": _sha256(mr_path),
        "groundTruth": str(gt_path),
        "detections": str(detections_path),
    }
    if not accumulator.detections:
        return {
            **provenance,
            "metrics": {f"MR/{name}": 1.0 for name in OFFICIAL_SETUP_NAMES},
        }

    coco_module = _load_module("_citypersons_official_coco", coco_path)
    mr_module = _load_module("_citypersons_official_mr", mr_path)

    # The pinned evaluator predates NumPy 2. Keep its code unmodified and scope
    # the two compatibility aliases to this call.
    old_float = getattr(np, "float", None)
    old_linspace = np.linspace

    def compatible_linspace(start, stop, num=50, *args, **kwargs):
        return old_linspace(start, stop, int(num), *args, **kwargs)

    setattr(np, "float", float)
    np.linspace = compatible_linspace
    result_path = output_dir / "citypersons_official_results.txt"
    metrics = {}
    try:
        with result_path.open("w", encoding="utf-8") as result_stream:
            for setup_id, setup_name in enumerate(OFFICIAL_SETUP_NAMES):
                coco_gt = coco_module.COCO(str(gt_path))
                coco_dt = coco_gt.loadRes(str(detections_path))
                evaluator = mr_module.COCOeval(coco_gt, coco_dt, "bbox")
                evaluator.params.imgIds = sorted(coco_gt.getImgIds())
                evaluator.evaluate(setup_id)
                evaluator.accumulate()
                evaluator.summarize(setup_id, result_stream)
                metrics[f"MR/{setup_name}"] = _official_miss_rate(evaluator)
    finally:
        np.linspace = old_linspace
        if old_float is None:
            delattr(np, "float")
        else:
            setattr(np, "float", old_float)
    return {**provenance, "results": str(result_path), "metrics": metrics}
