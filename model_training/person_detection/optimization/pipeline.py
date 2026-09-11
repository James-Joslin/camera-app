"""Manifest-driven OpenVINO export, INT8 calibration, validation, and benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
import torchvision
from torchvision.ops import nms

from person_detection.data.dataset import (
    PRODUCTION_PREPROCESSING,
    CanonicalPersonDetectionDataset,
    clip_box_to_image,
    map_letterbox_box,
    preprocess_rgb_image,
    select_stratified_indices,
    size_slice,
)
from person_detection.core.contracts import ModelOptimizationPipeline
from person_detection.evaluation.inference import MAPCalculator
from person_detection.training.pipeline import (
    AzuriteClient,
    SSDPersonDetector,
    TrainingConfig,
    MODEL_FORMAT_VERSION,
    load_detector_state_dict,
)


IOU_THRESHOLDS = [0.5 + index * 0.05 for index in range(10)]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class ManifestRecord:
    """A selected immutable split record and its canonical data loader."""

    dataset: CanonicalPersonDetectionDataset
    index: int

    @property
    def sample(self) -> dict:
        return self.dataset.samples[self.index]

    def image_bytes(self) -> bytes:
        sample = self.sample
        data = self.dataset.azurite.get_object_bytes(
            self.dataset.config.azurite_data_bucket, sample["image"]
        )
        if data is None:
            raise RuntimeError(f"Calibration image is missing: {sample['image']}")
        if sha256_bytes(data) != sample["image_sha256"]:
            raise RuntimeError(f"Calibration image checksum mismatch: {sample['image']}")
        return data

    def image_rgb(self, encoded: bytes | None = None) -> np.ndarray:
        encoded = self.image_bytes() if encoded is None else encoded
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Calibration image cannot be decoded: {self.sample['image']}")
        annotation = self.dataset._load_annotation(self.sample)
        if image.shape[:2] != (annotation.height, annotation.width):
            raise RuntimeError(f"Calibration image dimensions mismatch: {self.sample['image']}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def input_tensor(self) -> np.ndarray:
        return preprocess_rgb_image(
            self.image_rgb(),
            self.dataset.input_height,
            self.dataset.input_width,
            add_batch=True,
        )

    def identity(self) -> dict:
        sample = self.sample
        return {
            "manifestIndex": sample["manifest_index"],
            "image": sample["image_relative"],
            "annotation": sample["annotation_relative"],
            "labelStatus": sample["status"],
            "checksums": {
                "image": sample["image_sha256"],
                "annotation": sample["annotation_sha256"],
            },
            "strata": sorted(self.dataset.strata_for_sample(self.index)),
        }


def select_records(
    dataset: CanonicalPersonDetectionDataset,
    sample_count: int,
    *,
    seed: int,
    stratified: bool,
) -> list[ManifestRecord]:
    if not dataset.samples:
        raise RuntimeError(f"Canonical {dataset.split} split has no eligible records")
    count = min(sample_count, len(dataset))
    if count <= 0:
        raise ValueError("Sample counts must be positive")
    if stratified:
        strata = [dataset.strata_for_sample(index) for index in range(len(dataset))]
        indices = select_stratified_indices(strata, count, seed)
    else:
        indices = list(range(count))
    return [ManifestRecord(dataset, index) for index in indices]


def verify_checkpoint_dataset(checkpoint: dict, dataset: CanonicalPersonDetectionDataset) -> dict:
    expected = dataset.dataset_metadata
    actual = checkpoint.get("dataset")
    keys = ("versionPrefix", "manifestSha256", "schemaVersion")
    if not isinstance(actual, dict):
        raise RuntimeError(
            "Checkpoint has no canonical dataset provenance; retrain it or pass "
            "--allow-unverified-checkpoint explicitly"
        )
    mismatches = {
        key: {"checkpoint": actual.get(key), "selectedDataset": expected.get(key)}
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint/dataset provenance mismatch: {mismatches}")
    return {"verified": True, "checkpointDataset": actual}


def load_checkpoint_model(
    checkpoint: dict, input_height: int, input_width: int
) -> SSDPersonDetector:
    model = SSDPersonDetector(
        num_classes=1,
        input_height=input_height,
        input_width=input_width,
        pretrained=False,
    )
    load_detector_state_dict(model, checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def decode_predictions(
    compiled_model: ov.CompiledModel,
    image: np.ndarray,
    anchors: np.ndarray,
    input_height: int,
    input_width: int,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
    pre_nms_topk: int | None,
    *,
    stage_timings: dict[str, list[float]] | None = None,
    candidate_counts: list[dict] | None = None,
) -> list[dict]:
    started = time.perf_counter()
    result = compiled_model([image])
    outputs = [np.asarray(result[output]) for output in compiled_model.outputs]
    inference_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    cls_candidates = [
        value for value in outputs
        if value.ndim == 3 and value.shape[-1] in (1, 2)
    ]
    box_candidates = [
        value for value in outputs if value.ndim == 3 and value.shape[-1] == 4
    ]
    if len(cls_candidates) != 1 or len(box_candidates) != 1:
        shapes = [list(value.shape) for value in outputs]
        raise RuntimeError(f"Cannot identify detector outputs from shapes {shapes}")
    logits, offsets = cls_candidates[0][0], box_candidates[0][0]
    if len(offsets) != len(anchors):
        raise RuntimeError(
            f"Model produced {len(offsets)} boxes but {len(anchors)} anchors were generated"
        )
    if logits.shape[-1] == 1:
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -80, 80)))
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        scores = probabilities[:, 1]

    score_indices = np.flatnonzero(scores >= score_threshold)
    after_score_filter = len(score_indices)
    if pre_nms_topk is not None and len(score_indices) > pre_nms_topk:
        local_top = np.argpartition(scores[score_indices], -pre_nms_topk)[-pre_nms_topk:]
        score_indices = score_indices[local_top]
    selected_scores = scores[score_indices].astype(np.float32, copy=False)
    selected_offsets = offsets[score_indices]
    selected_anchors = anchors[score_indices]

    if len(selected_scores):
        centers_x = selected_offsets[:, 0] * selected_anchors[:, 2] + selected_anchors[:, 0]
        centers_y = selected_offsets[:, 1] * selected_anchors[:, 3] + selected_anchors[:, 1]
        widths = np.exp(np.clip(selected_offsets[:, 2], -10, 10)) * selected_anchors[:, 2]
        heights = np.exp(np.clip(selected_offsets[:, 3], -10, 10)) * selected_anchors[:, 3]
        boxes = np.stack([
            (centers_x - widths / 2) * input_width,
            (centers_y - heights / 2) * input_height,
            (centers_x + widths / 2) * input_width,
            (centers_y + heights / 2) * input_height,
        ], axis=1).astype(np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, selected_scores = boxes[valid], selected_scores[valid]
    else:
        boxes = np.empty((0, 4), dtype=np.float32)
    decode_filter_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    if len(selected_scores):
        keep = nms(
            torch.from_numpy(boxes), torch.from_numpy(selected_scores), nms_threshold
        )
        keep = keep[:max_detections].numpy()
    else:
        keep = np.empty((0,), dtype=np.int64)
    nms_ms = (time.perf_counter() - started) * 1000

    if stage_timings is not None:
        stage_timings.setdefault("inference", []).append(inference_ms)
        stage_timings.setdefault("boxDecodeAndFilter", []).append(decode_filter_ms)
        stage_timings.setdefault("nms", []).append(nms_ms)
    if candidate_counts is not None:
        candidate_counts.append({
            "modelOutputs": len(scores),
            "afterScoreFilter": after_score_filter,
            "enteringNms": len(boxes),
            "afterNmsAndLimit": len(keep),
        })
    return [
        {"box": boxes[index].tolist(), "score": float(selected_scores[index]), "class": 1}
        for index in keep
    ]


def map_annotation_to_canvas(record: ManifestRecord):
    annotation = record.dataset._load_annotation(record.sample)
    input_height = record.dataset.input_height
    input_width = record.dataset.input_width
    scale = min(input_width / annotation.width, input_height / annotation.height)
    resized_width = max(1, round(annotation.width * scale))
    resized_height = max(1, round(annotation.height * scale))
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    boxes = []
    metadata = []
    for obj in annotation.objects:
        clipped = clip_box_to_image(obj.full_box, annotation.width, annotation.height)
        if clipped is None:
            continue
        boxes.append(map_letterbox_box(clipped, scale, pad_x, pad_y))
        metadata.append({
            "size": size_slice(obj.full_box),
            "sourceLabel": obj.source_label,
            "visibility": obj.attributes.get("visibility", "unknown"),
        })
    ignores = []
    for box in annotation.ignore_regions:
        clipped = clip_box_to_image(box, annotation.width, annotation.height)
        if clipped is not None:
            ignores.append(map_letterbox_box(clipped, scale, pad_x, pad_y))
    return boxes, ignores, metadata


def evaluate_compiled_model(
    compiled_model: ov.CompiledModel,
    records: Iterable[ManifestRecord],
    anchors: np.ndarray,
    *,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
    pre_nms_topk: int | None,
    include_slices: bool,
) -> dict:
    calculator = MAPCalculator(IOU_THRESHOLDS, recall_fppi=0.1)
    record_count = 0
    for image_id, record in enumerate(records):
        calculator.add_image(image_id)
        detections = decode_predictions(
            compiled_model,
            record.input_tensor(),
            anchors,
            record.dataset.input_height,
            record.dataset.input_width,
            score_threshold,
            nms_threshold,
            max_detections,
            pre_nms_topk,
        )
        boxes, ignores, metadata = map_annotation_to_canvas(record)
        calculator.add_predictions(image_id, detections)
        calculator.add_ignore_regions(image_id, ignores)
        calculator.add_ground_truths(image_id, boxes, [1] * len(boxes), metadata)
        record_count += 1
    metrics = calculator.compute_map(verbose=False)
    result = {"records": record_count, "metrics": metrics}
    if include_slices:
        result["slices"] = calculator.compute_slices()
    return result


def compile_for_cpu(model_or_path, threads: int) -> ov.CompiledModel:
    core = ov.Core()
    model = core.read_model(model_or_path) if isinstance(model_or_path, Path) else model_or_path
    return core.compile_model(
        model,
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": str(threads),
            "INFERENCE_PRECISION_HINT": "f32",
        },
    )


def latency_summary(timings: list[float]) -> dict:
    return {
        "meanMs": float(statistics.mean(timings)),
        "p50Ms": float(np.percentile(timings, 50)),
        "p95Ms": float(np.percentile(timings, 95)),
        "fps": float(1000.0 / statistics.mean(timings)),
        "iterations": len(timings),
    }


def count_summary(counts: list[int]) -> dict:
    return {
        "mean": float(statistics.mean(counts)),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "maximum": int(max(counts)),
    }


def benchmark_core(
    compiled_model: ov.CompiledModel, sample: np.ndarray, iterations: int
) -> dict:
    for _ in range(8):
        compiled_model([sample])
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        compiled_model([sample])
        timings.append((time.perf_counter() - started) * 1000)
    return latency_summary(timings)


def benchmark_pipeline(
    compiled_model: ov.CompiledModel,
    records: list[ManifestRecord],
    anchors: np.ndarray,
    iterations: int,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
    pre_nms_topk: int,
    *,
    encoded_input: bool,
) -> dict:
    benchmark_record_count = min(len(records), iterations, 16)
    benchmark_records = records[:benchmark_record_count]
    encoded_images = [record.image_bytes() for record in benchmark_records]
    raw_images = None
    if not encoded_input:
        raw_images = [
            cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            for encoded in encoded_images
        ]
        if any(image is None for image in raw_images):
            raise RuntimeError("A benchmark image could not be decoded")
    stage_timings: dict[str, list[float]] = {
        "jpegDecode": [],
        "colorConversion": [],
        "letterboxAndNormalization": [],
        "inference": [],
        "boxDecodeAndFilter": [],
        "nms": [],
    }
    total_timings = []
    candidates: list[dict] = []
    for iteration in range(iterations):
        index = iteration % benchmark_record_count
        started_total = time.perf_counter()
        if encoded_input:
            started = time.perf_counter()
            image_bgr = cv2.imdecode(
                np.frombuffer(encoded_images[index], np.uint8), cv2.IMREAD_COLOR
            )
            stage_timings["jpegDecode"].append(
                (time.perf_counter() - started) * 1000
            )
        else:
            image_bgr = raw_images[index]
        started = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stage_timings["colorConversion"].append(
            (time.perf_counter() - started) * 1000
        )
        started = time.perf_counter()
        tensor = preprocess_rgb_image(
            image_rgb,
            benchmark_records[index].dataset.input_height,
            benchmark_records[index].dataset.input_width,
            add_batch=True,
        )
        stage_timings["letterboxAndNormalization"].append(
            (time.perf_counter() - started) * 1000
        )
        decode_predictions(
            compiled_model,
            tensor,
            anchors,
            benchmark_records[index].dataset.input_height,
            benchmark_records[index].dataset.input_width,
            score_threshold,
            nms_threshold,
            max_detections,
            pre_nms_topk,
            stage_timings=stage_timings,
            candidate_counts=candidates,
        )
        total_timings.append((time.perf_counter() - started_total) * 1000)
    stages = {
        name: latency_summary(values)
        for name, values in stage_timings.items()
        if values
    }
    candidate_report = {
        key: count_summary([record[key] for record in candidates])
        for key in candidates[0]
    }
    return {
        **latency_summary(total_timings),
        "source": "jpeg" if encoded_input else "raw-bgr-frame",
        "representativeFrames": benchmark_record_count,
        "scoreThreshold": score_threshold,
        "preNmsTopK": pre_nms_topk,
        "stages": stages,
        "candidates": candidate_report,
    }


def model_artifact(path: Path) -> dict:
    bin_path = path.with_suffix(".bin")
    return {
        "xml": str(path),
        "bin": str(bin_path),
        "xmlBytes": path.stat().st_size,
        "weightsBytes": bin_path.stat().st_size,
        "totalBytes": path.stat().st_size + bin_path.stat().st_size,
    }


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest-driven, accuracy-controlled OpenVINO optimization"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model_fp32.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("optimized"))
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--no-azurite", action="store_true")
    parser.add_argument(
        "--azurite-endpoint",
        default=os.getenv(
            "AZURITE_BLOB_ENDPOINT", "http://127.0.0.1:10000/devstoreaccount1"
        ),
    )
    parser.add_argument("--azurite-account", default=os.getenv("AZURITE_ACCOUNT_NAME", "devstoreaccount1"))
    parser.add_argument("--azurite-key", default=os.getenv("AZURITE_ACCOUNT_KEY", ""))
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--selection-seed", type=int, default=1337)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument(
        "--score-threshold",
        "--evaluation-score-threshold",
        dest="evaluation_score_threshold",
        type=float,
        default=0.01,
        help="Low score threshold used only for accuracy evaluation",
    )
    parser.add_argument(
        "--benchmark-score-threshold",
        type=float,
        default=0.5,
        help="Production-like score threshold used only for latency benchmarks",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--max-topk-quality-drop", type=float, default=0.0001)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--release-status", choices=("experimental", "production"), default="experimental"
    )
    parser.add_argument("--min-map-50", type=float, default=0.25)
    parser.add_argument("--min-map-50-95", type=float, default=0.10)
    parser.add_argument("--min-recall-fppi", type=float, default=0.20)
    parser.add_argument(
        "--camera-metrics",
        type=Path,
        help="JSON evaluation metrics from representative camera-domain footage",
    )
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    return parser.parse_args()


def quality_gate(metrics: dict, minimums: dict) -> dict:
    measured = {
        "mAP@0.50": float(metrics.get("mAP@0.50", 0.0)),
        "mAP@0.50:0.95": float(metrics.get("mAP@0.50:0.95", 0.0)),
        "Recall@FPPI=0.10": float(metrics.get("Recall@FPPI=0.10", 0.0)),
    }
    checks = {
        metric: measured[metric] + 1e-12 >= minimum
        for metric, minimum in minimums.items()
    }
    return {"metrics": measured, "checks": checks, "accepted": all(checks.values())}


def read_camera_metrics(path: Path | None, minimums: dict) -> dict:
    if path is None:
        return {
            "required": True,
            "evaluated": False,
            "accepted": False,
            "reason": "No representative camera-domain metrics were supplied",
        }
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Camera metrics JSON must be an object")
    project_binary = payload.get("projectBinary")
    metrics = (
        project_binary.get("metrics")
        if isinstance(project_binary, dict)
        else payload.get("metrics", payload.get("mapResults", payload))
    )
    if not isinstance(metrics, dict):
        raise ValueError("Camera metrics JSON must contain a metrics mapping")
    return {
        "required": True,
        "evaluated": True,
        "path": str(path),
        "sha256": sha256_bytes(payload_bytes),
        **quality_gate(metrics, minimums),
    }


def release_is_accepted(
    status: str,
    *,
    quantization_accepted: bool,
    topk_accepted: bool,
    quality_accepted: bool,
    camera_accepted: bool,
) -> bool:
    if status not in ("experimental", "production"):
        raise ValueError(f"Unsupported release status: {status}")
    production_evidence = quality_accepted and camera_accepted
    return quantization_accepted and topk_accepted and (
        status == "experimental" or production_evidence
    )


class ManifestDrivenOpenVINOOptimizer(ModelOptimizationPipeline):
    """Export, calibrate, validate, benchmark, and gate release artifacts."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> dict:
        args = self.args
        if args.max_accuracy_drop < 0 or args.max_topk_quality_drop < 0:
            raise ValueError("Accuracy-drop limits must be non-negative")
        if args.threads <= 0 or args.benchmark_iterations <= 0:
            raise ValueError("Thread and benchmark iteration counts must be positive")
        if args.pre_nms_topk <= 0:
            raise ValueError("--pre-nms-topk must be positive")
        for name in (
            "evaluation_score_threshold",
            "benchmark_score_threshold",
            "nms_threshold",
            "min_map_50",
            "min_map_50_95",
            "min_recall_fppi",
        ):
            value = getattr(args, name)
            if not 0 <= value <= 1:
                raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_bytes = args.checkpoint.read_bytes()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("modelFormatVersion") != MODEL_FORMAT_VERSION:
            raise RuntimeError(
                f"Checkpoint format {checkpoint.get('modelFormatVersion')} cannot be released "
                f"with rectangular model format {MODEL_FORMAT_VERSION}; retrain or resume first"
            )
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        input_height = args.input_height or checkpoint_config.get("input_height", 360)
        input_width = args.input_width or checkpoint_config.get("input_width", 640)
        if input_width <= input_height:
            raise ValueError("--input-width must be greater than --input-height")

        config = TrainingConfig(
            data_root=args.data_root,
            input_height=input_height,
            input_width=input_width,
            use_azurite=not args.no_azurite,
            azurite_endpoint=args.azurite_endpoint,
            azurite_access_key=args.azurite_account,
            azurite_secret_key=args.azurite_key,
            enable_quantization=False,
        )
        client = AzuriteClient(config)
        calibration_source = CanonicalPersonDetectionDataset(
            client,
            split="train",
            input_height=input_height,
            input_width=input_width,
            augment=False,
        )
        validation_source = CanonicalPersonDetectionDataset(
            client,
            split="val",
            input_height=input_height,
            input_width=input_width,
            augment=False,
        )
        try:
            provenance = verify_checkpoint_dataset(checkpoint, calibration_source)
        except RuntimeError as exc:
            if not args.allow_unverified_checkpoint:
                raise
            provenance = {"verified": False, "warning": str(exc)}

        calibration_records = select_records(
            calibration_source,
            args.calibration_samples,
            seed=args.selection_seed,
            stratified=True,
        )
        validation_records = select_records(
            validation_source,
            args.validation_samples,
            seed=args.selection_seed,
            stratified=False,
        )
        calibration_manifest = {
            "schemaVersion": 2,
            "dataset": calibration_source.dataset_metadata,
            "split": "train",
            "selection": {
                "name": "deterministic-greedy-stratified-v1",
                "seed": args.selection_seed,
                "requestedRecords": args.calibration_samples,
                "selectedRecords": len(calibration_records),
            },
            "inputHeight": input_height,
            "inputWidth": input_width,
            "preprocessing": PRODUCTION_PREPROCESSING,
            "records": [record.identity() for record in calibration_records],
        }
        calibration_manifest_path = args.output_dir / "calibration_manifest.json"
        write_json(calibration_manifest_path, calibration_manifest)
        calibration_manifest_sha256 = sha256_bytes(calibration_manifest_path.read_bytes())

        model = load_checkpoint_model(checkpoint, input_height, input_width)
        anchor_specification = model.anchor_generator.specification()
        if checkpoint.get("anchors") != anchor_specification:
            raise RuntimeError("Checkpoint anchor specification does not match the exported model")
        anchors = model.anchor_generator.get_anchors().numpy()
        example = torch.zeros(1, 3, input_height, input_width)
        ov_model = ov.convert_model(
            model, example_input=example, input=[1, 3, input_height, input_width]
        )
        fp32_path = args.output_dir / "person_detector_fp32.xml"
        fp16_path = args.output_dir / "person_detector_fp16.xml"
        int8_path = args.output_dir / "person_detector_int8.xml"
        ov.save_model(ov_model, fp32_path, compress_to_fp16=False)
        ov.save_model(ov_model, fp16_path, compress_to_fp16=True)

        calibration_dataset = nncf.Dataset(
            calibration_records, lambda record: record.input_tensor()
        )
        validation_dataset = nncf.Dataset(
            validation_records, lambda record: record.input_tensor()
        )

        def validation_fn(compiled_model, records):
            result = evaluate_compiled_model(
                compiled_model,
                records,
                anchors,
                score_threshold=args.evaluation_score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
                pre_nms_topk=args.pre_nms_topk,
                include_slices=False,
            )
            return result["metrics"]["mAP@0.50:0.95"]

        quantized_model = nncf.quantize_with_accuracy_control(
            ov_model,
            calibration_dataset,
            validation_dataset,
            validation_fn,
            max_drop=args.max_accuracy_drop,
            drop_type=nncf.DropType.ABSOLUTE,
            preset=nncf.QuantizationPreset.MIXED,
            target_device=nncf.TargetDevice.CPU,
            subset_size=len(calibration_records),
            fast_bias_correction=True,
        )
        ov.save_model(quantized_model, int8_path, compress_to_fp16=False)

        variants = {}
        representative_input = calibration_records[0].input_tensor()
        for name, model_path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
            compiled = compile_for_cpu(model_path, args.threads)
            accuracy = evaluate_compiled_model(
                compiled,
                validation_records,
                anchors,
                score_threshold=args.evaluation_score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
                pre_nms_topk=args.pre_nms_topk,
                include_slices=True,
            )
            jpeg_pipeline = benchmark_pipeline(
                compiled,
                validation_records,
                anchors,
                args.benchmark_iterations,
                args.benchmark_score_threshold,
                args.nms_threshold,
                args.max_detections,
                args.pre_nms_topk,
                encoded_input=True,
            )
            raw_pipeline = benchmark_pipeline(
                compiled,
                validation_records,
                anchors,
                args.benchmark_iterations,
                args.benchmark_score_threshold,
                args.nms_threshold,
                args.max_detections,
                args.pre_nms_topk,
                encoded_input=False,
            )
            variants[name] = {
                **model_artifact(model_path),
                "accuracy": accuracy,
                "coreLatency": benchmark_core(
                    compiled, representative_input, args.benchmark_iterations
                ),
                "endToEndLatency": jpeg_pipeline,
                "rawFrameLatency": raw_pipeline,
            }
            if name in ("fp32", "int8"):
                variants[name]["uncappedAccuracy"] = evaluate_compiled_model(
                    compiled,
                    validation_records,
                    anchors,
                    score_threshold=args.evaluation_score_threshold,
                    nms_threshold=args.nms_threshold,
                    max_detections=args.max_detections,
                    pre_nms_topk=None,
                    include_slices=False,
                )

        metric_name = "mAP@0.50:0.95"
        fp32_metric = variants["fp32"]["accuracy"]["metrics"][metric_name]
        int8_metric = variants["int8"]["accuracy"]["metrics"][metric_name]
        measured_drop = max(0.0, fp32_metric - int8_metric)
        quantization_accepted = measured_drop <= args.max_accuracy_drop + 1e-12

        topk_metrics = ("mAP@0.50", "mAP@0.50:0.95", "Recall@FPPI=0.10")
        topk_variants = {}
        for name in ("fp32", "int8"):
            baseline = variants[name]["uncappedAccuracy"]["metrics"]
            capped = variants[name]["accuracy"]["metrics"]
            drops = {
                metric: max(0.0, float(baseline[metric]) - float(capped[metric]))
                for metric in topk_metrics
            }
            topk_variants[name] = {
                "uncapped": {metric: float(baseline[metric]) for metric in topk_metrics},
                "capped": {metric: float(capped[metric]) for metric in topk_metrics},
                "drops": drops,
                "accepted": all(
                    value <= args.max_topk_quality_drop + 1e-12
                    for value in drops.values()
                ),
            }
        topk_accepted = all(value["accepted"] for value in topk_variants.values())

        minimums = {
            "mAP@0.50": args.min_map_50,
            "mAP@0.50:0.95": args.min_map_50_95,
            "Recall@FPPI=0.10": args.min_recall_fppi,
        }
        quality_variants = {
            name: quality_gate(variants[name]["accuracy"]["metrics"], minimums)
            for name in ("fp32", "int8")
        }
        quality_accepted = all(value["accepted"] for value in quality_variants.values())
        camera_gate = read_camera_metrics(args.camera_metrics, minimums)
        production_evidence_accepted = quality_accepted and camera_gate["accepted"]
        release_accepted = release_is_accepted(
            args.release_status,
            quantization_accepted=quantization_accepted,
            topk_accepted=topk_accepted,
            quality_accepted=quality_accepted,
            camera_accepted=camera_gate["accepted"],
        )

        report = {
            "schemaVersion": 3,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_bytes(checkpoint_bytes),
                "modelFormatVersion": checkpoint.get("modelFormatVersion"),
                **provenance,
            },
            "dataset": calibration_source.dataset_metadata,
            "calibrationManifest": {
                "path": str(calibration_manifest_path),
                "sha256": calibration_manifest_sha256,
            },
            "validation": {
                "split": "val",
                "samplingPolicy": "natural-order-v1",
                "records": [record.identity() for record in validation_records],
            },
            "preprocessing": {
                **PRODUCTION_PREPROCESSING,
                "inputHeight": input_height,
                "inputWidth": input_width,
            },
            "anchors": anchor_specification,
            "accuracyControl": {
                "metric": metric_name,
                "dropType": "absolute",
                "maximumDrop": args.max_accuracy_drop,
                "measuredDrop": measured_drop,
                "accepted": quantization_accepted,
            },
            "topKValidation": {
                "preNmsTopK": args.pre_nms_topk,
                "evaluationScoreThreshold": args.evaluation_score_threshold,
                "maximumQualityDrop": args.max_topk_quality_drop,
                "variants": topk_variants,
                "accepted": topk_accepted,
            },
            "release": {
                "status": args.release_status,
                "accepted": release_accepted,
                "minimums": minimums,
                "qualityGate": {
                    "variants": quality_variants,
                    "accepted": quality_accepted,
                },
                "cameraDomainGate": camera_gate,
                "productionEvidenceAccepted": production_evidence_accepted,
                "experimentalWarning": (
                    None if args.release_status == "production" else
                    "Experimental artifacts have not passed production quality and camera-domain gates"
                ),
            },
            "benchmarkConfiguration": {
                "device": "CPU",
                "cpu": cpu_model_name(),
                "threads": args.threads,
                "streams": 1,
                "performanceHint": "LATENCY",
                "evaluationScoreThreshold": args.evaluation_score_threshold,
                "benchmarkScoreThreshold": args.benchmark_score_threshold,
                "preNmsTopK": args.pre_nms_topk,
                "coreLatencyScope": "compiled model inference only",
                "jpegPipelineStages": [
                    "jpegDecode", "colorConversion", "letterboxAndNormalization", "inference",
                    "boxDecodeAndFilter", "nms",
                ],
                "rawFramePipelineStages": [
                    "colorConversion", "letterboxAndNormalization", "inference",
                    "boxDecodeAndFilter", "nms",
                ],
            },
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "openvino": ov.__version__,
                "nncf": nncf.__version__,
            },
            "variants": variants,
            "speedupVsFp32": {
                name: {
                    "core": variants["fp32"]["coreLatency"]["meanMs"]
                    / values["coreLatency"]["meanMs"],
                    "jpegPipeline": variants["fp32"]["endToEndLatency"]["meanMs"]
                    / values["endToEndLatency"]["meanMs"],
                    "rawFramePipeline": variants["fp32"]["rawFrameLatency"]["meanMs"]
                    / values["rawFrameLatency"]["meanMs"],
                }
                for name, values in variants.items()
            },
        }
        report_path = args.output_dir / "optimization_report.json"
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not release_accepted:
            reasons = []
            if not quantization_accepted:
                reasons.append(
                    f"INT8 {metric_name} drop {measured_drop:.6f} exceeds "
                    f"{args.max_accuracy_drop:.6f}"
                )
            if not topk_accepted:
                reasons.append("pre-NMS top-k reduced validation quality beyond its limit")
            if args.release_status == "production" and not quality_accepted:
                reasons.append("FP32 or INT8 failed the minimum validation quality gate")
            if args.release_status == "production" and not camera_gate["accepted"]:
                reasons.append("representative camera-domain evidence is absent or below minimums")
            raise RuntimeError("Release rejected: " + "; ".join(reasons))
        return report

def main() -> None:
    ManifestDrivenOpenVINOOptimizer(parse_args()).run()


if __name__ == "__main__":
    main()
