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

from canonical_dataset import (
    PRODUCTION_PREPROCESSING,
    CanonicalPersonDetectionDataset,
    clip_box_to_image,
    map_letterbox_box,
    preprocess_rgb_image,
    select_stratified_indices,
    size_slice,
)
from person_detection.contracts import ModelOptimizationPipeline
from test_inference import MAPCalculator
from train_tune_detector import (
    AzuriteClient,
    SSDPersonDetector,
    TrainingConfig,
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
            self.image_rgb(), self.dataset.input_size, add_batch=True
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


def load_checkpoint_model(checkpoint: dict, input_size: int) -> SSDPersonDetector:
    model = SSDPersonDetector(num_classes=1, input_size=input_size, pretrained=False)
    load_detector_state_dict(model, checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def decode_predictions(
    compiled_model: ov.CompiledModel,
    image: np.ndarray,
    anchors: np.ndarray,
    input_size: int,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> list[dict]:
    result = compiled_model([image])
    outputs = [np.asarray(result[output]) for output in compiled_model.outputs]
    cls_candidates = [value for value in outputs if value.ndim == 3 and value.shape[-1] in (1, 2)]
    box_candidates = [value for value in outputs if value.ndim == 3 and value.shape[-1] == 4]
    if len(cls_candidates) != 1 or len(box_candidates) != 1:
        shapes = [list(value.shape) for value in outputs]
        raise RuntimeError(f"Cannot identify detector outputs from shapes {shapes}")
    logits, offsets = cls_candidates[0][0], box_candidates[0][0]
    if logits.shape[-1] == 1:
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -80, 80)))
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        scores = probabilities[:, 1]

    centers_x = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    centers_y = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    widths = np.exp(np.clip(offsets[:, 2], -10, 10)) * anchors[:, 2]
    heights = np.exp(np.clip(offsets[:, 3], -10, 10)) * anchors[:, 3]
    boxes = np.stack([
        (centers_x - widths / 2) * input_size,
        (centers_y - heights / 2) * input_size,
        (centers_x + widths / 2) * input_size,
        (centers_y + heights / 2) * input_size,
    ], axis=1).astype(np.float32)
    boxes = np.clip(boxes, 0, input_size)
    keep_mask = (
        (scores >= score_threshold)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes, scores = boxes[keep_mask], scores[keep_mask].astype(np.float32)
    if not len(scores):
        return []
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_threshold)
    keep = keep[:max_detections].numpy()
    return [
        {"box": boxes[index].tolist(), "score": float(scores[index]), "class": 1}
        for index in keep
    ]


def map_annotation_to_canvas(record: ManifestRecord):
    annotation = record.dataset._load_annotation(record.sample)
    size = record.dataset.input_size
    scale = min(size / annotation.width, size / annotation.height)
    resized_width = max(1, round(annotation.width * scale))
    resized_height = max(1, round(annotation.height * scale))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    boxes = []
    metadata = []
    for obj in annotation.objects:
        clipped = clip_box_to_image(
            obj.full_box, annotation.width, annotation.height
        )
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
            record.dataset.input_size,
            score_threshold,
            nms_threshold,
            max_detections,
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


def benchmark_end_to_end(
    compiled_model: ov.CompiledModel,
    records: list[ManifestRecord],
    anchors: np.ndarray,
    iterations: int,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> dict:
    encoded_images = [record.image_bytes() for record in records]
    timings = []
    for iteration in range(iterations):
        record = records[iteration % len(records)]
        encoded = encoded_images[iteration % len(encoded_images)]
        started = time.perf_counter()
        image = record.image_rgb(encoded)
        tensor = preprocess_rgb_image(
            image, record.dataset.input_size, add_batch=True
        )
        decode_predictions(
            compiled_model,
            tensor,
            anchors,
            record.dataset.input_size,
            score_threshold,
            nms_threshold,
            max_detections,
        )
        timings.append((time.perf_counter() - started) * 1000)
    return latency_summary(timings)


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
    parser.add_argument("--input-size", type=int)
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
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    return parser.parse_args()


class ManifestDrivenOpenVINOOptimizer(ModelOptimizationPipeline):
    """Export, calibrate, validate, and benchmark from immutable manifests."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> dict:
        args = self.args
        if args.max_accuracy_drop < 0:
            raise ValueError("--max-accuracy-drop must be non-negative")
        if args.threads <= 0 or args.benchmark_iterations <= 0:
            raise ValueError("Thread and benchmark iteration counts must be positive")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_bytes = args.checkpoint.read_bytes()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        input_size = args.input_size or checkpoint_config.get("input_size", 480)

        config = TrainingConfig(
            data_root=args.data_root,
            input_size=input_size,
            use_azurite=not args.no_azurite,
            azurite_endpoint=args.azurite_endpoint,
            azurite_access_key=args.azurite_account,
            azurite_secret_key=args.azurite_key,
            enable_quantization=False,
        )
        client = AzuriteClient(config)
        calibration_source = CanonicalPersonDetectionDataset(
            client, split="train", input_size=input_size, augment=False
        )
        validation_source = CanonicalPersonDetectionDataset(
            client, split="val", input_size=input_size, augment=False
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
            "schemaVersion": 1,
            "dataset": calibration_source.dataset_metadata,
            "split": "train",
            "selection": {
                "name": "deterministic-greedy-stratified-v1",
                "seed": args.selection_seed,
                "requestedRecords": args.calibration_samples,
                "selectedRecords": len(calibration_records),
            },
            "inputSize": input_size,
            "preprocessing": PRODUCTION_PREPROCESSING,
            "records": [record.identity() for record in calibration_records],
        }
        calibration_manifest_path = args.output_dir / "calibration_manifest.json"
        write_json(calibration_manifest_path, calibration_manifest)
        calibration_manifest_sha256 = sha256_bytes(calibration_manifest_path.read_bytes())

        model = load_checkpoint_model(checkpoint, input_size)
        anchors = model.anchor_generator.get_anchors().numpy()
        example = torch.zeros(1, 3, input_size, input_size)
        ov_model = ov.convert_model(
            model, example_input=example, input=[1, 3, input_size, input_size]
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
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
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
        for name, path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
            compiled = compile_for_cpu(path, args.threads)
            variants[name] = {
                **model_artifact(path),
                "accuracy": evaluate_compiled_model(
                    compiled,
                    validation_records,
                    anchors,
                    score_threshold=args.score_threshold,
                    nms_threshold=args.nms_threshold,
                    max_detections=args.max_detections,
                    include_slices=True,
                ),
                "coreLatency": benchmark_core(
                    compiled, representative_input, args.benchmark_iterations
                ),
                "endToEndLatency": benchmark_end_to_end(
                    compiled,
                    validation_records,
                    anchors,
                    args.benchmark_iterations,
                    args.score_threshold,
                    args.nms_threshold,
                    args.max_detections,
                ),
            }

        metric_name = "mAP@0.50:0.95"
        fp32_metric = variants["fp32"]["accuracy"]["metrics"][metric_name]
        int8_metric = variants["int8"]["accuracy"]["metrics"][metric_name]
        measured_drop = max(0.0, fp32_metric - int8_metric)
        accepted = measured_drop <= args.max_accuracy_drop + 1e-12
        report = {
            "schemaVersion": 2,
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
            "preprocessing": {**PRODUCTION_PREPROCESSING, "inputSize": input_size},
            "accuracyControl": {
                "metric": metric_name,
                "dropType": "absolute",
                "maximumDrop": args.max_accuracy_drop,
                "measuredDrop": measured_drop,
                "accepted": accepted,
            },
            "benchmarkConfiguration": {
                "device": "CPU",
                "cpu": cpu_model_name(),
                "threads": args.threads,
                "streams": 1,
                "performanceHint": "LATENCY",
                "coreLatencyScope": "compiled model inference only",
                "endToEndLatencyScope": "decode + RGB conversion + letterbox + normalization + inference + NMS",
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
                    "endToEnd": variants["fp32"]["endToEndLatency"]["meanMs"]
                    / values["endToEndLatency"]["meanMs"],
                }
                for name, values in variants.items()
            },
        }
        report_path = args.output_dir / "optimization_report.json"
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not accepted:
            raise RuntimeError(
                f"INT8 rejected: {metric_name} drop {measured_drop:.6f} exceeds "
                f"the {args.max_accuracy_drop:.6f} limit"
            )
        return report


def main() -> None:
    ManifestDrivenOpenVINOOptimizer(parse_args()).run()


if __name__ == "__main__":
    main()
