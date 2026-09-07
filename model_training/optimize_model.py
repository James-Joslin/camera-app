"""Export the trained detector to OpenVINO FP16/INT8 and benchmark both variants."""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov
import torch

from train_tune_detector import SSDPersonDetector


def image_tensor(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size)).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose(image, (2, 0, 1))[None, ...]


def benchmark(model_path: Path, iterations: int) -> dict:
    core = ov.Core()
    compiled = core.compile_model(core.read_model(model_path), "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    shape = compiled.input(0).shape
    sample = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    for _ in range(8):
        compiled([sample])
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        compiled([sample])
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    return {
        "meanLatencyMs": statistics.mean(timings),
        "p50LatencyMs": statistics.median(timings),
        "p95LatencyMs": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "fps": 1000 / statistics.mean(timings),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model_fp32.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("optimized"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("test_output"))
    parser.add_argument("--input-size", type=int, default=480)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SSDPersonDetector(num_classes=2, input_size=args.input_size, pretrained=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    example = torch.randn(1, 3, args.input_size, args.input_size)
    ov_model = ov.convert_model(model, example_input=example, input=[1, 3, args.input_size, args.input_size])

    fp16_path = args.output_dir / "person_detector_fp16.xml"
    ov.save_model(ov_model, fp16_path, compress_to_fp16=True)

    calibration_files = sorted(
        path for path in args.calibration_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.calibration_samples]
    if not calibration_files:
        raise RuntimeError(f"No calibration images found in {args.calibration_dir}")
    calibration = nncf.Dataset(calibration_files, lambda path: image_tensor(path, args.input_size))
    quantized_model = nncf.quantize(ov_model, calibration, subset_size=len(calibration_files))
    int8_path = args.output_dir / "person_detector_int8.xml"
    ov.save_model(quantized_model, int8_path, compress_to_fp16=False)

    report = {
        "checkpoint": str(args.checkpoint),
        "inputSize": args.input_size,
        "calibrationSamples": len(calibration_files),
        "fp16": {"path": str(fp16_path), "bytes": fp16_path.with_suffix(".bin").stat().st_size, **benchmark(fp16_path, args.benchmark_iterations)},
        "int8": {"path": str(int8_path), "bytes": int8_path.with_suffix(".bin").stat().st_size, **benchmark(int8_path, args.benchmark_iterations)},
    }
    report["speedup"] = report["fp16"]["meanLatencyMs"] / report["int8"]["meanLatencyMs"]
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

