import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


class OpenVinoPersonDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled = core.compile_model(
            model,
            os.getenv("OPENVINO_DEVICE", "CPU"),
            {"PERFORMANCE_HINT": os.getenv("OPENVINO_PERFORMANCE_HINT", "LATENCY")},
        )
        self.input = self.compiled.input(0)
        self.outputs = self.compiled.outputs
        self.input_size = int(self.input.shape[-1])
        self.anchors = generate_anchors(self.input_size)
        self.lock = threading.Lock()

    def predict(self, image: np.ndarray, threshold: float = 0.5, nms_threshold: float = 0.45) -> tuple[list[dict], float]:
        original_height, original_width = image.shape[:2]
        tensor = preprocess(image, self.input_size)
        started = time.perf_counter()
        with self.lock:
            raw = self.compiled([tensor])
        inference_ms = (time.perf_counter() - started) * 1000
        arrays = [raw[output] for output in self.outputs]
        classes = next(array for array in arrays if array.shape[-1] == 2)[0]
        offsets = next(array for array in arrays if array.shape[-1] == 4)[0]
        scores = softmax(classes, axis=1)[:, 1]
        boxes = decode_boxes(offsets, self.anchors, self.input_size)
        selected = scores >= threshold
        boxes, scores = boxes[selected], scores[selected]
        if not len(scores):
            return [], inference_ms
        keep = nms(boxes, scores, nms_threshold)[:100]
        scale_x, scale_y = original_width / self.input_size, original_height / self.input_size
        detections = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            detections.append(
                {
                    "label": "person",
                    "classId": 1,
                    "confidence": float(scores[index]),
                    "box": [
                        int(np.clip(x1 * scale_x, 0, original_width)),
                        int(np.clip(y1 * scale_y, 0, original_height)),
                        int(np.clip(x2 * scale_x, 0, original_width)),
                        int(np.clip(y2 * scale_y, 0, original_height)),
                    ],
                }
            )
        return detections, inference_ms


class ModelService:
    def __init__(self):
        configured = os.getenv("MODEL_PATH")
        candidates = [
            Path(configured) if configured else None,
            Path("/models/person_detector_int8.xml"),
            Path("/models/person_detector_fp16.xml"),
        ]
        self.model_path = next((path for path in candidates if path and path.exists()), None)
        self.runtime: OpenVinoPersonDetector | None = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def status(self) -> dict:
        return {
            "ready": self.model_path is not None and self.error is None,
            "loaded": self.runtime is not None,
            "model": self.model_path.name if self.model_path else None,
            "device": os.getenv("OPENVINO_DEVICE", "CPU"),
            "error": self.error,
        }

    def predict(self, image: np.ndarray, threshold: float) -> tuple[list[dict], float]:
        runtime = self._runtime()
        return runtime.predict(image, threshold)

    def _runtime(self) -> OpenVinoPersonDetector:
        if self.runtime is not None:
            return self.runtime
        if self.model_path is None:
            raise RuntimeError("No optimized OpenVINO model found. Run scripts/optimize-model.sh first.")
        with self.lock:
            if self.runtime is None:
                try:
                    self.runtime = OpenVinoPersonDetector(self.model_path)
                except Exception as exc:
                    self.error = str(exc)
                    raise
        return self.runtime


def preprocess(image: np.ndarray, input_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0
    normalized = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def generate_anchors(input_size: int) -> np.ndarray:
    strides = [8, 16, 32, 64, 128]
    scales = [[0.02, 0.04], [0.06, 0.10], [0.16, 0.24], [0.32, 0.48, 0.56], [0.64, 0.80, 0.95]]
    ratios = [2.0, 1.0, 0.5]
    anchors = []
    for stride, level_scales in zip(strides, scales):
        size = (input_size + stride - 1) // stride
        for row in range(size):
            for column in range(size):
                for scale in level_scales:
                    for ratio in ratios:
                        anchors.append([(column + 0.5) / size, (row + 0.5) / size, scale * np.sqrt(ratio), scale / np.sqrt(ratio)])
    return np.asarray(anchors, dtype=np.float32)


def decode_boxes(offsets: np.ndarray, anchors: np.ndarray, input_size: int) -> np.ndarray:
    center_x = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    center_y = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    width = np.exp(np.clip(offsets[:, 2], -10, 10)) * anchors[:, 2]
    height = np.exp(np.clip(offsets[:, 3], -10, 10)) * anchors[:, 3]
    return np.stack(
        [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2], axis=1
    ) * input_size


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    exponent = np.exp(values - np.max(values, axis=axis, keepdims=True))
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        intersection_x1 = np.maximum(x1[current], x1[order[1:]])
        intersection_y1 = np.maximum(y1[current], y1[order[1:]])
        intersection_x2 = np.minimum(x2[current], x2[order[1:]])
        intersection_y2 = np.minimum(y2[current], y2[order[1:]])
        intersection = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(0, intersection_y2 - intersection_y1)
        union = areas[current] + areas[order[1:]] - intersection + 1e-6
        order = order[np.where(intersection / union <= threshold)[0] + 1]
    return keep

