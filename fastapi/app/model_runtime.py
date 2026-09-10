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
        tensor, scale, pad_x, pad_y = preprocess(image, self.input_size)
        started = time.perf_counter()
        with self.lock:
            raw = self.compiled([tensor])
        inference_ms = (time.perf_counter() - started) * 1000
        arrays = [np.asarray(raw[output]) for output in self.outputs]
        class_candidates = [
            array for array in arrays
            if array.ndim == 3 and array.shape[-1] in (1, 2)
        ]
        box_candidates = [
            array for array in arrays
            if array.ndim == 3 and array.shape[-1] == 4
        ]
        if len(class_candidates) != 1 or len(box_candidates) != 1:
            shapes = [list(array.shape) for array in arrays]
            raise RuntimeError(f"Cannot identify detector outputs from shapes {shapes}")
        classes = class_candidates[0][0]
        offsets = box_candidates[0][0]
        scores = classification_scores(classes)
        if len(offsets) != len(self.anchors):
            raise RuntimeError(
                f"Detector produced {len(offsets)} boxes for {len(self.anchors)} anchors"
            )
        boxes = decode_boxes(offsets, self.anchors, self.input_size)
        selected = scores >= threshold
        boxes, scores = boxes[selected], scores[selected]
        if not len(scores):
            return [], inference_ms
        keep = nms(boxes, scores, nms_threshold)[:100]
        detections = []
        for index in keep:
            x1, y1, x2, y2 = map_box_from_letterbox(
                boxes[index],
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                original_width=original_width,
                original_height=original_height,
            )
            detections.append(
                {
                    "label": "person",
                    "classId": 1,
                    "confidence": float(scores[index]),
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
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


def preprocess(
    image: np.ndarray, input_size: int
) -> tuple[np.ndarray, float, int, int]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = rgb.shape[:2]
    scale = min(input_size / original_width, input_size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height))
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    normalized = (canvas.astype(np.float32) / 255.0 - np.array(
        [0.485, 0.456, 0.406], dtype=np.float32
    )) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
    return tensor, scale, pad_x, pad_y


def map_box_from_letterbox(
    box: np.ndarray,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_width: int,
    original_height: int,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [
        float(np.clip((x1 - pad_x) / scale, 0, original_width)),
        float(np.clip((y1 - pad_y) / scale, 0, original_height)),
        float(np.clip((x2 - pad_x) / scale, 0, original_width)),
        float(np.clip((y2 - pad_y) / scale, 0, original_height)),
    ]


def classification_scores(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        raise RuntimeError(f"Expected 2D classification logits, got {logits.shape}")
    if logits.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -80, 80)))
    if logits.shape[1] == 2:
        return softmax(logits, axis=1)[:, 1]
    raise RuntimeError(f"Unsupported classification output shape: {logits.shape}")


def generate_anchors(input_size: int) -> np.ndarray:
    strides = [8, 16, 32, 64, 128]
    scales = [[0.02, 0.04], [0.06, 0.10], [0.16, 0.24], [0.32, 0.48, 0.56], [0.64, 0.80, 0.95]]
    ratios = [
        [0.15, 0.25, 0.40],
        [0.15, 0.25, 0.40],
        [0.20, 0.33, 0.50],
        [0.25, 0.50, 1.00],
        [0.25, 0.50, 1.00],
    ]
    anchors = []
    for stride, level_scales, level_ratios in zip(strides, scales, ratios):
        size = (input_size + stride - 1) // stride
        for row in range(size):
            for column in range(size):
                for scale in level_scales:
                    for ratio in level_ratios:
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
