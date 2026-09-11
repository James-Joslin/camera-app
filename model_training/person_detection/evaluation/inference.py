"""
Test Inference Script with mAP Evaluation
==========================================
Loads trained SSD model, runs inference on test images,
draws predicted bounding boxes, saves visualized results,
and calculates mAP metrics.

Supports:
- FP32 PyTorch models (.pth)
- INT8 OpenVINO IR models (.xml)
- INT8 NNCF PyTorch models (.pth with NNCF wrapper)
"""

import json
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from torchvision.ops import nms
from tqdm import tqdm

from person_detection.data.layout import load_citypersons_manifest, load_citypersons_split, resolve_citypersons_prefix, version_blob
from person_detection.data.annotations import parse_canonical_annotation
from person_detection.data.dataset import (
    PRODUCTION_PREPROCESSING,
    clip_box_to_image,
    preprocess_rgb_image,
    size_slice,
)
from person_detection.evaluation.citypersons import (
    OfficialCityPersonsAccumulator,
    PinnedCityPersonsEvaluator,
)
from person_detection.core.contracts import DetectionBackend

# Try to import from main training script
try:
    from person_detection.training.pipeline import (
        SSDPersonDetector,
        MODEL_FORMAT_VERSION,
        load_detector_state_dict,
        person_scores_from_logits,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import from ssd_person_detection.py")
    print("Make sure the file is in the same directory or PYTHONPATH")

# Check for OpenVINO
try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("Note: OpenVINO not installed. INT8 IR inference disabled.")

# Check for NNCF
try:
    import nncf
    HAS_NNCF = True
except ImportError:
    HAS_NNCF = False


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Model settings
    model_path: str = 'best_model_fp32.pth'
    input_height: int = 360
    input_width: int = 640
    num_classes: int = 1  # one sigmoid localization-quality logit
    model_type: str = 'auto'  # 'auto', 'pytorch', 'openvino', 'nncf'

    # Inference settings
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    pre_nms_topk: int = 1000

    # Data settings
    data_root: str = './data'
    output_dir: str = './test_output'
    max_images: int = 100

    # mAP Evaluation settings
    evaluate_map: bool = True
    map_iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5 + index * 0.05 for index in range(10)]
    )
    map_score_threshold: float = 0.01  # Lower threshold for mAP calculation
    recall_fppi: float = 0.1
    official_evaluator_dir: Optional[str] = None

    # Azurite settings
    azurite_endpoint: str = os.getenv('AZURITE_BLOB_ENDPOINT', 'http://127.0.0.1:10000/devstoreaccount1')
    azurite_access_key: str = os.getenv('AZURITE_ACCOUNT_NAME', 'devstoreaccount1')
    azurite_secret_key: str = os.getenv('AZURITE_ACCOUNT_KEY', '')
    azurite_connection_string: str = os.getenv("AZURITE_CONNECTION_STRING", "")
    azurite_data_bucket: str = 'computer-vision-data'
    azurite_model_bucket: str = 'computer-vision-models'
    use_azurite: bool = True

    # Visualization settings
    box_color: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    gt_box_color: Tuple[int, int, int] = (255, 0, 0)  # Blue for ground truth
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    draw_ground_truth: bool = False  # Draw GT boxes alongside predictions

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# mAP CALCULATION UTILITIES
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-10)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-10)


def calculate_ap_voc(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate AP using VOC 2010+ method (all-point interpolation)."""
    # Prepend sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    recall_changes = np.where(recalls[1:] != recalls[:-1])[0]

    # Sum (delta recall) * precision
    ap = np.sum((recalls[recall_changes + 1] - recalls[recall_changes]) * precisions[recall_changes + 1])

    return ap


def calculate_ap_11point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate AP using 11-point interpolation (Pascal VOC 2007 style)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precisions[recalls >= t]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall)
    return ap / 11


class MAPCalculator:
    """
    Mean Average Precision calculator for object detection.

    Collects predictions and ground truths, then computes mAP.
    """

    def __init__(
        self, iou_thresholds: List[float] = [0.5], use_11_point: bool = False,
        recall_fppi: float = 0.1,
    ):
        """
        Args:
            iou_thresholds: IoU thresholds for matching (e.g., [0.5] for mAP@50)
            use_11_point: Use 11-point interpolation (VOC 2007) vs all-point (VOC 2010+)
        """
        self.iou_thresholds = iou_thresholds
        self.use_11_point = use_11_point
        self.recall_fppi = recall_fppi
        self.reset()

    def reset(self):
        """Reset all collected data."""
        self.predictions = []  # List of {'image_id', 'class_id', 'score', 'box'}
        self.ground_truths = []  # List of {'image_id', 'class_id', 'box'}
        self.ignore_regions = {}
        self.image_ids = set()

    def add_image(self, image_id: int):
        self.image_ids.add(image_id)

    def add_predictions(self, image_id: int, detections: List[Dict]):
        """
        Add predictions for an image.

        Args:
            image_id: Unique image identifier
            detections: List of {'box': [x1,y1,x2,y2], 'score': float, 'class': int}
        """
        self.add_image(image_id)
        for det in detections:
            self.predictions.append({
                'image_id': image_id,
                'class_id': det['class'],
                'score': det['score'],
                'box': np.array(det['box'], dtype=np.float32)
            })

    def add_ground_truths(
        self, image_id: int, gt_boxes: List[List[float]], gt_classes: List[int],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add ground truth annotations for an image.

        Args:
            image_id: Unique image identifier
            gt_boxes: List of [x1, y1, x2, y2] boxes
            gt_classes: List of class IDs
        """
        self.add_image(image_id)
        metadata = metadata or [{} for _ in gt_boxes]
        if len(metadata) != len(gt_boxes):
            raise ValueError("Ground-truth metadata length must match boxes")
        for box, cls, item_metadata in zip(gt_boxes, gt_classes, metadata):
            self.ground_truths.append({
                'image_id': image_id,
                'class_id': cls,
                'box': np.array(box, dtype=np.float32),
                'metadata': dict(item_metadata),
            })

    def add_ignore_regions(self, image_id: int, boxes: List[List[float]]):
        """Register regions whose overlapping unmatched predictions are neutral."""
        self.add_image(image_id)
        self.ignore_regions.setdefault(image_id, []).extend(
            np.asarray(box, dtype=np.float32) for box in boxes
        )

    def _ignored_prediction(self, image_id: int, box: np.ndarray, threshold: float = 0.5) -> bool:
        regions = self.ignore_regions.get(image_id, [])
        if not regions:
            return False
        regions = np.asarray(regions)
        top_left = np.maximum(box[None, :2], regions[:, :2])
        bottom_right = np.minimum(box[None, 2:], regions[:, 2:])
        intersection = np.maximum(0, bottom_right - top_left).prod(axis=1)
        prediction_area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-10)
        return bool(np.max(intersection / prediction_area) >= threshold)

    def compute_map(self, verbose: bool = True) -> Dict:
        """
        Compute mAP over all collected predictions and ground truths.

        Args:
            verbose: Print per-class results

        Returns:
            Dictionary with mAP metrics
        """
        if not self.predictions or not self.ground_truths:
            if verbose:
                print("  No predictions or ground truths to evaluate")
            empty = {f'mAP@{t:.2f}': 0.0 for t in self.iou_thresholds}
            if len(self.iou_thresholds) > 1:
                empty['mAP@0.50:0.95'] = 0.0
            empty[f'Recall@FPPI={self.recall_fppi:.2f}'] = 0.0
            return empty

        results = {}

        # Get unique classes (excluding background class 0)
        classes = sorted(set(gt['class_id'] for gt in self.ground_truths if gt['class_id'] > 0))

        if verbose:
            print(f"\n  Evaluating {len(self.predictions)} predictions against "
                  f"{len(self.ground_truths)} ground truths")
            print(f"  Classes to evaluate: {classes}")

        for iou_thresh in self.iou_thresholds:
            aps = []

            for class_id in classes:
                ap, stats = self._compute_ap_for_class(class_id, iou_thresh)
                aps.append(ap)

                if verbose:
                    print(f"    Class {class_id}: AP@{iou_thresh:.2f} = {ap:.4f} "
                          f"(TP={stats['tp']}, FP={stats['fp']}, GT={stats['num_gt']}, "
                          f"Preds={stats['num_preds']})")

            mean_ap = float(np.mean(aps)) if aps else 0.0
            results[f'mAP@{iou_thresh:.2f}'] = mean_ap

        # COCO-style mAP (average over IoU thresholds)
        if len(self.iou_thresholds) > 1:
            results['mAP@0.50:0.95'] = float(np.mean(
                [results[f'mAP@{t:.2f}'] for t in self.iou_thresholds]
            ))

        if classes:
            _, recall_stats = self._compute_ap_for_class(classes[0], 0.5)
            results[f'Recall@FPPI={self.recall_fppi:.2f}'] = recall_stats['recall_at_fppi']
        return results

    def _compute_ap_for_class(self, class_id: int, iou_threshold: float) -> Tuple[float, Dict]:
        """Compute AP for a single class at a given IoU threshold."""
        # Get predictions and GTs for this class
        class_preds = [p for p in self.predictions if p['class_id'] == class_id]
        class_gts = [g for g in self.ground_truths if g['class_id'] == class_id]

        stats = {
            'num_preds': len(class_preds),
            'num_gt': len(class_gts),
            'tp': 0,
            'fp': 0,
            'recall_at_fppi': 0.0,
        }

        if len(class_gts) == 0:
            return 0.0, stats

        if len(class_preds) == 0:
            return 0.0, stats

        # Sort predictions by score (descending)
        class_preds.sort(key=lambda x: x['score'], reverse=True)

        # Group GTs by image
        gt_by_image = {}
        for gt in class_gts:
            img_id = gt['image_id']
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt['box'])

        # Track which GTs have been matched (per image)
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in gt_by_image.items()}

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for pred_idx, pred in enumerate(class_preds):
            pred_box = pred['box']
            pred_img_id = pred['image_id']

            if pred_img_id not in gt_by_image:
                if not self._ignored_prediction(pred_img_id, pred_box):
                    fp[pred_idx] = 1
                continue

            img_gt_boxes = np.array(gt_by_image[pred_img_id])

            # Compute IoU with all GTs in this image
            ious = compute_iou_matrix(pred_box[None, :], img_gt_boxes)[0]

            # Find best matching GT
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_threshold and not gt_matched[pred_img_id][best_iou_idx]:
                tp[pred_idx] = 1
                gt_matched[pred_img_id][best_iou_idx] = True
            else:
                if not self._ignored_prediction(pred_img_id, pred_box):
                    fp[pred_idx] = 1

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # Calculate AP
        if self.use_11_point:
            ap = calculate_ap_11point(recalls, precisions)
        else:
            ap = calculate_ap_voc(recalls, precisions)

        stats['tp'] = int(tp.sum())
        stats['fp'] = int(fp.sum())
        image_count = max(len(self.image_ids), 1)
        within_fppi = fp_cumsum / image_count <= self.recall_fppi
        if np.any(within_fppi):
            stats['recall_at_fppi'] = float(np.max(recalls[within_fppi]))


        return ap, stats

    def compute_slices(self) -> Dict[str, Dict[str, float]]:
        """Evaluate natural-distribution size, source-label, and visibility slices."""
        dimensions = ("size", "sourceLabel", "visibility")
        observed = {
            (dimension, ground_truth["metadata"].get(dimension))
            for ground_truth in self.ground_truths
            for dimension in dimensions
            if isinstance(ground_truth["metadata"].get(dimension), str)
        }
        results = {}
        for dimension, value in sorted(observed):
            calculator = MAPCalculator(
                self.iou_thresholds,
                use_11_point=self.use_11_point,
                recall_fppi=self.recall_fppi,
            )
            calculator.image_ids = set(self.image_ids)
            calculator.predictions = list(self.predictions)
            for image_id, boxes in self.ignore_regions.items():
                calculator.add_ignore_regions(image_id, boxes)
            for ground_truth in self.ground_truths:
                metadata = ground_truth["metadata"]
                if metadata.get(dimension) == value:
                    calculator.ground_truths.append(ground_truth)
                else:
                    calculator.add_ignore_regions(
                        ground_truth["image_id"], [ground_truth["box"]]
                    )
            results[f"{dimension}/{value}"] = calculator.compute_map(verbose=False)
        return results

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'num_predictions': len(self.predictions),
            'num_ground_truths': len(self.ground_truths),
            'num_images': len(self.image_ids),
            'num_images_with_preds': len(set(p['image_id'] for p in self.predictions)),
            'num_images_with_gt': len(set(g['image_id'] for g in self.ground_truths))
        }


# ============================================================================
# ANNOTATION PARSING
# ============================================================================

def parse_yolo_annotation(
    label_data: Optional[bytes],
    image_width: int,
    image_height: int,
    min_box_size: int = 5
) -> Tuple[List[List[float]], List[int]]:
    """
    Parse YOLO format annotation to absolute coordinates.

    YOLO format: class_id center_x center_y width height (all normalized 0-1)

    Args:
        label_data: Raw bytes of annotation file
        image_width: Original image width
        image_height: Original image height
        min_box_size: Minimum box dimension in pixels

    Returns:
        (boxes, classes) - boxes in [x1, y1, x2, y2] format, class IDs
    """
    boxes = []
    classes = []

    if label_data is None:
        return boxes, classes

    try:
        content = label_data.decode('utf-8')
    except:
        return boxes, classes

    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            yolo_class = int(parts[0])  # 0 = person in YOLO
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
        except ValueError:
            continue

        # Validate normalized coordinates
        if not (0 <= cx_norm <= 1 and 0 <= cy_norm <= 1 and
                0 < w_norm <= 1 and 0 < h_norm <= 1):
            continue

        # Convert to absolute coordinates
        cx = cx_norm * image_width
        cy = cy_norm * image_height
        w = w_norm * image_width
        h = h_norm * image_height

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Clamp to image bounds
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        # Filter tiny boxes
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue

        boxes.append([x1, y1, x2, y2])
        # YOLO class 0 -> our class 1 (0 is background in SSD)
        classes.append(yolo_class + 1)

    return boxes, classes


def load_canonical_for_evaluation(data: bytes | None, record: Dict):
    """Load and validate a complete canonical sidecar for evaluation."""
    checksums = record.get("checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Split record is missing checksums")
    if (not isinstance(checksums.get("image"), str) or len(checksums["image"]) != 64 or
            not isinstance(checksums.get("annotation"), str) or len(checksums["annotation"]) != 64 or
            not isinstance(record.get("personCount"), int) or record["personCount"] < 0 or
            not isinstance(record.get("ignoredCount"), int) or record["ignoredCount"] < 0):
        raise ValueError("Split record has invalid checksums or object counts")

    return parse_canonical_annotation(
        data,
        expected_image_blob=record["image"],
        expected_image_sha256=checksums.get("image"),
        expected_status=record.get("labelStatus"),
        expected_sidecar_sha256=checksums.get("annotation"),
        expected_person_count=record.get("personCount"),
        expected_ignored_count=record.get("ignoredCount"),
    )

def parse_canonical_for_evaluation(data: bytes | None, record: Dict) -> Tuple[List, List, List]:
    """Return full person boxes, SSD class IDs, and ignore boxes from a sidecar."""
    annotation = load_canonical_for_evaluation(data, record)
    boxes = [obj.full_box for obj in annotation.objects]
    return boxes, [1] * len(boxes), annotation.ignore_regions



def get_label_path_from_image_path(image_path: str) -> str:
    """
    Convert image path to corresponding label path.

    Args:
        image_path: e.g., "datasets/citypersons/v2026-09-07/images/val/city/image.png"

    Returns:
        Label path: e.g., "datasets/.../labels/yolo-person-v1/val/city/image.txt"
    """
    marker = "/images/"
    if marker not in image_path:
        raise ValueError(f"Versioned image path does not contain {marker!r}: {image_path}")
    prefix, relative_path = image_path.split(marker, 1)
    label_path = f"{prefix}/labels/yolo-person-v1/{relative_path}"
    label_path = str(Path(label_path).with_suffix('.txt'))
    return label_path


# ============================================================================
# DETECTION PREDICTOR (unchanged from original)
# ============================================================================

class DetectionPredictor(DetectionBackend):
    """
    Handles model inference and post-processing for SSD detections
    """

    def __init__(self, model: nn.Module, anchors: torch.Tensor, config: InferenceConfig):
        self.model = model
        self.anchors = anchors
        self.config = config
        self.device = config.device

        # Move anchors to device
        self.anchors = self.anchors.to(self.device)

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for inference"""
        original_size = (image.shape[0], image.shape[1])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, scale, pad_x, pad_y = preprocess_rgb_image(
            image_rgb, self.config.input_height, self.config.input_width,
            add_batch=True, return_geometry=True
        )
        tensor = torch.from_numpy(tensor).to(self.device)

        return tensor, (*original_size, scale, pad_x, pad_y)

    def decode_boxes(
        self, pred_boxes: torch.Tensor, anchors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode predicted box offsets to absolute coordinates"""
        anchors = self.anchors if anchors is None else anchors
        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        anchor_w = anchors[:, 2]
        anchor_h = anchors[:, 3]

        dx, dy, dw, dh = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

        dw = torch.clamp(dw, max=10.0)
        dh = torch.clamp(dh, max=10.0)

        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h

        x1 = (pred_cx - pred_w / 2) * self.config.input_width
        y1 = (pred_cy - pred_h / 2) * self.config.input_height
        x2 = (pred_cx + pred_w / 2) * self.config.input_width
        y2 = (pred_cy + pred_h / 2) * self.config.input_height

        return torch.stack([x1, y1, x2, y2], dim=1)

    def postprocess(
        self,
        pred_cls: torch.Tensor,
        pred_boxes: torch.Tensor,
        original_size: Tuple[int, int],
        score_threshold: float = None
    ) -> List[Dict]:
        """Post-process predictions: decode boxes, apply NMS, filter by confidence"""
        if score_threshold is None:
            score_threshold = self.config.confidence_threshold

        pred_cls = pred_cls[0]
        pred_boxes = pred_boxes[0]

        person_scores = person_scores_from_logits(pred_cls)

        mask = person_scores > score_threshold
        filtered_scores = person_scores[mask]
        filtered_offsets = pred_boxes[mask]
        filtered_anchors = self.anchors[mask]
        if len(filtered_scores) == 0:
            return []
        if len(filtered_scores) > self.config.pre_nms_topk:
            filtered_scores, top_indices = filtered_scores.topk(self.config.pre_nms_topk)
            filtered_offsets = filtered_offsets[top_indices]
            filtered_anchors = filtered_anchors[top_indices]
        filtered_boxes = self.decode_boxes(filtered_offsets, filtered_anchors)

        keep_indices = nms(filtered_boxes, filtered_scores, self.config.nms_threshold)
        keep_indices = keep_indices[:self.config.max_detections]

        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]

        orig_h, orig_w, scale, pad_x, pad_y = original_size

        detections = []
        for i in range(len(final_boxes)):
            box = final_boxes[i].cpu().numpy()
            score = final_scores[i].cpu().item()

            x1 = int(max(0, min((box[0] - pad_x) / scale, orig_w)))
            y1 = int(max(0, min((box[1] - pad_y) / scale, orig_h)))
            x2 = int(max(0, min((box[2] - pad_x) / scale, orig_w)))
            y2 = int(max(0, min((box[3] - pad_y) / scale, orig_h)))

            detections.append({
                'box': [x1, y1, x2, y2],
                'score': score,
                'class': 1
            })

        return detections

    @torch.no_grad()
    def predict(self, image: np.ndarray, score_threshold: float = None) -> List[Dict]:
        """Run full inference pipeline on an image"""
        self.model.eval()

        tensor, original_size = self.preprocess(image)
        pred_cls, pred_boxes = self.model(tensor)
        detections = self.postprocess(pred_cls, pred_boxes, original_size, score_threshold)

        return detections


class OpenVINOPredictor(DetectionBackend):
    """OpenVINO-based predictor for INT8 models"""

    def __init__(self, model_path: str, anchors: torch.Tensor, config: InferenceConfig):
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO not installed. Install with: pip install openvino")

        self.config = config
        self.anchors = anchors.numpy()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"Loading OpenVINO model from: {model_path}")
        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")

        self.input_layer = self.compiled_model.input(0)
        self.output_layers = self.compiled_model.outputs
        input_shape = tuple(int(value) for value in self.input_layer.shape)
        expected_shape = (1, 3, config.input_height, config.input_width)
        if input_shape != expected_shape:
            raise ValueError(f"OpenVINO input shape {input_shape} does not match {expected_shape}")
        self.cls_output = next(
            output for output in self.output_layers if int(output.shape[-1]) in (1, 2)
        )
        self.box_output = next(
            output for output in self.output_layers if int(output.shape[-1]) == 4
        )

        print("✓ OpenVINO model loaded")
        print(f"  Input shape: {self.input_layer.shape}")
        print(f"  Outputs: {len(self.output_layers)}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for OpenVINO inference"""
        original_size = (image.shape[0], image.shape[1])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_batch, scale, pad_x, pad_y = preprocess_rgb_image(
            image_rgb, self.config.input_height, self.config.input_width,
            add_batch=True, return_geometry=True
        )

        return image_batch, (*original_size, scale, pad_x, pad_y)

    def decode_boxes(
        self, pred_boxes: np.ndarray, anchors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Decode predicted box offsets to absolute coordinates"""
        anchors = self.anchors if anchors is None else anchors
        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        anchor_w = anchors[:, 2]
        anchor_h = anchors[:, 3]

        dx, dy, dw, dh = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

        dw = np.clip(dw, -10.0, 10.0)
        dh = np.clip(dh, -10.0, 10.0)

        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_w = np.exp(dw) * anchor_w
        pred_h = np.exp(dh) * anchor_h

        x1 = (pred_cx - pred_w / 2) * self.config.input_width
        y1 = (pred_cy - pred_h / 2) * self.config.input_height
        x2 = (pred_cx + pred_w / 2) * self.config.input_width
        y2 = (pred_cy + pred_h / 2) * self.config.input_height

        return np.stack([x1, y1, x2, y2], axis=1)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numpy softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Numpy NMS implementation"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def postprocess(
        self,
        pred_cls: np.ndarray,
        pred_boxes: np.ndarray,
        original_size: Tuple[int, int],
        score_threshold: float = None
    ) -> List[Dict]:
        """Post-process OpenVINO predictions"""
        if score_threshold is None:
            score_threshold = self.config.confidence_threshold

        pred_cls = pred_cls[0]
        pred_boxes = pred_boxes[0]

        if pred_cls.shape[-1] == 1:
            person_scores = 1.0 / (1.0 + np.exp(-pred_cls[:, 0]))
        elif pred_cls.shape[-1] == 2:
            person_scores = self.softmax(pred_cls, axis=1)[:, 1]
        else:

            raise ValueError(
                f"Expected one quality logit or two legacy logits, got {pred_cls.shape}"
            )
        mask = person_scores > score_threshold
        filtered_scores = person_scores[mask]
        filtered_offsets = pred_boxes[mask]
        filtered_anchors = self.anchors[mask]
        if len(filtered_scores) == 0:
            return []
        if len(filtered_scores) > self.config.pre_nms_topk:
            top_indices = np.argpartition(
                filtered_scores, -self.config.pre_nms_topk
            )[-self.config.pre_nms_topk:]
            filtered_scores = filtered_scores[top_indices]
            filtered_offsets = filtered_offsets[top_indices]
            filtered_anchors = filtered_anchors[top_indices]
        filtered_boxes = self.decode_boxes(filtered_offsets, filtered_anchors)

        keep_indices = self.nms_numpy(filtered_boxes, filtered_scores, self.config.nms_threshold)
        keep_indices = keep_indices[:self.config.max_detections]

        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]

        orig_h, orig_w, scale, pad_x, pad_y = original_size

        detections = []
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            score = final_scores[i]

            x1 = int(np.clip((box[0] - pad_x) / scale, 0, orig_w))
            y1 = int(np.clip((box[1] - pad_y) / scale, 0, orig_h))
            x2 = int(np.clip((box[2] - pad_x) / scale, 0, orig_w))
            y2 = int(np.clip((box[3] - pad_y) / scale, 0, orig_h))

            detections.append({
                'box': [x1, y1, x2, y2],
                'score': float(score),
                'class': 1
            })

        return detections

    def predict(self, image: np.ndarray, score_threshold: float = None) -> List[Dict]:
        """Run full OpenVINO inference pipeline"""
        input_tensor, original_size = self.preprocess(image)
        results = self.compiled_model([input_tensor])

        pred_cls = results[self.cls_output]
        pred_boxes = results[self.box_output]

        detections = self.postprocess(pred_cls, pred_boxes, original_size, score_threshold)

        return detections


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    config: InferenceConfig,
    ground_truths: List[List[float]] = None
) -> np.ndarray:
    """
    Draw predicted bounding boxes on image, optionally with ground truth.

    Args:
        image: BGR image
        detections: List of detection dicts
        config: Visualization config
        ground_truths: Optional list of GT boxes [x1, y1, x2, y2]

    Returns:
        Image with bounding boxes drawn
    """
    image = image.copy()

    # Draw ground truth boxes first (so predictions overlay them)
    if ground_truths and config.draw_ground_truth:
        for gt_box in ground_truths:
            x1, y1, x2, y2 = [int(v) for v in gt_box]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                config.gt_box_color,
                config.box_thickness
            )

            # Label as GT
            cv2.putText(
                image,
                "GT",
                (x1 + 2, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.gt_box_color,
                1
            )

    # Draw predictions
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            config.box_color,
            config.box_thickness
        )

        label = f"person {score:.2f}"

        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.font_thickness
        )

        cv2.rectangle(
            image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 5, y1),
            config.box_color,
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            (0, 0, 0),
            config.font_thickness
        )

    return image


# ============================================================================
# AZURITE CLIENT
# ============================================================================

class AzuriteClientLocal:
    """Local Azurite client implementation for standalone use"""

    def __init__(self, config):
        self.config = config
        self.client = None

        if config.use_azurite:
            try:
                from person_detection.data.storage import AzuriteBlobCompat
                self.client = AzuriteBlobCompat(
                    endpoint=str(config.azurite_endpoint),
                    access_key=config.azurite_access_key,
                    secret_key=config.azurite_secret_key,
                    secure=False
                )
                self.client.bucket_exists(bucket_name=str(config.azurite_data_bucket))
                print(f"✓ Connected to Azurite at {config.azurite_endpoint}")
            except Exception as e:
                print(f"✗ Azurite connection failed: {e}")
                print("  Falling back to local filesystem")
                self.client = None

    def list_objects(self, bucket: str, prefix: str) -> List[str]:
        if self.client is None:
            local_path = Path(self.config.data_root) / prefix
            if local_path.exists():
                files = []
                for p in local_path.rglob('*'):
                    if p.is_file():
                        rel_path = str(p.relative_to(self.config.data_root))
                        files.append(rel_path)
                return files
            return []

        try:
            objects = self.client.list_objects(
                bucket_name=str(bucket),
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def get_object_bytes(self, bucket: str, object_name: str) -> Optional[bytes]:
        if self.client is None:
            local_path = Path(self.config.data_root) / object_name
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    return f.read()
            return None

        try:
            response = self.client.get_object(
                bucket_name=str(bucket),
                object_name=str(object_name)
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception:
            return None


# ============================================================================
# MODEL LOADING
# ============================================================================

def detect_model_type(model_path: str) -> str:
    """Detect model type from file extension and content"""
    if model_path.endswith('.xml'):
        return 'openvino'
    elif model_path.endswith('.pth') or model_path.endswith('.pt'):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            for key in state_dict.keys():
                if '__nncf' in key:
                    return 'nncf'

        return 'pytorch'
    else:
        return 'pytorch'


def validate_checkpoint_contract(
    checkpoint: dict, config: InferenceConfig, model: nn.Module
) -> None:
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint lacks the rectangular model contract")
    if checkpoint.get("modelFormatVersion") != MODEL_FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint format {checkpoint.get('modelFormatVersion')} is incompatible with "
            f"rectangular model format {MODEL_FORMAT_VERSION}"
        )
    checkpoint_config = checkpoint.get("config", {})
    expected = (config.input_height, config.input_width)
    actual = (
        checkpoint_config.get("input_height"),
        checkpoint_config.get("input_width"),
    )
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint input contract {actual} does not match requested {expected}"
        )
    if checkpoint.get("anchors") != model.anchor_generator.specification():
        raise RuntimeError("Checkpoint anchors do not match the evaluation model")


def load_model(config: InferenceConfig) -> Tuple[Union[nn.Module, 'OpenVINOPredictor'], torch.Tensor, str]:
    """Load trained model from checkpoint"""
    print(f"\nLoading model from: {config.model_path}")

    if config.model_type == 'auto':
        model_type = detect_model_type(config.model_path)
    else:
        model_type = config.model_type

    print(f"Model type: {model_type}")

    if not IMPORTS_AVAILABLE:
        raise ImportError("Cannot import from ssd_person_detection.py")

    base_model = SSDPersonDetector(
        num_classes=config.num_classes,
        pretrained=False,
        input_height=config.input_height,
        input_width=config.input_width,
    )
    anchors = base_model.anchor_generator.get_anchors()

    if model_type == 'openvino':
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO not installed. Install with: pip install openvino")

        predictor = OpenVINOPredictor(config.model_path, anchors, config)
        return predictor, anchors, 'openvino'

    elif model_type == 'nncf':
        if not HAS_NNCF:
            openvino_path = config.model_path.replace('_int8.pth', '_int8.xml').replace('.pth', '_int8.xml')
            if os.path.exists(openvino_path) and HAS_OPENVINO:
                print(f"  NNCF not installed, but found OpenVINO model at: {openvino_path}")
                print("  Using OpenVINO for INT8 inference instead.")
                predictor = OpenVINOPredictor(openvino_path, anchors, config)
                return predictor, anchors, 'openvino'
            else:
                raise RuntimeError(
                    "NNCF not installed and no OpenVINO IR found.\n"
                    "Options:\n"
                    "  1. Install NNCF: pip install nncf\n"
                    "  2. Use the OpenVINO IR model: person_detector_int8.xml\n"
                    "  3. Use the FP32 model: best_model_fp32.pth"
                )

        print("Loading NNCF quantized model...")

        checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=True)
        validate_checkpoint_contract(checkpoint, config, base_model)
        state_dict = checkpoint["model_state_dict"]

        class DummyDataset:
            def __iter__(self):
                for _ in range(1):
                    yield torch.randn(1, 3, config.input_height, config.input_width)
            def __len__(self):
                return 1

        def transform_fn(x):
            return x

        dummy_dataset = nncf.Dataset(DummyDataset(), transform_fn)

        base_model = base_model.cpu()
        quantized_model = nncf.quantize(
            base_model,
            dummy_dataset,
            subset_size=1,
            fast_bias_correction=False
        )

        quantized_model.load_state_dict(state_dict)
        quantized_model = quantized_model.to(config.device)
        quantized_model.eval()

        print("✓ NNCF quantized model loaded")
        return quantized_model, anchors, 'nncf'

    else:  # pytorch
        model = base_model

        if os.path.exists(config.model_path):
            checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=True)
            validate_checkpoint_contract(checkpoint, config, model)
            load_detector_state_dict(model, checkpoint["model_state_dict"])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if "best_val_loss" in checkpoint:
                print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            print(f"⚠ Warning: Model file not found at {config.model_path}")
            print("  Using randomly initialized model (for testing pipeline only)")

        model = model.to(config.device)
        model.eval()

        return model, anchors, 'pytorch'


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

class DetectionEvaluationWorkflow:
    """Run one detector backend over a canonical split and report all metrics."""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def run(self):
        config = self.config
        print("=" * 70)
        print("SSD Person Detection - Test Inference with mAP Evaluation")
        print("=" * 70)
        print(f"Device: {config.device}")
        print(f"Model: {config.model_path}")
        print(f"Confidence threshold: {config.confidence_threshold}")
        print(f"NMS threshold: {config.nms_threshold}")
        print(f"Evaluate mAP: {config.evaluate_map}")
        if config.evaluate_map:
            print(f"  IoU thresholds: {config.map_iou_thresholds}")
            print(f"  Score threshold for mAP: {config.map_score_threshold}")
        print(f"Output directory: {config.output_dir}")
        print("=" * 70)

        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created output directory: {output_path}")

        # Load model
        model_or_predictor, anchors, model_type = load_model(config)

        # Create predictor based on model type
        if model_type == 'openvino':
            predictor = model_or_predictor
            print("Using OpenVINO inference backend")
        else:
            predictor = DetectionPredictor(model_or_predictor, anchors, config)
            print(f"Using PyTorch inference backend ({model_type})")

        # Initialize Azurite client
        print("\nInitializing data client...")
        azurite_client = AzuriteClientLocal(config)

        # Initialize mAP calculator
        map_calculator = None
        official_accumulator = None
        if config.evaluate_map:
            map_calculator = MAPCalculator(
                iou_thresholds=config.map_iou_thresholds,
                use_11_point=False,
                recall_fppi=config.recall_fppi,
            )
            official_accumulator = OfficialCityPersonsAccumulator()
            print("✓ mAP calculator initialized")

        # Find test images
        print("\nSearching for test images...")

        if config.evaluate_map:
            # Prefer val/train which have annotations
            splits_to_try = ['val', 'train']
        else:
            # For inference-only, test is fine
            splits_to_try = ['test', 'val', 'train']

        image_paths = []
        records_by_image = {}
        used_split = None

        bucket = config.azurite_data_bucket
        read_blob = lambda name: azurite_client.get_object_bytes(bucket, name)
        dataset_prefix = resolve_citypersons_prefix(read_blob)

        dataset_manifest, dataset_manifest_sha256 = load_citypersons_manifest(read_blob, dataset_prefix)

        for split in splits_to_try:
            records = load_citypersons_split(read_blob, dataset_prefix, split)
            image_paths = [
                version_blob(dataset_prefix, record["image"])
                for record in records
                if isinstance(record.get("image"), str)
            ]
            records_by_image = {
                version_blob(dataset_prefix, record["image"]): record
                for record in records
                if isinstance(record.get("image"), str)
            }

            if image_paths:
                used_split = split
                print(f"✓ Found {len(image_paths)} images in '{split}' split")
                break
            else:
                print(f"  No images found in '{split}' split")

        if not image_paths:
            print("✗ No images found in any split!")
            return

        # Process images
        num_to_process = min(len(image_paths), config.max_images)
        if config.official_evaluator_dir and (
            used_split != "val" or num_to_process != len(image_paths)
        ):
            raise RuntimeError(
                "Official CityPersons metrics require the complete validation manifest"
            )
        print(f"\nProcessing {num_to_process} images from '{used_split}' split...")

        total_detections = 0
        images_with_detections = 0
        total_ground_truths = 0
        images_with_gt = 0

        # Use tqdm for progress bar
        for idx, img_path in enumerate(tqdm(image_paths[:num_to_process], desc="Processing")):
            # Load image
            img_data = azurite_client.get_object_bytes(config.azurite_data_bucket, img_path)

            if img_data is None:
                if config.evaluate_map:
                    raise RuntimeError(f"Evaluation image is missing: {img_path}")
                continue

            # Decode image
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                if config.evaluate_map:
                    raise RuntimeError(f"Evaluation image cannot be decoded: {img_path}")
                continue

            img_h, img_w = image.shape[:2]

            # Load ground truth annotations (if evaluating mAP)
            gt_boxes = []
            gt_classes = []
            ignore_regions = []
            annotation = None
            gt_metadata = []

            if config.evaluate_map:
                record = records_by_image[img_path]
                annotation_path = version_blob(dataset_prefix, record["annotation"])
                annotation_data = azurite_client.get_object_bytes(
                    config.azurite_data_bucket, annotation_path
                )
                annotation = load_canonical_for_evaluation(annotation_data, record)
                gt_boxes = []
                gt_metadata = []
                for obj in annotation.objects:
                    clipped = clip_box_to_image(
                        obj.full_box, annotation.width, annotation.height
                    )
                    if clipped is None:
                        continue
                    gt_boxes.append(clipped)
                    gt_metadata.append({
                        "size": size_slice(obj.full_box),
                        "sourceLabel": obj.source_label,
                        "visibility": obj.attributes.get("visibility", "unknown"),
                    })
                gt_classes = [1] * len(gt_boxes)
                ignore_regions = []
                for box in annotation.ignore_regions:
                    clipped = clip_box_to_image(
                        box, annotation.width, annotation.height
                    )
                    if clipped is not None:
                        ignore_regions.append(clipped)
                map_calculator.add_image(idx)
                map_calculator.add_ignore_regions(idx, ignore_regions)

                if gt_boxes:
                    map_calculator.add_ground_truths(idx, gt_boxes, gt_classes, gt_metadata)
                    total_ground_truths += len(gt_boxes)
                    images_with_gt += 1

            # Run inference with lower threshold for mAP calculation
            if config.evaluate_map:
                # Get detections with low threshold for mAP
                detections_for_map = predictor.predict(image, score_threshold=config.map_score_threshold)
                map_calculator.add_predictions(idx, detections_for_map)
                official_accumulator.add_image(idx, record["image"], annotation, detections_for_map)

                # Get detections with normal threshold for visualization
                detections_for_viz = [d for d in detections_for_map if d['score'] >= config.confidence_threshold]
            else:
                detections_for_viz = predictor.predict(image)

            # Update stats
            num_dets = len(detections_for_viz)
            total_detections += num_dets
            if num_dets > 0:
                images_with_detections += 1

            # Draw detections
            image_with_boxes = draw_detections(
                image,
                detections_for_viz,
                config,
                ground_truths=gt_boxes if config.draw_ground_truth else None
            )

            # Create output filename
            safe_name = img_path.replace('/', '_').replace('\\', '_')
            output_filename = output_path / safe_name

            # Save image
            cv2.imwrite(str(output_filename), image_with_boxes)

        # Print summary
        print("\n" + "=" * 70)
        print("Inference Complete!")
        print("=" * 70)
        print(f"Images processed: {num_to_process}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Total detections (conf >= {config.confidence_threshold}): {total_detections}")
        print(f"Average detections per image: {total_detections / num_to_process:.2f}")

        if config.evaluate_map:
            print(f"\nGround Truth Statistics:")
            print(f"  Images with annotations: {images_with_gt}")
            print(f"  Total ground truth boxes: {total_ground_truths}")

        print(f"\nOutput saved to: {output_path}")

        # Calculate and print mAP
        if config.evaluate_map and map_calculator:
            print("\n" + "=" * 70)
            print("mAP Evaluation Results")
            print("=" * 70)

            summary = map_calculator.get_summary()
            print(f"\nEvaluation Summary:")
            print(f"  Total predictions: {summary['num_predictions']}")
            print(f"  Total ground truths: {summary['num_ground_truths']}")
            print(f"  Images with predictions: {summary['num_images_with_preds']}")
            print(f"  Images with GT: {summary['num_images_with_gt']}")

            print("\nComputing mAP...")
            map_results = map_calculator.compute_map(verbose=True)
            slice_results = map_calculator.compute_slices()
            official_gt, official_detections = official_accumulator.write(output_path)
            official_results = {
                "status": "inputs_generated",
                "groundTruth": str(official_gt),
                "detections": str(official_detections),
            }
            if config.official_evaluator_dir:
                if used_split != "val":
                    raise RuntimeError("Official CityPersons metrics may only be reported on val")
                official_results = {
                    "status": "complete",
                    **PinnedCityPersonsEvaluator(Path(config.official_evaluator_dir)).evaluate(
                        official_accumulator, output_path
                    ),
                }
            evaluation_report = {
                "dataset": {
                    "versionPrefix": dataset_prefix,
                    "manifestSha256": dataset_manifest_sha256,
                    "schemaVersion": dataset_manifest.get("schemaVersion"),
                    "split": used_split,
                    "records": num_to_process,
                },
                "preprocessing": {
                    **PRODUCTION_PREPROCESSING,
                    "inputHeight": config.input_height,
                    "inputWidth": config.input_width,
                },
                "postprocessing": {
                    "evaluationScoreThreshold": config.map_score_threshold,
                    "nmsThreshold": config.nms_threshold,
                    "preNmsTopK": config.pre_nms_topk,
                    "maxDetections": config.max_detections,
                },
                "projectBinary": {"metrics": map_results, "slices": slice_results},
                "officialCityPersons": official_results,
            }
            metrics_json = output_path / "evaluation_metrics.json"
            metrics_json.write_text(
                json.dumps(evaluation_report, indent=2) + "\n", encoding="utf-8"
            )

            print("\n" + "-" * 40)
            print("Final Results:")
            print("-" * 40)
            for metric, value in map_results.items():
                print(f"  {metric}: {value:.4f}")

            # Save results to file
            results_file = output_path / 'map_results.txt'
            with open(results_file, 'w') as f:
                f.write("mAP Evaluation Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Model: {config.model_path}\n")
                f.write(f"Split: {used_split}\n")
                f.write(f"Images evaluated: {num_to_process}\n")
                f.write(f"Confidence threshold (viz): {config.confidence_threshold}\n")
                f.write(f"Input canvas (H x W): {config.input_height} x {config.input_width}\n")
                f.write(f"Score threshold (mAP): {config.map_score_threshold}\n")
                f.write(f"Pre-NMS top-K: {config.pre_nms_topk}\n")
                f.write(f"Dataset version: {dataset_prefix}\n")
                f.write(f"Dataset schema: {dataset_manifest.get('schemaVersion')}\n")
                f.write(f"Manifest SHA256: {dataset_manifest_sha256}\n")
                f.write(f"NMS threshold: {config.nms_threshold}\n")
                f.write(f"IoU thresholds: {config.map_iou_thresholds}\n\n")
                f.write("Results:\n")
                for metric, value in map_results.items():
                    f.write(f"  {metric}: {value:.4f}\n")

            print(f"\nResults saved to: {results_file}")

            print(f"Machine-readable evaluation saved to: {metrics_json}")
        print("=" * 70)

        return map_results if config.evaluate_map else None


def run_inference(config: InferenceConfig):
    """Compatibility wrapper for the object-oriented evaluation workflow."""
    return DetectionEvaluationWorkflow(config).run()


def main():
    """Main entry point with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(description='SSD Person Detection - Test Inference with mAP')

    # Model settings
    parser.add_argument('--model-path', type=str, default='best_model_fp32.pth',
                        help='Path to model checkpoint (.pth for PyTorch, .xml for OpenVINO)')
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'pytorch', 'openvino', 'nncf'],
                        help='Model type: auto (detect), pytorch, openvino, or nncf')
    parser.add_argument('--input-height', type=int, default=360,
                        help='Model input canvas height')
    parser.add_argument('--input-width', type=int, default=640,
                        help='Model input canvas width (must exceed height)')

    # Inference settings
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for visualization')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--pre-nms-topk', type=int, default=1000,
                        help='Maximum scored candidates decoded and passed to NMS')

    # mAP evaluation settings
    parser.add_argument('--no-map', action='store_true',
                        help='Disable mAP evaluation')
    parser.add_argument('--map-threshold', type=float, default=0.01,
                        help='Score threshold for mAP calculation (lower than viz threshold)')
    parser.add_argument('--iou-thresholds', type=float, nargs='+', default=None,
                        help='IoU thresholds for mAP (e.g., 0.5 0.75 or 0.5 0.55 0.6 ... 0.95)')
    parser.add_argument('--coco-map', action='store_true',
                        help='Use COCO-style mAP@0.50:0.95 (10 IoU thresholds)')
    parser.add_argument('--recall-fppi', type=float, default=0.1,
                        help='False positives per image used for the reported recall operating point')
    parser.add_argument(
        '--official-evaluator-dir',
        default=os.getenv('CITYPERSONS_EVALUATOR_DIR'),
        help='Path to the official evaluation/eval_script directory')

    # Data settings
    parser.add_argument('--output-dir', type=str, default='./test_output',
                        help='Output directory for visualized images')
    parser.add_argument('--max-images', type=int, default=500,
                        help='Maximum number of images to process')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Local data root (if not using Azurite)')

    # Visualization settings
    parser.add_argument('--draw-gt', action='store_true',
                        help='Draw ground truth boxes alongside predictions')

    # Azurite settings
    parser.add_argument('--no-azurite', action='store_true',
                        help='Use local filesystem instead of Azurite')
    parser.add_argument('--azurite-endpoint', type=str, default=os.getenv('AZURITE_BLOB_ENDPOINT', 'http://127.0.0.1:10000/devstoreaccount1'),
                        help='Azurite endpoint')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Handle COCO mAP
    iou_thresholds = args.iou_thresholds
    if args.coco_map or iou_thresholds is None:
        iou_thresholds = [0.5 + i * 0.05 for i in range(10)]  # 0.50 to 0.95

    if args.input_width <= args.input_height:
        parser.error("--input-width must be greater than --input-height")
    if args.pre_nms_topk < 1:
        parser.error("--pre-nms-topk must be at least 1")

    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        input_height=args.input_height,
        input_width=args.input_width,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        pre_nms_topk=args.pre_nms_topk,
        evaluate_map=not args.no_map,
        map_iou_thresholds=iou_thresholds,
        map_score_threshold=args.map_threshold,
        recall_fppi=args.recall_fppi,
        official_evaluator_dir=args.official_evaluator_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        data_root=args.data_root,
        draw_ground_truth=args.draw_gt,
        use_azurite=not args.no_azurite,
        azurite_endpoint=args.azurite_endpoint,
        device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Run inference
    run_inference(config)


if __name__ == '__main__':
    main()
