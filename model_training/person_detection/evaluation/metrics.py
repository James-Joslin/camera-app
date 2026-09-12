"""Shared project detection metrics used by training and release evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np


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
