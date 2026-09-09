"""Anchor-assignment interfaces and the ATSS implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class AssignmentResult:
    matched_boxes: torch.Tensor
    matched_labels: torch.Tensor
    positive_mask: torch.Tensor


class AnchorAssigner(ABC):
    """Assign model anchors to ground-truth objects and neutral regions."""

    @abstractmethod
    def assign(
        self,
        anchor_boxes: torch.Tensor,
        anchors_per_level: Sequence[int],
        ground_truth_boxes: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        ignore_boxes: torch.Tensor,
    ) -> AssignmentResult:
        """Return targets and a positive mask for every anchor."""


class ATSSAnchorAssigner(AnchorAssigner):
    """Adaptive Training Sample Selection with ignore-region neutralization."""

    def __init__(self, top_k: int = 9, ignore_overlap_threshold: float = 0.5):
        if top_k <= 0:
            raise ValueError("ATSS top_k must be positive")
        self.top_k = top_k
        self.ignore_overlap_threshold = ignore_overlap_threshold

    @staticmethod
    def anchors_overlapping_ignore(
        anchor_boxes: torch.Tensor,
        ignore_boxes: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        if ignore_boxes.numel() == 0:
            return torch.zeros(
                anchor_boxes.size(0), dtype=torch.bool, device=anchor_boxes.device
            )
        top_left = torch.maximum(anchor_boxes[:, None, :2], ignore_boxes[None, :, :2])
        bottom_right = torch.minimum(anchor_boxes[:, None, 2:], ignore_boxes[None, :, 2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
        anchor_area = (
            (anchor_boxes[:, 2] - anchor_boxes[:, 0]).clamp(min=1e-6)
            * (anchor_boxes[:, 3] - anchor_boxes[:, 1]).clamp(min=1e-6)
        )
        return (intersection / anchor_area[:, None]).amax(dim=1) >= threshold

    @staticmethod
    def _validate_levels(
        anchors_per_level: Sequence[int], num_anchors: int
    ) -> list[int]:
        counts = list(anchors_per_level) or [num_anchors]
        if any(count <= 0 for count in counts) or sum(counts) != num_anchors:
            raise ValueError("anchors_per_level does not match the anchor tensor")
        return counts

    def assign(
        self,
        anchor_boxes: torch.Tensor,
        anchors_per_level: Sequence[int],
        ground_truth_boxes: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        ignore_boxes: torch.Tensor,
    ) -> AssignmentResult:
        num_anchors = anchor_boxes.size(0)
        device = anchor_boxes.device
        level_counts = self._validate_levels(anchors_per_level, num_anchors)
        ignored = self.anchors_overlapping_ignore(
            anchor_boxes, ignore_boxes, self.ignore_overlap_threshold
        )
        if ground_truth_boxes.numel() == 0:
            labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            labels[ignored] = -1
            return AssignmentResult(
                anchor_boxes.new_zeros((num_anchors, 4)), labels, labels > 0
            )

        ious = box_iou(anchor_boxes, ground_truth_boxes)
        anchor_centers = (anchor_boxes[:, :2] + anchor_boxes[:, 2:]) / 2
        gt_centers = (ground_truth_boxes[:, :2] + ground_truth_boxes[:, 2:]) / 2
        distances = (
            (anchor_centers[:, None, :] - gt_centers[None, :, :]) ** 2
        ).sum(dim=2)

        candidate_mask = torch.zeros_like(ious, dtype=torch.bool)
        start = 0
        gt_indices = torch.arange(ground_truth_boxes.size(0), device=device)
        for count in level_counts:
            top_k = min(self.top_k, count)
            candidate_indices = (
                distances[start:start + count].topk(top_k, dim=0, largest=False).indices
                + start
            )
            candidate_mask[candidate_indices, gt_indices] = True
            start += count

        thresholds = torch.stack([
            ious[candidate_mask[:, gt_index], gt_index].mean()
            + ious[candidate_mask[:, gt_index], gt_index].std(unbiased=False)
            for gt_index in range(ground_truth_boxes.size(0))
        ])
        left = anchor_centers[:, None, 0] - ground_truth_boxes[None, :, 0]
        top = anchor_centers[:, None, 1] - ground_truth_boxes[None, :, 1]
        right = ground_truth_boxes[None, :, 2] - anchor_centers[:, None, 0]
        bottom = ground_truth_boxes[None, :, 3] - anchor_centers[:, None, 1]
        centers_inside = torch.stack([left, top, right, bottom], dim=2).amin(dim=2) > 0
        positives = candidate_mask & centers_inside & (ious >= thresholds[None, :])

        for gt_index in range(ground_truth_boxes.size(0)):
            if not positives[:, gt_index].any():
                candidates = torch.where(candidate_mask[:, gt_index])[0]
                best = candidates[ious[candidates, gt_index].argmax()]
                positives[best, gt_index] = True

        positive_quality = torch.where(positives, ious, torch.full_like(ious, -1))
        best_iou, best_gt_index = positive_quality.max(dim=1)
        positive_mask = best_iou >= 0
        matched_boxes = ground_truth_boxes[best_gt_index]
        matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        matched_labels[positive_mask] = ground_truth_labels[best_gt_index[positive_mask]]
        matched_labels[ignored & ~positive_mask] = -1
        return AssignmentResult(matched_boxes, matched_labels, positive_mask)
