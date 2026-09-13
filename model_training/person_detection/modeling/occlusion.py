"""Training-only visibility and crowd-repulsion losses; dense-head adaptations of Repulsion Loss (CVPR 2018)."""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss


@dataclass
class OcclusionConfig:
    # Library defaults preserve detector-only training; production enables the auxiliary losses.
    visible_loss_weight: float = 0.0
    repgt_loss_weight: float = 0.0
    repbox_loss_weight: float = 0.0
    auxiliary_ramp_epochs: int = 5
    repgt_sigma: float = 0.5
    repbox_sigma: float = 0.0
    repbox_predictions_per_gt: int = 4
    repulsion_chunk_size: int = 64
    checkpoint_selection: str = "detection_loss"

    def validate_occlusion(self):
        weights = (self.visible_loss_weight, self.repgt_loss_weight, self.repbox_loss_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Occlusion loss weights must be finite and nonnegative")
        if any(weights) and self.model_variant != "clean_ltrb":
            raise ValueError("Visibility and crowd-repulsion training require the clean_ltrb detector")
        if self.auxiliary_ramp_epochs < 0:
            raise ValueError("auxiliary_ramp_epochs must be nonnegative")
        if not all(0 <= s < 1 for s in (self.repgt_sigma, self.repbox_sigma)):
            raise ValueError("Repulsion smoothing thresholds must be in [0, 1)")
        if min(self.repbox_predictions_per_gt, self.repulsion_chunk_size) < 1:
            raise ValueError("Repulsion sample and chunk limits must be positive")
        if self.checkpoint_selection not in ("ap", "detection_loss", "final"):
            raise ValueError("checkpoint_selection must be ap, detection_loss, or final")
        if self.checkpoint_selection == "ap" and self.validation_ap_every_n_epochs <= 0:
            raise ValueError("AP checkpoint selection requires validation AP")


def occlusion_recipe(config):
    recipe = {key: getattr(config, key) for key in asdict(OcclusionConfig())}
    recipe["coarse_dropout"] = config.visible_loss_weight == 0
    recipe["version"] = 1
    return recipe


def auxiliary_scale(epoch, ramp_epochs):
    """Zero-based epoch: first epoch zero, sixth epoch full for a five-epoch ramp."""
    return 1.0 if ramp_epochs == 0 else min(max(epoch / ramp_epochs, 0.0), 1.0)


def smooth_ln(overlap, sigma):
    overlap = overlap.clamp(0, 1)
    # Clamp the log branch even where it is not selected: avoid 0 * inf gradients.
    return torch.where(overlap <= sigma, -torch.log1p(-overlap.clamp(max=sigma)),
                       (overlap - sigma) / (1 - sigma) - math.log1p(-sigma))


def repgt_sum(boxes, owners, gt, sigma=0.5, chunk_size=64):
    total = boxes.sum() * 0
    if len(gt) < 2 or not len(boxes):
        return total, 0
    active = 0
    for start in range(0, len(boxes), chunk_size):
        b = boxes[start:start + chunk_size]
        with torch.no_grad():
            ious = box_iou(b.detach(), gt)
            ious.scatter_(1, owners[start:start + chunk_size, None], -1)
            neighbor = ious.argmax(dim=1)
        other = gt[neighbor]
        intersection = (torch.minimum(b[:, 2:], other[:, 2:]) -
                        torch.maximum(b[:, :2], other[:, :2])).clamp(min=0).prod(dim=1)
        iog = intersection / (other[:, 2:] - other[:, :2]).prod(dim=1).clamp(min=1e-6)
        total = total + smooth_ln(iog, sigma).sum()
        active += int((iog.detach() > 0).sum())
    return total, active


def select_repbox_predictions(boxes, owners, gt, limit):
    selected = []
    with torch.no_grad():
        for owner in owners.unique(sorted=True):
            indices = torch.where(owners == owner)[0]
            quality = box_iou(boxes[indices].detach(), gt[owner].unsqueeze(0)).flatten()
            selected.append(indices[torch.argsort(quality, descending=True, stable=True)[:limit]])
    return torch.cat(selected) if selected else owners.new_empty(0)


def repbox_sum(boxes, owners, gt, sigma=0.0, limit=4, chunk_size=64):
    indices = select_repbox_predictions(boxes, owners, gt, limit)
    b, ids = boxes[indices], owners[indices]
    total, count = boxes.sum() * 0, 0
    for start in range(0, len(b), chunk_size):
        for other in range(start, len(b), chunk_size):
            left, right = b[start:start + chunk_size], b[other:other + chunk_size]
            overlaps = box_iou(left, right)
            valid = ids[start:start + len(left), None] != ids[None, other:other + len(right)]
            rows = torch.arange(start, start + len(left), device=b.device)
            cols = torch.arange(other, other + len(right), device=b.device)
            valid &= rows[:, None] < cols[None, :]
            valid &= overlaps.detach() > 0
            total = total + smooth_ln(overlaps[valid], sigma).sum()
            count += int(valid.sum())
    return total, count, len(indices)


def center_size(boxes):
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2,
                      boxes[:, 2:] - boxes[:, :2]), dim=1)


class OcclusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = 1.0

    def forward(self, predictions, targets, assignments, visible_predictions=None, strides=None):
        c = self.config
        # Explicit float casts keep all geometry FP32 under AMP.
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            boxes = predictions.float()
            zero = boxes.sum() * 0
            visible_sum = zero if visible_predictions is None else zero + visible_predictions.float().sum() * 0
            repgt, repbox = zero, zero
            counts = dict(visible_valid_pairs=0, visible_skipped_pairs=0, visible_noncontained_pairs=0,
                          visible_supervised=0, repgt_positive_count=0, repgt_overlap_count=0,
                          repbox_pair_count=0, repbox_selected_count=0)
            for i, (target, assignment) in enumerate(zip(targets, assignments)):
                gt = target['boxes'].to(boxes.device).float()
                mask = assignment.positive_mask
                owners = assignment.matched_gt_indices[mask]
                positive = boxes[i, mask]
                if c.repgt_loss_weight:
                    value, active = repgt_sum(positive, owners, gt, c.repgt_sigma, c.repulsion_chunk_size)
                    repgt = repgt + value
                    counts['repgt_positive_count'] += len(positive)
                    counts['repgt_overlap_count'] += active
                if c.repbox_loss_weight:
                    value, pairs, selected = repbox_sum(positive, owners, gt, c.repbox_sigma,
                                                       c.repbox_predictions_per_gt, c.repulsion_chunk_size)
                    repbox = repbox + value
                    counts['repbox_pair_count'] += pairs
                    counts['repbox_selected_count'] += selected
                if not c.visible_loss_weight:
                    continue
                if visible_predictions is None or strides is None:
                    raise ValueError('Visible supervision requires auxiliary predictions and strides')
                visible = target.get('visible_boxes', gt.new_empty((0, 4))).to(boxes.device).float()
                mapping = target.get('visible_box_indices', owners.new_empty(0)).to(boxes.device)
                if len(mapping) != len(visible) or mapping.dtype != torch.long:
                    raise ValueError('Visible box mapping must contain one int64 index per box')
                if len(mapping) and (int(mapping.min()) < 0 or int(mapping.max()) >= len(gt)
                                     or len(mapping.unique()) != len(mapping)):
                    raise ValueError('Visible box mapping must identify distinct surviving full boxes')
                valid = torch.isfinite(visible).all(dim=1) & (visible[:, 2:] > visible[:, :2]).all(dim=1)
                counts['visible_valid_pairs'] += int(valid.sum())
                counts['visible_skipped_pairs'] += len(gt) - int(valid.sum())
                visible, mapping = visible[valid], mapping[valid]
                full = gt[mapping]
                contained = ((visible[:, :2] >= full[:, :2] - 1e-4).all(dim=1)
                             & (visible[:, 2:] <= full[:, 2:] + 1e-4).all(dim=1))
                counts['visible_noncontained_pairs'] += int((~contained).sum())
                lookup = owners.new_full((len(gt),), -1)
                lookup[mapping] = torch.arange(len(mapping), device=boxes.device)
                rows = lookup[owners]
                supervised = rows >= 0
                predicted = visible_predictions[i, mask][supervised].float()
                expected = visible[rows[supervised]]
                scales = strides.to(boxes.device).float()[mask][supervised]
                visible_sum = visible_sum + F.smooth_l1_loss(center_size(predicted) / scales,
                                                            center_size(expected) / scales,
                                                            reduction='none').mean(dim=1).sum()
                visible_sum = visible_sum + generalized_box_iou_loss(predicted, expected, reduction='sum')
                counts['visible_supervised'] += len(predicted)
            values = dict(visible_loss=visible_sum / max(counts['visible_supervised'], 1),
                          repgt_loss=repgt / max(counts['repgt_positive_count'], 1),
                          repbox_loss=repbox / max(counts['repbox_pair_count'], 1))
            total, metrics = zero, dict(counts, auxiliary_scale=self.scale)
            for name, value in values.items():
                weighted = value * getattr(c, name + '_weight') * self.scale
                total = total + weighted
                metrics[name] = value.detach().item()
                metrics[name + '_weighted'] = weighted.detach().item()
            return total, metrics
