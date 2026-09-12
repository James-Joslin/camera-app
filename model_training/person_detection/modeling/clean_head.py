"""Clean shared towers and the single-location LTRB prediction contract."""

import math

import torch
from torch import nn


DEFAULT_MODEL_VARIANT = "clean_ltrb"
MODEL_VARIANTS = ("anchor", "clean_anchor", "clean_ltrb")
XYXY_OUTPUT = "boxes_xyxy_pixels"


def box_encoding(variant):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {MODEL_VARIANTS}")
    return "xyxy_pixels" if variant == "clean_ltrb" else "anchor_offsets"


def model_format(variant):
    box_encoding(variant)
    return 3 if variant == "anchor" else 4


def checkpoint_variant(checkpoint):
    variant = checkpoint.get("config", {}).get("model_variant", "anchor")
    if checkpoint.get("modelFormatVersion") != model_format(variant):
        raise RuntimeError("Checkpoint format does not match its model variant")
    if checkpoint.get("boxEncoding", "anchor_offsets") != box_encoding(variant):
        raise RuntimeError("Checkpoint box encoding does not match its model variant")
    return variant


def mark_openvino_outputs(ov_model, variant):
    """Persist box semantics through IR serialization and quantization."""
    encoding = box_encoding(variant)
    ov_model.output(0).get_tensor().set_names({"person_quality_logits"})
    ov_model.output(1).get_tensor().set_names(
        {XYXY_OUTPUT if encoding == "xyxy_pixels" else "boxes_anchor_offsets"}
    )
    ov_model.set_rt_info(variant, ["person_detector", "variant"])
    ov_model.set_rt_info(encoding, ["person_detector", "box_encoding"])


def has_decoded_boxes(outputs):
    return any(XYXY_OUTPUT in getattr(output, "get_names", lambda: set())() for output in outputs)


class PointReferenceGenerator:
    """One point per cell; virtual boxes are used only for ATSS assignment.

    Centers use canvas/feature dimensions, matching the baseline grid on odd maps.
    The reference side is 8 nominal strides, as in single-reference ATSS.
    """

    def __init__(self, input_height, input_width, feature_map_shapes):
        self.input_height = input_height
        self.input_width = input_width
        self.feature_map_shapes = list(feature_map_shapes)
        self.strides = [8, 16, 32, 64, 128]
        if len(self.feature_map_shapes) != len(self.strides):
            raise ValueError("Expected five feature levels")
        references, points, strides = [], [], []
        for (height, width), stride in zip(self.feature_map_shapes, self.strides):
            for row in range(height):
                for column in range(width):
                    x, y = (column + 0.5) / width, (row + 0.5) / height
                    references.append([x, y, 8 * stride / input_width, 8 * stride / input_height])
                    points.append([x * input_width, y * input_height])
                    strides.append([stride])
        self.anchors = torch.tensor(references, dtype=torch.float32)
        self.points = torch.tensor(points, dtype=torch.float32)
        self.distance_scales = torch.tensor(strides, dtype=torch.float32)
        self.num_anchors_per_level = [h * w for h, w in self.feature_map_shapes]

    def get_anchors(self):
        return self.anchors

    def get_num_anchors_per_location(self, level):
        return 1

    def specification(self):
        return {
            "coordinateSystem": "normalized-xy-separate-v1",
            "inputHeight": self.input_height,
            "inputWidth": self.input_width,
            "featureMapShapes": [list(shape) for shape in self.feature_map_shapes],
            "grid": "canvas-divided-by-feature-shape-cell-centers-v1",
            "referenceBox": "square-8-times-nominal-stride-training-only",
            "strides": self.strides,
            "boxEncoding": "xyxy_pixels",
            "regression": "relu-ltrb-times-stride-v1",
        }


class SharedSeparableBlock(nn.Module):
    """Shared convolution weights with independent BatchNorm for each level."""

    def __init__(self, channels, levels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norms = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(levels)])
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, level):
        return self.activation(self.norms[level](self.pointwise(self.depthwise(x))))


class CleanDetectionHead(nn.Module):
    """Separate cls/box towers shared across levels, without attention or double norm."""

    def __init__(self, channels, anchors_per_location, ltrb=False):
        super().__init__()
        levels = len(anchors_per_location)
        self.shared = SharedSeparableBlock(channels, levels)
        self.classification = SharedSeparableBlock(channels, levels)
        self.regression = SharedSeparableBlock(channels, levels)
        self.dropout = nn.Dropout2d(0.1)
        # Anchor projections retain level-specific template semantics. LTRB shares
        # its final projection too, since every level has the same representation.
        counts = [1] if ltrb else anchors_per_location
        self.cls_outputs = nn.ModuleList([nn.Conv2d(channels, n, 3, padding=1) for n in counts])
        self.box_outputs = nn.ModuleList([nn.Conv2d(channels, 4 * n, 3, padding=1) for n in counts])
        self.ltrb = ltrb
        for layer in self.cls_outputs:
            nn.init.constant_(layer.bias, math.log(0.01 / 0.99))
        for layer in self.box_outputs:
            nn.init.constant_(layer.bias, 1.0 if ltrb else 0.0)
            if ltrb:
                nn.init.normal_(layer.weight, std=0.01)

    def forward(self, x, level):
        x = self.dropout(self.shared(x, level))
        index = 0 if self.ltrb else level
        cls = self.cls_outputs[index](self.classification(x, level))
        boxes = self.box_outputs[index](self.regression(x, level))
        batch = x.shape[0]
        cls = cls.permute(0, 2, 3, 1).reshape(batch, -1, 1)
        boxes = boxes.permute(0, 2, 3, 1).reshape(batch, -1, 4)
        if self.ltrb:
            boxes = torch.relu(boxes) + 1e-3
        return cls, boxes
