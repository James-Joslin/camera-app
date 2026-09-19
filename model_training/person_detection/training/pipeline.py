"""
Optimized SSD Person Detection Training Pipeline
================================================
Features:
- Canonical CityPersons JSON annotations with full, visible, and ignore boxes
- Azurite object storage integration
- ATSS assignment, GIoU regression, and quality focal classification
- Person-optimized anchor ratios
- Mixed precision training (AMP)
- Export to OpenVINO IR format
- Separate manifest-driven, accuracy-controlled INT8 optimization

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import box_iou, nms
import numpy as np
import json
import os
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import random

from person_detection.data.dataset import CanonicalPersonDetectionDataset
from person_detection.modeling.occlusion import (
    OcclusionConfig, OcclusionLoss, occlusion_recipe, auxiliary_scale,
)
from person_detection.modeling.assignment import ATSSAnchorAssigner
from person_detection.modeling.clean_head import (
    DEFAULT_MODEL_VARIANT, CleanDetectionHead, PointReferenceGenerator, box_encoding, model_format,
    mark_openvino_outputs,
)
from person_detection.evaluation.metrics import MAPCalculator

try:
    from person_detection.data.storage import AzuriteBlobCompat
    HAS_AZURITE = True
except ImportError:
    HAS_AZURITE = False
    print("Warning: azure-storage-blob not installed. Using local filesystem.")


try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("Warning: openvino not installed. OpenVINO export will be disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig(OcclusionConfig):
    """Training configuration with sensible defaults"""
    # Data
    data_root: str = './data'
    input_height: int = 360
    input_width: int = 640
    batch_size: int = 32   # Increased - more stable gradients
    num_workers: int = 1

    # Model
    num_classes: int = 1  # one sigmoid localization-quality logit per prediction
    model_variant: str = DEFAULT_MODEL_VARIANT
    backbone: str = "mobilenetv3_small"
    use_stride4: bool = False
    use_pan: bool = True
    regression_depth: int = 1

    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-3
    use_varied_lr: bool = False
    weight_decay: float = 1e-4
    momentum: float = 0.937

    # Loss
    neg_pos_ratio: int = 3  # Used only when focal loss is disabled
    use_focal_loss: bool = True
    focal_alpha: float = 0.4
    focal_gamma: float = 2.0
    atss_topk: int = 9
    giou_weight: float = 2.0
    use_stratified_oversampling: bool = True
    sampling_seed: int = 1337

    # Optimization
    use_amp: bool = True  # Mixed precision training
    gradient_clip: float = 10.0
    warmup_epochs: int = 3

    # Periodic validation metrics and visual logging
    validation_ap_every_n_epochs: int = 5
    validation_ap_score_threshold: float = 0.01
    validation_ap_nms_threshold: float = 0.5
    validation_ap_pre_nms_topk: int = 1000
    validation_ap_max_detections: int = 100
    validation_recall_fppi: float = 0.1
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = 'tensorboard'

    # Retain the retired-QAT guard for old configurations; INT8 belongs to optimization.
    export_openvino_after_training: bool = True
    enable_quantization: bool = False

    # Azurite (optional)
    azurite_endpoint: str = ''
    azurite_access_key: str = ''
    azurite_secret_key: str = ''
    azurite_connection_string: str = ''
    azurite_data_bucket: str = 'computer-vision-data'
    azurite_model_bucket: str = 'computer-vision-models'
    use_azurite: bool = True

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# AZURITE CLIENT
# ============================================================================

class AzuriteClient:
    """Azurite client wrapper with caching and error handling"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.client = None
        self._cache = {}

        if config.use_azurite and HAS_AZURITE:
            try:
                self.client = AzuriteBlobCompat(
                    endpoint=str(config.azurite_endpoint),
                    access_key=config.azurite_access_key,
                    secret_key=config.azurite_secret_key,
                    secure=False
                )
                # Test connection
                self.client.bucket_exists(bucket_name=str(config.azurite_data_bucket))
                print(f"✓ Connected to Azurite at {config.azurite_endpoint}")
            except Exception as e:
                print(f"✗ Azurite connection failed: {e}")
                print("  Falling back to local filesystem")
                self.client = None

    def list_objects(self, bucket: str, prefix: str) -> List[str]:
        """List objects in bucket with prefix (recursive)"""
        if self.client is None:
            # Fallback to local filesystem
            local_path = Path(self.config.data_root) / prefix
            if local_path.exists():
                # Recursively find all files and return relative paths
                files = []
                for p in local_path.rglob('*'):
                    if p.is_file():
                        # Return path relative to data_root (e.g., "images/train/aachen/file.png")
                        rel_path = str(p.relative_to(self.config.data_root))
                        files.append(rel_path)
                return files
            return []

        try:
            objects = self.client.list_objects(bucket_name=str(bucket), prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def get_object_bytes(self, bucket: str, object_name: str) -> Optional[bytes]:
        """Get object as bytes"""
        cache_key = f"{bucket}/{object_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.client is None:
            # Fallback to local filesystem
            local_path = Path(self.config.data_root) / object_name
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    data = f.read()
                return data
            return None

        try:
            response = self.client.get_object(bucket_name=str(bucket), object_name=str(object_name))
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            print(f"Error getting object {object_name}: {e}")
            return None

    def put_object(self, bucket: str, object_name: str, file_path: str):
        """Upload file to Azurite"""
        if self.client is None:
            # Fallback: save locally
            local_path = Path(self.config.data_root) / object_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(file_path, local_path)
            return

        try:
            self.client.fput_object(
                bucket_name=str(bucket),
                object_name=str(object_name),
                file_path=str(file_path)
            )
        except Exception as e:
            print(f"Error uploading {object_name}: {e}")

    def clear_cache(self):
        """Clear object cache to free memory"""
        self._cache.clear()

# ============================================================================
# MODEL SUMMARRISER
# ============================================================================

class ModelType(Enum):
    """Enumeration for different model types."""
    SINGLE_INPUT = "single_input"
    MULTI_INPUT = "multi_input"
    TRANSFORMER = "transformer"
    MULTI_INPUT_TRANSFORMER = "multi_transformer"

def model_summary(model: nn.Module, model_type: ModelType, input1_size: tuple, input2_size: tuple = (0, 0)) -> bool:
    """
    Function to print a summary of a PyTorch model.

    Args:
        model (nn.Module): The model to summarize.
        model_type (ModelType): The type of the model.
        input1_size (tuple): The size of the first input.
        input2_size (tuple, optional): The size of the second input. Defaults to (0,0).

    Returns:
        bool: True if the summary was printed successfully, False otherwise.

    Raises:
        ValueError: If input sizes are not tuples or if model_type is not a valid ModelType.
        RuntimeError: If the model output is not available.
    """

    if not isinstance(input1_size, tuple) or not isinstance(input2_size, tuple):
        raise ValueError("Input sizes must be tuples.")

    if not isinstance(model_type, ModelType):
        raise ValueError("model_type must be an instance of ModelType.")

    total_params = 0
    output = None  # Initialize output to None
    hooks = []
    completed = False

    def print_title_block(model_type: ModelType, size_string: str = ""):
        """
        Function to print the title block of the summary.

        Args:
            model_type (ModelType): The type of the model.
            size_string (str, optional): The string representation of the input size. Defaults to "".
        """
        print("\n")
        print("----------------------------------------------------------------")
        print(f"Model Type:                {str(model_type.value.replace('_', ' ').title()).ljust(25)}")
        print("----------------------------------------------------------------")
        print(f"Input Shape:               {size_string.ljust(25)}")
        print("----------------------------------------------------------------")
        print("Layer (type)               Output Shape         Param #")
        print("================================================================")
        print("----------------------------------------------------------------")

    def print_footer_block():
        """
        Function to print the footer block of the summary.
        """
        # Calculate and print the model size in KB, MB, or GB
        total_size_bytes = total_params  # Total number of bytes
        total_size_kb = total_size_bytes / 1024
        total_size_mb = total_size_kb / 1024
        total_size_gb = total_size_mb / 1024

        if total_size_gb >= 1:
            total_size_string = f"{total_size_gb:.2f} GB"
        elif total_size_mb >= 1:
            total_size_string = f"{total_size_mb:.2f} MB"
        else:
            total_size_string = f"{total_size_kb:.2f} KB"

        print("----------------------------------------------------------------")
        print(f"Total Parameters: {total_params:,}")
        print(f"Output Shape: {output_shape_string.ljust(25)}")
        print(f'Model on: {next(model.parameters()).device}')
        print(f'Approximate Model Size: {total_size_string}')
        print("----------------------------------------------------------------")

    def register_hook(module):
        """
        Function to register a hook for a module.

        Args:
            module (nn.Module): The module to register the hook for.
        """
        def hook(module, input, output):
            nonlocal total_params
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += num_params

            if isinstance(output, tuple):
                output_shape = [str(list(o.shape)) if torch.is_tensor(o) else str(type(o)) for o in output]
                output_shape = output_shape[0]  # Assuming the first output shape if multiple
            else:
                output_shape = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))

            if len(list(module.named_children())) == 0 or isinstance(module, nn.MultiheadAttention):  # Only print leaf nodes or attention layers
                print(f"{module.__class__.__name__.ljust(25)}  {output_shape.ljust(25)} {f'{num_params:,}'}")

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    DEVICE = next(model.parameters()).device

    try:
        match model_type:
            case ModelType.SINGLE_INPUT:
                print_title_block(model_type, size_string=f"{str(input1_size)}")
                input_tensor = torch.ones(*input1_size, requires_grad=False).to(DEVICE).float()
                output = model(input_tensor)

            case ModelType.MULTI_INPUT:
                print_title_block(model_type, size_string=f"{input1_size, input2_size}")
                input_tensor1 = torch.ones(*input1_size, requires_grad=False).to(DEVICE).float()
                input_tensor2 = torch.ones(*input2_size, requires_grad=False).to(DEVICE).float()
                output = model(input_tensor1, input_tensor2)

            case ModelType.TRANSFORMER:
                print_title_block(model_type, size_string=f"{str(input1_size)}")
                input_tensor = torch.ones(*input1_size, requires_grad=False).to(DEVICE).float()
                output = model(input_tensor, input_tensor)

            case ModelType.MULTI_INPUT_TRANSFORMER:
                print_title_block(model_type, size_string=f"{input1_size, input2_size}")
                input_tensor1 = torch.ones(*input1_size, requires_grad=False).to(DEVICE).long()
                input_tensor2 = torch.ones(*input2_size, requires_grad=False).to(DEVICE).float()
                output = model(input_tensor1, input_tensor2)  # assumes categories then continuous as input order

            case _:
                raise ValueError("Model type unidentified")
    except Exception as e:
        for h in hooks:
            h.remove()
        raise RuntimeError(f"Error during model forward pass: {e}")

    for h in hooks:
        h.remove()

    if output is not None:
        output_shape_string = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))
        if isinstance(output, tuple):
            output_shape_string += " ("
            for item in output:
                if torch.is_tensor(item):
                    output_shape_string += f"{str(tuple(item.shape))}, "
            output_shape_string = output_shape_string[:-2]
            output_shape_string += ")"
    else:
        raise RuntimeError("Model output is not available")

    print_footer_block()
    completed = True

    return completed

# ============================================================================
# DATASET
# ============================================================================

class PersonAnchorGenerator:
    """Generate normalized anchors from real rectangular feature-map shapes."""

    def __init__(
        self,
        input_height: int = 360,
        input_width: int = 640,
        feature_map_shapes: Optional[List[Tuple[int, int]]] = None,
    ):
        if input_width <= input_height:
            raise ValueError("Person detector input must be landscape: width > height")
        self.input_height = input_height
        self.input_width = input_width
        self.strides = [8, 16, 32, 64, 128]
        self.feature_map_shapes = feature_map_shapes or [
            (
                (input_height + stride - 1) // stride,
                (input_width + stride - 1) // stride,
            )
            for stride in self.strides
        ]
        if len(self.feature_map_shapes) != len(self.strides):
            raise ValueError("Expected one feature-map shape for each pyramid level")

        self.scales = [
            [0.02, 0.04],
            [0.06, 0.10],
            [0.16, 0.24],
            [0.32, 0.48, 0.56],
            [0.64, 0.80, 0.95],
        ]
        # Pixel-space width / height ratios. Normalize x by width and y by height.
        self.aspect_ratios = [
            [0.15, 0.25, 0.40],
            [0.15, 0.25, 0.40],
            [0.20, 0.33, 0.50],
            [0.25, 0.50, 1.00],
            [0.25, 0.50, 1.00],
        ]
        self.anchors = self._generate_anchors()
        self.num_anchors_per_level = self._count_anchors_per_level()
        print(
            f"Generated {len(self.anchors)} anchors across "
            f"{len(self.feature_map_shapes)} rectangular levels"
        )

    def _generate_anchors(self) -> torch.Tensor:
        all_anchors = []
        reference_pixels = np.sqrt(self.input_height * self.input_width)
        for level_idx, (feature_height, feature_width) in enumerate(self.feature_map_shapes):
            scales = self.scales[level_idx]
            ratios = self.aspect_ratios[level_idx]
            for row in range(feature_height):
                for column in range(feature_width):
                    center_x = (column + 0.5) / feature_width
                    center_y = (row + 0.5) / feature_height
                    for scale in scales:
                        base_pixels = scale * reference_pixels
                        for ratio in ratios:
                            width_pixels = base_pixels * np.sqrt(ratio)
                            height_pixels = base_pixels / np.sqrt(ratio)
                            all_anchors.append([
                                center_x,
                                center_y,
                                width_pixels / self.input_width,
                                height_pixels / self.input_height,
                            ])
        return torch.tensor(all_anchors, dtype=torch.float32)

    def _count_anchors_per_level(self) -> List[int]:
        return [
            feature_height * feature_width * len(self.scales[level]) * len(self.aspect_ratios[level])
            for level, (feature_height, feature_width) in enumerate(self.feature_map_shapes)
        ]

    def get_anchors(self) -> torch.Tensor:
        return self.anchors

    def get_num_anchors_per_location(self, level: int) -> int:
        return len(self.scales[level]) * len(self.aspect_ratios[level])

    def specification(self) -> Dict:
        return {
            "coordinateSystem": "normalized-xy-separate-v1",
            "sizeBasis": "sqrt-canvas-area-pixels-v1",
            "featureMapShapes": [list(shape) for shape in self.feature_map_shapes],
            "scales": self.scales,
            "aspectRatios": self.aspect_ratios,
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficiency"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, stride=stride, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256, separable=False, use_pan=False):
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 convs to unify channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions (reduce aliasing after upsampling)
        self.output_convs = nn.ModuleList([
            SeparableConv2d(out_channels, out_channels) if separable else nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
        # Independent modules provide independent BatchNorm per transition.
        self.pan_downsamples = nn.ModuleList([
            SeparableConv2d(out_channels, out_channels, stride=2)
            for _ in range(len(in_channels_list) - 1)
        ]) if use_pan else nn.ModuleList()
        self.pan_fusions = nn.ModuleList([
            SeparableConv2d(out_channels, out_channels)
            for _ in range(len(in_channels_list) - 1)
        ]) if use_pan else nn.ModuleList()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: [C3, C4, C5] from low to high level (small to large stride)

        # Start from the highest level (smallest spatial size)
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = nn.functional.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        if self.pan_downsamples:
            bottom_up = [outputs[0]]
            for index, (downsample, fusion) in enumerate(zip(self.pan_downsamples, self.pan_fusions)):
                down = downsample(bottom_up[-1])
                # Stride-2 padding gives ceil(H/2), including odd feature maps.
                bottom_up.append(fusion(outputs[index + 1] + down))
            outputs = bottom_up
        return outputs

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Spatial attention module"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention

class AttentionDetectionHead(nn.Module):
    """Detection head with one sigmoid quality-aware person logit per anchor."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        if num_classes != 1:
            raise ValueError("The quality-aware binary head requires num_classes=1")
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Shared features with attention
        self.shared_conv = SeparableConv2d(in_channels, in_channels)
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention()

        # Deeper subnets (helps with pose variation)
        self.cls_subnet = nn.Sequential(
            SeparableConv2d(in_channels, in_channels),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            # SeparableConv2d(in_channels, in_channels),
            # nn.GroupNorm(32, in_channels),
            # nn.SiLU(inplace=True),
        )
        self.bbox_subnet = nn.Sequential(
            SeparableConv2d(in_channels, in_channels),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            # SeparableConv2d(in_channels, in_channels),
            # nn.GroupNorm(32, in_channels),
            # nn.SiLU(inplace=True),
        )

        self.cls_conv = nn.Conv2d(in_channels, num_anchors, 3, padding=1)
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.dropout = nn.Dropout2d(0.1)

        self._init_weights()

    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_conv.bias, bias_value)
        nn.init.zeros_(self.bbox_conv.bias)

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.dropout(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)

        cls_feat = self.cls_subnet(x)
        bbox_feat = self.bbox_subnet(x)

        batch_size = x.size(0)

        cls = self.cls_conv(cls_feat)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)

        bbox = self.bbox_conv(bbox_feat)
        bbox = bbox.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        return cls, bbox

class SSDPersonDetector(nn.Module):
    """Enhanced SSD with an FPN on a fixed landscape input canvas."""

    def __init__(
        self,
        num_classes: int = 1,
        input_height: int = 360,
        input_width: int = 640,
        pretrained: bool = True,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        visible_auxiliary: bool = False,
        backbone: str = "mobilenetv3_small",
        use_stride4: bool = False,
        use_pan: bool = False,
        regression_depth: int = 1,
    ):
        super().__init__()
        if input_width <= input_height:
            raise ValueError("Person detector input must be landscape: width > height")
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        if num_classes != 1:
            raise ValueError("Person detector requires one quality-aware person logit")
        self.model_variant = model_variant
        self.box_encoding = box_encoding(model_variant)
        if visible_auxiliary and model_variant != "clean_ltrb":
            raise ValueError("Visible auxiliary requires clean_ltrb")
        self.auxiliary_head = nn.Conv2d(128, 4, 3, padding=1) if visible_auxiliary else None

        if backbone not in ("mobilenetv3_small", "mobilenetv4_conv_small"):
            raise ValueError(f"Unknown backbone: {backbone!r}")
        self.use_stride4 = use_stride4 and model_variant == "clean_ltrb"
        if regression_depth not in (1, 2):
            raise ValueError("regression_depth must be 1 or 2")
        self.use_pan = use_pan
        self.regression_depth = regression_depth
        if model_variant == "anchor" and regression_depth != 1:
            raise ValueError("The second regression block requires a clean head")
        self.backbone_name = backbone
        if backbone == "mobilenetv3_small":
            network = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = network.features
            self.feature_indices = [1, 3, 6, 12] if self.use_stride4 else [3, 6, 12]
            channels = [16, 24, 40, 576] if self.use_stride4 else [24, 40, 576]
        else:
            import timm
            self.features = timm.create_model(
                "mobilenetv4_conv_small.e2400_r224_in1k", pretrained=pretrained,
                features_only=True, out_indices=(1, 2, 3, 4) if self.use_stride4 else (2, 3, 4),
            )
            channels = self.features.feature_info.channels()
            expected_strides = [4, 8, 16, 32] if self.use_stride4 else [8, 16, 32]
            if self.features.feature_info.reduction() != expected_strides:
                raise ValueError(f"Backbone must provide strides {expected_strides}")
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[-1], 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            ),
        ])

        backbone_channels = channels + [256, 128]
        fpn_channels = 128
        self.fpn = FPN(backbone_channels, out_channels=fpn_channels,
                       separable=model_variant != "anchor", use_pan=use_pan)
        feature_map_shapes = self._infer_feature_map_shapes()
        generator_type = PointReferenceGenerator if model_variant == "clean_ltrb" else PersonAnchorGenerator
        self.anchor_generator = generator_type(
            input_height, input_width, feature_map_shapes,
            **({"strides": [4, 8, 16, 32, 64, 128]} if self.use_stride4 else {}),
        )
        if model_variant != "anchor":
            self.detection_heads = CleanDetectionHead(
                fpn_channels,
                [self.anchor_generator.get_num_anchors_per_location(level) for level in range(len(feature_map_shapes))],
                ltrb=model_variant == "clean_ltrb",
                regression_depth=regression_depth,
            )
            if model_variant == "clean_ltrb":
                self.register_buffer("point_centers", self.anchor_generator.points, persistent=False)
                self.register_buffer("distance_scales", self.anchor_generator.distance_scales, persistent=False)
        else:
            self.detection_heads = nn.ModuleList([
                AttentionDetectionHead(
                    fpn_channels,
                    self.anchor_generator.get_num_anchors_per_location(level),
                    num_classes,
                )
                for level in range(len(feature_map_shapes))
            ])

    def _extract_backbone_features(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        if self.backbone_name == "mobilenetv4_conv_small":
            features = list(self.features(tensor))
            tensor = features[-1]
        else:
            features = []
            for index, layer in enumerate(self.features):
                tensor = layer(tensor)
                if index in self.feature_indices:
                    features.append(tensor)
        for extra_layer in self.extra_layers:
            tensor = extra_layer(tensor)
            features.append(tensor)
        return features

    def _infer_feature_map_shapes(self) -> List[Tuple[int, int]]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            features = self._extract_backbone_features(
                torch.zeros(1, 3, self.input_height, self.input_width)
            )
        self.train(was_training)
        return [(int(feature.shape[-2]), int(feature.shape[-1])) for feature in features]

    def forward(self, tensor: torch.Tensor):
        return self._forward(tensor, auxiliary=False)

    def forward_auxiliary(self, tensor: torch.Tensor):
        if self.auxiliary_head is None:
            raise ValueError("No visibility auxiliary configured")
        return self._forward(tensor, auxiliary=True)

    def _forward(self, tensor: torch.Tensor, auxiliary=False):
        input_shape = tuple(tensor.shape[-2:])
        if not torch.jit.is_tracing() and input_shape != (self.input_height, self.input_width):
            raise ValueError(
                f"Expected {self.input_height}x{self.input_width} input, got "
                f"{input_shape[0]}x{input_shape[1]}"
            )
        fpn_features = self.fpn(self._extract_backbone_features(tensor))
        all_cls, all_bbox, all_visible = [], [], []
        for level, feature in enumerate(fpn_features):
            if self.model_variant == "anchor":
                classification, boxes = self.detection_heads[level](feature)
            elif auxiliary:
                classification, boxes, visible = self.detection_heads(feature, level, self.auxiliary_head)
                all_visible.append(visible)
            else:
                classification, boxes = self.detection_heads(feature, level)
            all_cls.append(classification)
            all_bbox.append(boxes)
        boxes = torch.cat(all_bbox, dim=1)
        if self.model_variant == "clean_ltrb":
            distances = boxes * self.distance_scales
            boxes = torch.cat((self.point_centers - distances[..., :2],
                               self.point_centers + distances[..., 2:]), dim=-1)
        if auxiliary:
            raw = torch.cat(all_visible, dim=1).float()
            center = self.point_centers + raw[..., :2] * self.distance_scales
            size = (nn.functional.softplus(raw[..., 2:]) + 1e-3) * self.distance_scales
            visible = torch.cat((center - size / 2, center + size / 2), dim=-1)
            return torch.cat(all_cls, dim=1), boxes, visible
        return torch.cat(all_cls, dim=1), boxes


MODEL_FORMAT_VERSION = 4


def person_scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode current binary-quality logits and legacy two-class logits."""
    if logits.size(-1) == 1:
        return logits[..., 0].sigmoid()
    if logits.size(-1) == 2:
        return logits.softmax(dim=-1)[..., 1]
    raise ValueError(f"Unsupported classification output shape: {tuple(logits.shape)}")


def load_detector_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], *, strict: bool = True
):
    """Load v2 weights, migrating a legacy background/person softmax head if needed."""
    target_state = model.state_dict()
    migrated = dict(state_dict)
    converted = []
    for key, target in target_state.items():
        source = migrated.get(key)
        if source is None or source.shape == target.shape or ".cls_conv." not in key:
            continue
        if source.shape[0] != target.shape[0] * 2:
            continue
        if key.endswith(".weight"):
            paired = source.reshape(target.shape[0], 2, *source.shape[1:])
            migrated[key] = paired[:, 1] - paired[:, 0]
        elif key.endswith(".bias"):
            paired = source.reshape(target.shape[0], 2)
            migrated[key] = paired[:, 1] - paired[:, 0]
        else:
            continue
        converted.append(key)
    incompatible = model.load_state_dict(migrated, strict=strict)
    if converted:
        warnings.warn(
            "Migrated legacy two-class logits to one binary logit via person-background "
            f"difference ({len(converted)} tensors). Fine-tuning is recommended.",
            stacklevel=2,
        )
    return incompatible


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class QualityFocalLoss(nn.Module):
    """Binary Quality Focal Loss with continuous IoU targets for positives."""

    def __init__(self, alpha: float = 0.4, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = logits.sigmoid()
        modulation = (targets - probabilities).abs().pow(self.beta)
        balance = torch.where(targets > 0, self.alpha, 1 - self.alpha)
        return balance * modulation * nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )


class SSDLoss(nn.Module):
    """ATSS assignment, quality-aware binary classification, and GIoU regression."""

    def __init__(
        self,
        num_classes: int = 1,
        neg_pos_ratio: int = 3,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.4,
        focal_gamma: float = 2.0,
        input_height: int = 360,
        input_width: int = 640,
        anchors_per_level: Optional[List[int]] = None,
        atss_topk: int = 9,
        giou_weight: float = 2.0,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        occlusion_config: Optional[OcclusionConfig] = None,
        distance_scales: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("SSDLoss requires the one-logit binary detection head")
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.use_focal_loss = use_focal_loss
        self.input_height = input_height
        self.input_width = input_width
        self.anchors_per_level = list(anchors_per_level or [])
        self.box_encoding = box_encoding(model_variant)
        self.assigner = ATSSAnchorAssigner(top_k=atss_topk,
                                          point_regression=model_variant == "clean_ltrb")
        self.giou_weight = giou_weight
        self.occlusion = OcclusionLoss(occlusion_config or OcclusionConfig())
        self.distance_scales = distance_scales
        self.cls_loss_fn = QualityFocalLoss(focal_alpha, focal_gamma)

    def anchor_boxes(self, anchors: torch.Tensor) -> torch.Tensor:
        boxes = torch.zeros_like(anchors)
        boxes[:, 0] = (anchors[:, 0] - anchors[:, 2] / 2) * self.input_width
        boxes[:, 1] = (anchors[:, 1] - anchors[:, 3] / 2) * self.input_height
        boxes[:, 2] = (anchors[:, 0] + anchors[:, 2] / 2) * self.input_width
        boxes[:, 3] = (anchors[:, 1] + anchors[:, 3] / 2) * self.input_height
        return boxes

    def decode_boxes(self, offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        if self.box_encoding == "xyxy_pixels":
            return offsets
        centers_x = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
        centers_y = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
        widths = offsets[:, 2].clamp(max=10).exp() * anchors[:, 2]
        heights = offsets[:, 3].clamp(max=10).exp() * anchors[:, 3]
        return torch.stack([
            (centers_x - widths / 2) * self.input_width,
            (centers_y - heights / 2) * self.input_height,
            (centers_x + widths / 2) * self.input_width,
            (centers_y + heights / 2) * self.input_height,
        ], dim=1)

    @staticmethod
    def anchors_overlapping_ignore(
        anchor_boxes: torch.Tensor, ignore_boxes: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        return ATSSAnchorAssigner.anchors_overlapping_ignore(
            anchor_boxes, ignore_boxes, threshold
        )

    def match_anchors(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        anchors: torch.Tensor,
        ignore_boxes: Optional[torch.Tensor] = None,
        **_: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delegate assignment to the configured strategy."""
        anchor_boxes = self.anchor_boxes(anchors)
        ignore_boxes = (
            ignore_boxes
            if ignore_boxes is not None
            else anchor_boxes.new_zeros((0, 4))
        )
        result = self.assigner.assign(
            anchor_boxes,
            self.anchors_per_level,
            gt_boxes,
            gt_labels,
            ignore_boxes,
        )
        return result.matched_boxes, result.matched_labels, result.positive_mask

    def hard_negative_mining(
        self,
        cls_loss: torch.Tensor,
        labels: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        num_pos = pos_mask.sum().item()
        num_neg = int(self.neg_pos_ratio * num_pos)

        if num_neg == 0:
            num_neg = 100

        neg_mask = labels == 0

        neg_loss = cls_loss.clone()
        neg_loss[~neg_mask] = -float('inf')

        _, neg_indices = neg_loss.sort(descending=True)
        hard_neg_mask = torch.zeros_like(neg_mask)
        hard_neg_mask[neg_indices[:num_neg]] = True
        hard_neg_mask = hard_neg_mask & neg_mask

        return pos_mask | hard_neg_mask

    @staticmethod
    def aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
        area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
        area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
        return intersection / (area1 + area2 - intersection).clamp(min=1e-6)

    @classmethod
    def aligned_giou_loss(cls, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        intersection_top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        intersection_bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        intersection = (
            intersection_bottom_right - intersection_top_left
        ).clamp(min=0).prod(dim=1)
        enclosing_top_left = torch.minimum(boxes1[:, :2], boxes2[:, :2])
        enclosing_bottom_right = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
        enclosing_area = (
            enclosing_bottom_right - enclosing_top_left
        ).clamp(min=0).prod(dim=1).clamp(min=1e-6)
        area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
        area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
        union = (area1 + area2 - intersection).clamp(min=1e-6)
        iou = intersection / union
        giou = iou - (enclosing_area - union) / enclosing_area
        return 1 - giou

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        anchors: torch.Tensor,
        visible_predictions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if pred_cls.ndim != 3 or pred_cls.size(-1) != 1:
            raise ValueError("Quality-aware classification output must have shape [B, N, 1]")
        device = pred_cls.device
        anchors = anchors.to(device)
        total_cls_loss = pred_cls.new_zeros(())
        total_loc_loss = pred_cls.new_zeros(())
        total_quality = pred_cls.new_zeros(())
        total_pos = 0
        unmatched_gt = 0
        assignments = []

        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            ignore_boxes = target.get("ignore_regions", gt_boxes.new_zeros((0, 4))).to(device)
            assignment = self.assigner.assign(
                self.anchor_boxes(anchors), self.anchors_per_level, gt_boxes, gt_labels, ignore_boxes
            )
            assignments.append(assignment)
            matched_boxes, matched_labels, positive_mask = (
                assignment.matched_boxes, assignment.matched_labels, assignment.positive_mask
            )
            unmatched_gt += self.assigner.unmatched_ground_truths
            decoded = self.decode_boxes(pred_boxes[batch_index], anchors)
            quality_targets = pred_cls.new_zeros(anchors.size(0))
            if positive_mask.any():
                positive_quality = self.aligned_iou(
                    decoded[positive_mask].detach(), matched_boxes[positive_mask]
                ).clamp(0, 1)
                positive_quality = positive_quality.to(dtype=quality_targets.dtype)
                quality_targets[positive_mask] = positive_quality
                total_quality += positive_quality.float().sum()
                total_loc_loss += self.aligned_giou_loss(
                    decoded[positive_mask], matched_boxes[positive_mask]
                ).sum() * self.giou_weight

            valid_mask = matched_labels >= 0
            logits = pred_cls[batch_index, :, 0]
            losses = self.cls_loss_fn(logits, quality_targets)
            if self.use_focal_loss:
                total_cls_loss += losses[valid_mask].sum()
            else:
                selected = self.hard_negative_mining(
                    losses, matched_labels, positive_mask
                )
                total_cls_loss += losses[selected & valid_mask].sum()
            total_pos += int(positive_mask.sum())

        normalizer = max(total_pos, 1)
        cls_loss = total_cls_loss / normalizer
        loc_loss = total_loc_loss / normalizer
        detection_loss = cls_loss + loc_loss
        auxiliary_loss, auxiliary_metrics = self.occlusion(
            pred_boxes, targets, assignments, visible_predictions, self.distance_scales
        )
        total_loss = detection_loss + auxiliary_loss
        return total_loss, {
            **auxiliary_metrics,
            "detection_loss": detection_loss.item(),
            "cls_loss": cls_loss.item(),
            "loc_loss": loc_loss.item(),
            "mean_quality_target": (total_quality / normalizer).item(),
            "num_pos": total_pos,
            "unmatched_gt": unmatched_gt,
        }

# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def decode_boxes(
    pred_offsets: torch.Tensor,
    anchors: torch.Tensor,
    input_height: int,
    input_width: int,
    encoding: str = "anchor_offsets",
) -> torch.Tensor:
    """
    Decode predicted offsets to absolute box coordinates.

    Args:
        pred_offsets: [N, 4] tensor of (dx, dy, dw, dh)
        anchors: [N, 4] tensor of (cx, cy, w, h) in normalized coords
        input_height/input_width: Canvas dimensions for absolute coordinates

    Returns:
        boxes: [N, 4] tensor of (x1, y1, x2, y2) in absolute pixels
    """
    if encoding == "xyxy_pixels":
        return pred_offsets.clone()
    if encoding != "anchor_offsets":
        raise ValueError(f"Unsupported box encoding: {encoding}")
    pred_cx = pred_offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    pred_cy = pred_offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    pred_w = torch.exp(pred_offsets[:, 2].clamp(max=10)) * anchors[:, 2]
    pred_h = torch.exp(pred_offsets[:, 3].clamp(max=10)) * anchors[:, 3]

    x1 = (pred_cx - pred_w / 2) * input_width
    y1 = (pred_cy - pred_h / 2) * input_height
    x2 = (pred_cx + pred_w / 2) * input_width
    y2 = (pred_cy + pred_h / 2) * input_height

    return torch.stack([x1, y1, x2, y2], dim=1)

class OpenVINOExporter:
    """Standalone FP32 export; release optimization owns evaluation and benchmarking."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        if not HAS_OPENVINO:
            print("✗ OpenVINO not available. Export disabled.")
            return

        self.core = ov.Core()
        print("✓ OpenVINO initialized")

    def export_to_openvino(
        self,
        model: nn.Module,
        output_path: str,
        input_height: int = 360,
        input_width: int = 640,
        compress_to_fp16: bool = False
    ) -> Optional[str]:
        """Export PyTorch model to OpenVINO IR format."""
        if not HAS_OPENVINO:
            return None

        print(f"\nExporting model to OpenVINO IR: {output_path}")

        model = model.cpu()
        model.eval()

        dummy_input = torch.randn(1, 3, input_height, input_width)

        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            ov_model = ov.convert_model(
                model,
                example_input=dummy_input,
                input=[1, 3, input_height, input_width]
            )

            mark_openvino_outputs(ov_model, model.model_variant)
            ov.save_model(ov_model, output_path, compress_to_fp16=compress_to_fp16)

            print(f"✓ Model exported to {output_path}")
            bin_path = output_path.replace('.xml', '.bin')
            print(f"  Weights saved to {bin_path}")

            return output_path

        except Exception as e:
            raise RuntimeError(f"OpenVINO export failed: {output_path}") from e

def initialize_loader_worker(worker_id: int) -> None:
    """Avoid one retained file descriptor per tensor storage in worker batches.

    Targets contain several independent tensors per image. The default
    file_descriptor transport can exhaust a worker's descriptor limit, causing
    the parent's resource_sharer receive to fail with EOFError. Set this in each
    worker so it also applies when multiprocessing uses spawn.
    """
    del worker_id
    torch.multiprocessing.set_sharing_strategy("file_system")


def collate_fn(batch):
    """Custom collate for variable number of boxes"""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WarmupScheduler:
    """Learning rate warmup scheduler"""

    def __init__(self, optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, epoch: int):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False

def training_forward_loss(model, criterion, images, targets, anchors):
    if criterion.occlusion.config.visible_loss_weight:
        cls, boxes, visible = model.forward_auxiliary(images)
    else:
        cls, boxes = model(images)
        visible = None
    loss, metrics = criterion(cls, boxes, targets, anchors, visible_predictions=visible)
    return cls, boxes, loss, metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SSDLoss,
    anchors: torch.Tensor,
    device: str,
    epoch: int,
    config: TrainingConfig,
    scaler: Optional[GradScaler] = None
) -> Dict:
    """Train for one epoch with optional mixed precision"""
    model.train()
    criterion.occlusion.scale = auxiliary_scale(epoch, config.auxiliary_ramp_epochs)
    component_meters = {}

    loss_meter = AverageMeter('Loss')
    cls_meter = AverageMeter('Cls')
    loc_meter = AverageMeter('Loc')
    unmatched_gt = 0

    start_time = time.time()

    # Wrap dataloader with tqdm - creates new progress bar each epoch
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch+1}", ncols=100)

    for batch_idx, (images, targets) in pbar:
        images = images.to(device)

        if config.use_amp and scaler is not None:
            with autocast(device_type=config.device):
                pred_cls, pred_boxes, loss, loss_dict = training_forward_loss(model, criterion, images, targets, anchors)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_cls, pred_boxes, loss, loss_dict = training_forward_loss(model, criterion, images, targets, anchors)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        for name, value in loss_dict.items():
            component_meters.setdefault(name, AverageMeter(name)).update(value)
        unmatched_gt += loss_dict.get("unmatched_gt", 0)
        loss_meter.update(loss.item())
        cls_meter.update(loss_dict['cls_loss'])
        loc_meter.update(loss_dict['loc_loss'])

        # Update progress bar with loss information
        pbar.set_postfix({
            'Loss': f'{loss_meter.avg:.4f}',
            'Cls': f'{cls_meter.avg:.4f}',
            'Loc': f'{loc_meter.avg:.4f}'
        })

    pbar.close()  # Ensure progress bar is closed properly

    if unmatched_gt:
        print(f"Unassigned ground-truth instances this epoch: {unmatched_gt}")
    elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1} completed in {elapsed:.1f}s | "
          f"Avg Loss: {loss_meter.avg:.4f} | "
          f"Cls: {cls_meter.avg:.4f} | "
          f"Loc: {loc_meter.avg:.4f}\n")

    return {
        **{name: (meter.sum if name.endswith(("_count", "_pairs", "_supervised")) else meter.avg)
           for name, meter in component_meters.items()},
        'loss': loss_meter.avg,
        'cls_loss': cls_meter.avg,
        'loc_loss': loc_meter.avg,
        'duration_seconds': elapsed,
        'unmatched_gt': unmatched_gt,
    }

def add_validation_map_batch(
    calculator: MAPCalculator,
    pred_cls: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: List[Dict],
    anchors: torch.Tensor,
    config: TrainingConfig,
) -> None:
    """Accumulate one validation batch using the release evaluator's semantics."""
    anchors = anchors.to(pred_boxes.device)
    for batch_index, target in enumerate(targets):
        scores = person_scores_from_logits(pred_cls[batch_index])
        selected = torch.where(scores >= config.validation_ap_score_threshold)[0]
        if len(selected) > config.validation_ap_pre_nms_topk:
            selected = selected[
                scores[selected].topk(config.validation_ap_pre_nms_topk).indices
            ]

        detections = []
        if len(selected) > 0:
            selected_scores = scores[selected]
            boxes = decode_boxes(
                pred_boxes[batch_index, selected],
                anchors[selected],
                config.input_height,
                config.input_width,
                encoding=box_encoding(config.model_variant),
            )
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, config.input_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, config.input_height)
            keep = nms(boxes, selected_scores, config.validation_ap_nms_threshold)
            keep = keep[:config.validation_ap_max_detections]
            for box, score in zip(boxes[keep], selected_scores[keep]):
                detections.append({
                    "class": 1,
                    "score": float(score.detach().cpu()),
                    "box": box.detach().cpu().tolist(),
                })

        image_id = int(target["image_id"])
        calculator.add_predictions(image_id, detections)
        calculator.add_ground_truths(
            image_id,
            target["boxes"].detach().cpu().tolist(),
            target["labels"].detach().cpu().tolist(),
        )
        ignore_regions = target.get("ignore_regions")
        if ignore_regions is not None:
            calculator.add_ignore_regions(
                image_id, ignore_regions.detach().cpu().tolist()
            )


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SSDLoss,
    anchors: torch.Tensor,
    device: str,
    *,
    compute_ap: bool = False,
    config: Optional[TrainingConfig] = None,
) -> Dict:
    """Validate loss and optionally calculate dataset-level AP in the same pass."""
    if compute_ap and config is None:
        raise ValueError("TrainingConfig is required when validation AP is enabled")
    model.eval()
    loss_meter = AverageMeter('Loss')
    component_meters = {}
    calculator = (
        MAPCalculator(
            [0.5 + index * 0.05 for index in range(10)],
            recall_fppi=config.validation_recall_fppi,
        )
        if compute_ap else None
    )

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            pred_cls, pred_boxes, loss, loss_dict = training_forward_loss(model, criterion, images, targets, anchors)
            for name, value in loss_dict.items():
                component_meters.setdefault(name, AverageMeter(name)).update(value)
            loss_meter.update(loss.item())
            if calculator is not None:
                add_validation_map_batch(
                    calculator, pred_cls, pred_boxes, targets, anchors, config
                )

    results = {name: (meter.sum if name.endswith(("_count", "_pairs", "_supervised")) else meter.avg)
           for name, meter in component_meters.items()}
    results['loss'] = loss_meter.avg
    if calculator is not None:
        results.update(calculator.compute_map(verbose=False))
        results.update({
            f"metric_{key}": value
            for key, value in calculator.get_summary().items()
        })
    return results

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def detector_state_dict(model):
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith("auxiliary_head.")}


def load_training_state(model, checkpoint):
    state = dict(checkpoint["model_state_dict"])
    if model.auxiliary_head is not None:
        state.update({"auxiliary_head." + key: value
                      for key, value in checkpoint["auxiliary_state_dict"].items()})
    load_detector_state_dict(model, state)


def checkpoint_is_selected(selection, *, loss_improved, ap_improved, final_epoch):
    return {"ap": ap_improved, "detection_loss": loss_improved, "final": final_epoch}[selection]


def checkpoint_resume_mismatches(
    checkpoint: Dict,
    config: TrainingConfig,
    dataset_metadata: Dict,
    anchor_specification: Dict,
) -> Dict[str, Dict]:
    """Explain why a checkpoint cannot safely resume this exact training contract."""
    mismatches: Dict[str, Dict] = {}

    def compare(name: str, checkpoint_value, current_value) -> None:
        if checkpoint_value != current_value:
            mismatches[name] = {
                "checkpoint": checkpoint_value,
                "current": current_value,
            }

    compare("modelFormatVersion", checkpoint.get("modelFormatVersion"), model_format(config.model_variant))
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        mismatches["config"] = {"checkpoint": None, "current": "mapping"}
    else:
        compare("inputHeight", checkpoint_config.get("input_height"), config.input_height)
        compare("inputWidth", checkpoint_config.get("input_width"), config.input_width)
        compare("numClasses", checkpoint_config.get("num_classes"), config.num_classes)
        compare("modelVariant", checkpoint_config.get("model_variant", "anchor"), config.model_variant)
        compare("backbone", checkpoint_config.get("backbone", "mobilenetv3_small"), config.backbone)
        compare("use_pan", checkpoint_config.get("use_pan", False), config.use_pan)
        compare("regression_depth", checkpoint_config.get("regression_depth", 1), config.regression_depth)
        compare("use_stride4", checkpoint_config.get("use_stride4", False) and checkpoint_config.get("model_variant") == "clean_ltrb",
                config.use_stride4 and config.model_variant == "clean_ltrb")

    checkpoint_dataset = checkpoint.get("dataset")
    for key in ("versionPrefix", "manifestSha256", "schemaVersion"):
        compare(
            f"dataset.{key}",
            checkpoint_dataset.get(key) if isinstance(checkpoint_dataset, dict) else None,
            dataset_metadata.get(key),
        )
    compare("anchors", checkpoint.get("anchors"), anchor_specification)
    compare("boxEncoding", checkpoint.get("boxEncoding", "anchor_offsets"), box_encoding(config.model_variant))

    previous_recipe = checkpoint.get("occlusion", occlusion_recipe(TrainingConfig()))
    compare("occlusion", previous_recipe, occlusion_recipe(config))
    if config.visible_loss_weight and not checkpoint.get("auxiliary_state_dict"):
        mismatches["auxiliaryState"] = {"checkpoint": "missing", "current": "required"}

    required_state = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "completedEpochs",
        "best_val_loss",
    )
    missing = [key for key in required_state if key not in checkpoint]
    if missing:
        mismatches["resumeState"] = {"checkpoint": missing, "current": "all required"}
    return mismatches


def make_training_checkpoint(
    *,
    epoch: int,
    training_complete: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_val_loss: float,
    config: TrainingConfig,
    dataset_metadata: Dict,
    anchor_generator: PersonAnchorGenerator,
    sampling_generator: Optional[torch.Generator],
    best_val_map: float = -1.0,
    last_validation_metrics: Optional[Dict] = None,
) -> Dict:
    return {
        "epoch": epoch,
        "completedEpochs": epoch + 1,
        "trainingComplete": training_complete,
        "model_state_dict": detector_state_dict(model),
        "auxiliary_state_dict": (model.auxiliary_head.state_dict()
                                 if getattr(model, "auxiliary_head", None) is not None else {}),
        "occlusion": occlusion_recipe(config),
        "selection": {"metric": config.checkpoint_selection, "epoch": epoch + 1},
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                            *np.random.get_state()[2:]),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "sampler_generator_state": (
            sampling_generator.get_state() if sampling_generator is not None else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "best_val_loss": best_val_loss,
        "best_val_map_50_95": best_val_map,
        "last_validation_metrics": dict(last_validation_metrics or {}),
        "config": config.__dict__,
        "dataset": dataset_metadata,
        "modelFormatVersion": model_format(config.model_variant),
        "boxEncoding": box_encoding(config.model_variant),
        "anchors": anchor_generator.specification(),
        "assignment": {"name": "ATSS-point" if config.model_variant == "clean_ltrb" else "ATSS",
                       "topKPerLevel": config.atss_topk},
        "classificationHead": {"name": "binary-quality-v1", "target": "predictedIoU"},
    }


def save_training_checkpoint(checkpoint: Dict, path: str) -> None:
    """Atomically replace a checkpoint so interruption cannot leave a partial file."""
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def log_epoch_to_tensorboard(
    writer: SummaryWriter,
    epoch_number: int,
    train_metrics: Dict,
    val_metrics: Dict,
    learning_rate: float,
) -> None:
    """Write stable scalar names so resumed and compared runs share one dashboard."""
    writer.add_scalar("Loss/train", train_metrics["loss"], epoch_number)
    writer.add_scalar("Loss/validation", val_metrics["loss"], epoch_number)
    writer.add_scalar("Loss/classification", train_metrics["cls_loss"], epoch_number)
    writer.add_scalar("Loss/localization", train_metrics["loc_loss"], epoch_number)
    writer.add_scalar("Optimization/learning_rate", learning_rate, epoch_number)
    if "unmatched_gt" in train_metrics:
        writer.add_scalar("Assignment/unmatched_gt", train_metrics["unmatched_gt"], epoch_number)
    writer.add_scalar(
        "Timing/train_epoch_seconds", train_metrics["duration_seconds"], epoch_number
    )
    metric_tags = {
        "mAP@0.50": "Metrics/validation_mAP_50",
        "mAP@0.50:0.95": "Metrics/validation_mAP_50_95",
        "Recall@FPPI=0.10": "Metrics/validation_recall_fppi_0_10",
    }
    for metric_name, tag in metric_tags.items():
        if metric_name in val_metrics:
            writer.add_scalar(tag, val_metrics[metric_name], epoch_number)
    for split, metrics in (("train", train_metrics), ("validation", val_metrics)):
        for name, value in metrics.items():
            if name.startswith(("visible_", "repgt_", "repbox_", "auxiliary_")) or name == "detection_loss":
                writer.add_scalar(f"Occlusion/{split}/{name}", value, epoch_number)
    writer.flush()


class DetectorTrainingPipeline:
    """Own the canonical train/validate/checkpoint/export workflow."""

    def __init__(self, config: TrainingConfig):
        config.validate_occlusion()
        if config.regression_depth not in (1, 2):
            raise ValueError("TRAINING_REGRESSION_DEPTH must be 1 or 2")
        if config.model_variant == "anchor" and config.regression_depth != 1:
            raise ValueError("The anchor variant requires TRAINING_REGRESSION_DEPTH=1")
        self.config = config

    @classmethod
    def from_environment(cls) -> "DetectorTrainingPipeline":
        config = TrainingConfig(
            export_openvino_after_training=os.getenv("TRAINING_EXPORT_OPENVINO", "true").lower() == "true",
            backbone=os.getenv("TRAINING_BACKBONE", "mobilenetv3_small"),
            use_stride4=os.getenv("TRAINING_USE_STRIDE4", "false").lower() == "true",
            use_pan=os.getenv("TRAINING_USE_PAN", "true").lower() == "true",
            regression_depth=int(os.getenv("TRAINING_REGRESSION_DEPTH", "2")),
            model_variant=os.getenv("TRAINING_MODEL_VARIANT", DEFAULT_MODEL_VARIANT),
            visible_loss_weight=float(os.getenv("TRAINING_VISIBLE_LOSS_WEIGHT", "0.0")),
            repgt_loss_weight=float(os.getenv("TRAINING_REPGT_LOSS_WEIGHT", "0.0")),
            repbox_loss_weight=float(os.getenv("TRAINING_REPBOX_LOSS_WEIGHT", "0.0")),
            auxiliary_ramp_epochs=int(os.getenv("TRAINING_AUXILIARY_RAMP_EPOCHS", "5")),
            repgt_sigma=float(os.getenv("TRAINING_REPGT_SIGMA", "0.5")),
            repbox_sigma=float(os.getenv("TRAINING_REPBOX_SIGMA", "0.0")),
            repbox_predictions_per_gt=int(os.getenv("TRAINING_REPBOX_PREDICTIONS_PER_GT", "4")),
            repulsion_chunk_size=int(os.getenv("TRAINING_REPULSION_CHUNK_SIZE", "64")),
            checkpoint_selection=os.getenv("TRAINING_CHECKPOINT_SELECTION", "detection_loss"),
            azurite_endpoint=os.getenv("AZURITE_BLOB_ENDPOINT", "http://127.0.0.1:10000/devstoreaccount1"),
            azurite_access_key=os.getenv("AZURITE_ACCOUNT_NAME", "devstoreaccount1"),
            azurite_secret_key=os.getenv("AZURITE_ACCOUNT_KEY", ""),
            azurite_connection_string=os.getenv("AZURITE_CONNECTION_STRING", ""),
            azurite_data_bucket=os.getenv("AZURITE_DATA_CONTAINER", "computer-vision-data"),
            azurite_model_bucket=os.getenv("AZURITE_MODEL_CONTAINER", "computer-vision-models"),
            use_azurite=os.getenv("USE_AZURITE", "true").lower() == "true",
            data_root=os.getenv("DATA_ROOT", "./data"),
            enable_quantization=os.getenv("ENABLE_QUANTIZATION", "false").lower() == "true",
            input_height=int(os.getenv("TRAINING_INPUT_HEIGHT", "360")),
            input_width=int(os.getenv("TRAINING_INPUT_WIDTH", "640")),
            batch_size=int(os.getenv("TRAINING_BATCH_SIZE", "32")),
            num_workers=int(os.getenv("TRAINING_NUM_WORKERS", "1")),
            num_epochs=int(os.getenv("TRAINING_EPOCHS", "100")),
            validation_ap_every_n_epochs=int(
                os.getenv("TRAINING_AP_EVERY_N_EPOCHS", "5")
            ),
            validation_ap_score_threshold=float(
                os.getenv("TRAINING_AP_SCORE_THRESHOLD", "0.01")
            ),
            validation_ap_nms_threshold=float(
                os.getenv("TRAINING_AP_NMS_THRESHOLD", "0.5")
            ),
            validation_ap_pre_nms_topk=int(
                os.getenv("TRAINING_AP_PRE_NMS_TOPK", "1000")
            ),
            validation_ap_max_detections=int(
                os.getenv("TRAINING_AP_MAX_DETECTIONS", "100")
            ),
            validation_recall_fppi=float(
                os.getenv("TRAINING_RECALL_FPPI", "0.1")
            ),
            tensorboard_enabled=os.getenv(
                "TRAINING_TENSORBOARD_ENABLED", "true"
            ).lower() == "true",
            tensorboard_log_dir=os.getenv(
                "TRAINING_TENSORBOARD_LOG_DIR", "tensorboard"
            ),
        )
        config.validate_occlusion()
        if config.num_epochs < 1:
            raise ValueError("TRAINING_EPOCHS must be at least 1")
        if config.input_height < 32 or config.input_width < 32:
            raise ValueError("TRAINING_INPUT_HEIGHT and TRAINING_INPUT_WIDTH must be at least 32")
        if config.input_width <= config.input_height:
            raise ValueError("TRAINING_INPUT_WIDTH must be greater than TRAINING_INPUT_HEIGHT")
        if config.batch_size < 1:
            raise ValueError("TRAINING_BATCH_SIZE must be at least 1")
        if config.num_workers < 0:
            raise ValueError("TRAINING_NUM_WORKERS cannot be negative")
        if config.validation_ap_every_n_epochs < 0:
            raise ValueError("TRAINING_AP_EVERY_N_EPOCHS cannot be negative")
        if config.validation_ap_pre_nms_topk < 1:
            raise ValueError("TRAINING_AP_PRE_NMS_TOPK must be at least 1")
        if config.validation_ap_max_detections < 1:
            raise ValueError("TRAINING_AP_MAX_DETECTIONS must be at least 1")
        for name, value in (
            ("TRAINING_AP_SCORE_THRESHOLD", config.validation_ap_score_threshold),
            ("TRAINING_AP_NMS_THRESHOLD", config.validation_ap_nms_threshold),
            ("TRAINING_RECALL_FPPI", config.validation_recall_fppi),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not config.tensorboard_log_dir.strip():
            raise ValueError("TRAINING_TENSORBOARD_LOG_DIR cannot be empty")
        return cls(config)

    def run(self) -> None:
        config = self.config
        if config.backbone not in ("mobilenetv3_small", "mobilenetv4_conv_small"):
            raise ValueError(f"Unknown TRAINING_BACKBONE: {config.backbone!r}")
        box_encoding(config.model_variant)  # Validate before loading data.
        if config.enable_quantization:
            raise RuntimeError(
                "The legacy in-training QAT path is retired. Train FP32 with "
                "ENABLE_QUANTIZATION=false, then run python -m "
                "person_detection.optimization.pipeline for manifest-driven "
                "accuracy-controlled INT8 calibration."
            )

        print("=" * 60)
        print("SSD Person Detection Training Pipeline")
        print(f"Backbone: {config.backbone}")
        print(f"Variant: {config.model_variant}; ATSS + quality-aware binary classification")
        print("=" * 60)
        print(f"Device: {config.device}")
        print(f"Input canvas: {config.input_height}x{config.input_width} (HxW)")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Occlusion training settings: {json.dumps(occlusion_recipe(config), sort_keys=True)}")
        print(
            "Validation AP: "
            + (
                f"every {config.validation_ap_every_n_epochs} epoch(s)"
                if config.validation_ap_every_n_epochs > 0 else "disabled"
            )
        )
        print(
            f"TensorBoard: {config.tensorboard_log_dir}"
            if config.tensorboard_enabled else "TensorBoard: disabled"
        )
        print(f"Use AMP: {config.use_amp}")
        print(f"Use Focal Loss: {config.use_focal_loss}")
        print(f"Enable Quantization: {config.enable_quantization}")
        if config.use_azurite:
            print(f"Azurite endpoint: {config.azurite_endpoint}")
        else:
            print(f"Local data: {config.data_root}")
        print("=" * 60)

        azurite_client = AzuriteClient(config)
        exporter = OpenVINOExporter(config)
        best_checkpoint_path = "best_model_fp32.pth"
        best_ap_checkpoint_path = "best_model_ap.pth"
        best_loss_checkpoint_path = "best_model_loss.pth"
        last_checkpoint_path = "last_training_checkpoint.pth"
        checkpoint_prefix = ("person_detector_ssd" if config.model_variant == "anchor"
                             else f"person_detector_ssd/{config.model_variant}")

        print("\nLoading datasets...")
        train_dataset = CanonicalPersonDetectionDataset(
            azurite_client,
            split="train",
            input_height=config.input_height,
            input_width=config.input_width,
            augment=True,
            coarse_dropout=config.visible_loss_weight == 0,
        )
        val_dataset = CanonicalPersonDetectionDataset(
            azurite_client,
            split="val",
            input_height=config.input_height,
            input_width=config.input_width,
            augment=False,
        )
        print("\nValidating dataset structure...")
        train_dataset.validate_samples(num_samples=3)
        val_dataset.validate_samples(num_samples=3)

        train_sampler = None
        sampling_generator = None
        if config.use_stratified_oversampling:
            sampling_weights = train_dataset.build_sampling_weights()
            sampling_generator = torch.Generator().manual_seed(config.sampling_seed)
            train_sampler = WeightedRandomSampler(
                sampling_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampling_generator,
            )
            print(f"Using {train_dataset.dataset_metadata['samplingPolicy']['name']} sampling")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            worker_init_fn=initialize_loader_worker,
            collate_fn=collate_fn,
            pin_memory=config.device == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=initialize_loader_worker,
            collate_fn=collate_fn,
            pin_memory=config.device == "cuda",
        )

        print("\nInitializing model...")
        model = SSDPersonDetector(
            backbone=config.backbone, use_stride4=config.use_stride4,
            use_pan=config.use_pan, regression_depth=config.regression_depth,
            model_variant=config.model_variant,
            visible_auxiliary=config.visible_loss_weight > 0,
            num_classes=config.num_classes,
            input_height=config.input_height,
            input_width=config.input_width,
        ).to(config.device)
        anchors = model.anchor_generator.get_anchors()
        model_summary(
            model,
            model_type=ModelType.SINGLE_INPUT,
            input1_size=(config.batch_size, 3, config.input_height, config.input_width),
        )
        criterion = SSDLoss(
            num_classes=config.num_classes,
            neg_pos_ratio=config.neg_pos_ratio,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            input_height=config.input_height,
            input_width=config.input_width,
            anchors_per_level=model.anchor_generator.num_anchors_per_level,
            atss_topk=config.atss_topk,
            giou_weight=config.giou_weight,
            model_variant=config.model_variant,
            occlusion_config=config,
            distance_scales=getattr(model, "distance_scales", None),
        )

        if config.use_varied_lr:
            optimizer = optim.SGD(
                ([{"params": model.auxiliary_head.parameters(), "lr": config.learning_rate}]
                 if model.auxiliary_head is not None else []) + [
                    {"params": model.features.parameters(), "lr": config.learning_rate * 0.1},
                    {"params": model.fpn.parameters(), "lr": config.learning_rate},
                    {"params": model.extra_layers.parameters(), "lr": config.learning_rate},
                    {"params": model.detection_heads.parameters(), "lr": config.learning_rate},
                ],
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        warmup = WarmupScheduler(optimizer, config.warmup_epochs, config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.num_epochs - config.warmup_epochs),
            eta_min=config.learning_rate * 0.1,
        )
        scaler = GradScaler(device=config.device, enabled=config.use_amp)

        start_epoch = 0
        best_val_loss = float("inf")
        best_val_map = -1.0
        training_complete = False
        resume_path = next(
            (path for path in (last_checkpoint_path, best_checkpoint_path) if os.path.exists(path)),
            None,
        )
        if resume_path is not None:
            candidate = torch.load(resume_path, map_location=config.device, weights_only=True)
            mismatches = checkpoint_resume_mismatches(
                candidate,
                config,
                train_dataset.dataset_metadata,
                model.anchor_generator.specification(),
            )
            if mismatches:
                raise RuntimeError(
                    f"Checkpoint {resume_path} is incompatible: {mismatches}. "
                    "Use a fresh training run/directory to preserve the baseline."
                )
            else:
                load_training_state(model, candidate)
                if "python_rng_state" in candidate:
                    random.setstate(candidate["python_rng_state"])
                if "numpy_rng_state" in candidate:
                    numpy_state = candidate["numpy_rng_state"]
                    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                         *numpy_state[2:]))
                optimizer.load_state_dict(candidate["optimizer_state_dict"])
                scheduler.load_state_dict(candidate["scheduler_state_dict"])
                scaler.load_state_dict(candidate["scaler_state_dict"])
                start_epoch = int(candidate["completedEpochs"])
                best_val_loss = float(candidate["best_val_loss"])
                best_val_map = float(candidate.get("best_val_map_50_95", -1.0))
                training_complete = bool(candidate.get("trainingComplete", False))
                if sampling_generator is not None and candidate.get("sampler_generator_state") is not None:
                    sampling_generator.set_state(candidate["sampler_generator_state"])
                if candidate.get("torch_rng_state") is not None:
                    torch.set_rng_state(candidate["torch_rng_state"].cpu())
                if torch.cuda.is_available() and candidate.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all(candidate["cuda_rng_state"])
                print(
                    f"Resuming {resume_path} after epoch {start_epoch}; "
                    f"best validation loss is {best_val_loss:.4f}; "
                    f"best validation AP50:95 is {best_val_map:.4f}"
                )

        writer = None
        if config.tensorboard_enabled:
            tensorboard_path = Path(config.tensorboard_log_dir)
            tensorboard_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(
                log_dir=str(tensorboard_path),
                purge_step=start_epoch + 1 if start_epoch > 0 else None,
            )
            writer.add_text(
                "run/config",
                "```json\n" + json.dumps(config.__dict__, indent=2, default=str) + "\n```",
                start_epoch,
            )
            writer.flush()

        if training_complete and start_epoch >= config.num_epochs:
            print(
                f"Training is already complete ({start_epoch}/{config.num_epochs} epochs); "
                "skipping optimization steps."
            )
        else:
            print("\n" + "=" * 60)
            print(f"PHASE 1: FP32 Training (epochs {start_epoch + 1}-{config.num_epochs})")
            print("=" * 60)
            for epoch in range(start_epoch, config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
                print("-" * 40)
                is_warmup = warmup.step(epoch)
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, criterion,
                    anchors, config.device, epoch, config, scaler
                )
                compute_ap = (
                    config.validation_ap_every_n_epochs > 0
                    and (
                        (epoch + 1) % config.validation_ap_every_n_epochs == 0
                        or epoch + 1 == config.num_epochs
                    )
                )
                if compute_ap:
                    print(
                        "Calculating validation AP at score threshold "
                        f"{config.validation_ap_score_threshold:.3f}..."
                    )
                val_metrics = validate(
                    model,
                    val_loader,
                    criterion,
                    anchors,
                    config.device,
                    compute_ap=compute_ap,
                    config=config,
                )
                if not is_warmup:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                summary = (
                    f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.6f}"
                )
                if compute_ap:
                    summary += (
                        f" | AP50: {val_metrics['mAP@0.50']:.4f}"
                        f" | AP50:95: {val_metrics['mAP@0.50:0.95']:.4f}"
                        f" | Recall@FPPI=0.10: "
                        f"{val_metrics['Recall@FPPI=0.10']:.4f}"
                    )
                print(summary)

                improved = val_metrics["detection_loss"] < best_val_loss
                if improved:
                    best_val_loss = val_metrics["detection_loss"]
                measured_map = val_metrics.get("mAP@0.50:0.95")
                map_improved = measured_map is not None and measured_map > best_val_map
                if map_improved:
                    best_val_map = measured_map
                if writer is not None:
                    log_epoch_to_tensorboard(
                        writer, epoch + 1, train_metrics, val_metrics, current_lr
                    )
                checkpoint = make_training_checkpoint(
                    epoch=epoch,
                    training_complete=epoch + 1 >= config.num_epochs,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    config=config,
                    dataset_metadata=train_dataset.dataset_metadata,
                    anchor_generator=model.anchor_generator,
                    sampling_generator=sampling_generator,
                    best_val_map=best_val_map,
                    last_validation_metrics=val_metrics,
                )
                save_training_checkpoint({**checkpoint, "selection": {"metric": "latest", "epoch": epoch + 1}},
                                         last_checkpoint_path)
                if config.use_azurite:
                    azurite_client.put_object(
                        config.azurite_model_bucket,
                        f"{checkpoint_prefix}/last_training_checkpoint.pth",
                        last_checkpoint_path,
                    )
                if improved:
                    save_training_checkpoint({**checkpoint, "selection": {"metric": "detection_loss", "epoch": epoch + 1}},
                                             best_loss_checkpoint_path)
                    if config.use_azurite:
                        azurite_client.put_object(config.azurite_model_bucket,
                            f"{checkpoint_prefix}/best_model_loss.pth", best_loss_checkpoint_path)
                if checkpoint_is_selected(config.checkpoint_selection, loss_improved=improved,
                                          ap_improved=map_improved, final_epoch=epoch + 1 == config.num_epochs):
                    save_training_checkpoint(checkpoint, best_checkpoint_path)
                    if config.use_azurite:
                        azurite_client.put_object(
                            config.azurite_model_bucket,
                            f"{checkpoint_prefix}/best_model_fp32.pth",
                            best_checkpoint_path,
                        )
                    print(f"✓ Saved selected FP32 model ({config.checkpoint_selection}, epoch {epoch + 1})")
                if map_improved:
                    save_training_checkpoint({**checkpoint, "selection": {"metric": "ap", "epoch": epoch + 1}},
                                             best_ap_checkpoint_path)
                    if config.use_azurite:
                        azurite_client.put_object(
                            config.azurite_model_bucket,
                            f"{checkpoint_prefix}/best_model_ap.pth",
                            best_ap_checkpoint_path,
                        )
                    print(
                        "✓ Saved best validation-AP model "
                        f"(AP50:95: {best_val_map:.4f})"
                    )
                print(f"✓ Saved resumable epoch {epoch + 1} state")
            print("\n✓ FP32 training complete!")

        if writer is not None:
            writer.close()

        if not config.export_openvino_after_training:
            print(f"Training complete; selected checkpoint: {Path(best_checkpoint_path).resolve()}")
            return

        print("\n" + "=" * 60)
        print("Loading best FP32 checkpoint...")
        print("=" * 60)
        if not os.path.exists(best_checkpoint_path):
            raise RuntimeError("Training completed without producing a best FP32 checkpoint")
        checkpoint = torch.load(best_checkpoint_path, map_location=config.device, weights_only=True)
        mismatches = checkpoint_resume_mismatches(
            checkpoint,
            config,
            train_dataset.dataset_metadata,
            model.anchor_generator.specification(),
        )
        if mismatches:
            raise RuntimeError(f"Best checkpoint is incompatible with this run: {mismatches}")
        model = SSDPersonDetector(backbone=config.backbone, use_stride4=config.use_stride4,
                                  use_pan=config.use_pan, regression_depth=config.regression_depth, model_variant=config.model_variant, num_classes=config.num_classes,
                                  input_height=config.input_height, input_width=config.input_width,
                                  pretrained=False).to(config.device).eval()
        load_detector_state_dict(model, checkpoint["model_state_dict"])
        print(
            f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(val_loss: {checkpoint['best_val_loss']})"
        )

        print("\n" + "=" * 60)
        print("PHASE 2: Export FP32 Model to OpenVINO")
        print("=" * 60)
        exporter.export_to_openvino(
            model,
            "person_detector_fp32.xml",
            input_height=config.input_height,
            input_width=config.input_width,
            compress_to_fp16=False,
        )

        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        print("\nOutput files:")
        print("  - last_training_checkpoint.pth (resumable training state)")
        print("  - best_model_fp32.pth (selected detection weights)")
        if os.path.exists(best_ap_checkpoint_path):
            print("  - best_model_ap.pth (best measured validation AP50:95 weights)")
        if config.tensorboard_enabled:
            print(f"  - {config.tensorboard_log_dir}/ (TensorBoard event logs)")
        print("  - person_detector_fp32.xml/bin (OpenVINO FP32)")

def main() -> None:
    DetectorTrainingPipeline.from_environment().run()


if __name__ == "__main__":
    main()
