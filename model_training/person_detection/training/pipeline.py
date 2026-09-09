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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import box_iou
import cv2
import numpy as np
import os
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import random

from person_detection.data.layout import (
    TRAINABLE_LABEL_STATUSES,
    load_citypersons_split,
    resolve_citypersons_prefix,
    version_blob,
)
from person_detection.data.dataset import CanonicalPersonDetectionDataset
from person_detection.modeling.assignment import ATSSAnchorAssigner

# Optional imports with fallbacks
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic transforms.")

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
class TrainingConfig:
    """Training configuration with sensible defaults"""
    # Data
    data_root: str = './data'
    input_size: int = 480  # Reduced from 600 - better latency, multi-scale compensates
    batch_size: int = 32   # Increased - more stable gradients
    num_workers: int = 1

    # Model
    num_classes: int = 1  # one sigmoid localization-quality logit per anchor

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

    # Legacy in-training QAT; release INT8 artifacts are built by the optimization pipeline.
    enable_quantization: bool = False
    qat_epochs: int = 10  # Quantization-aware training epochs
    qat_learning_rate: float = 1e-4  # Usually 1/10 of original
    calibration_samples: int = 1000  # Samples for calibration

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

class LegacyYoloPersonDetectionDataset(Dataset):
    """
    Person detection dataset with YOLO format annotations

    YOLO format: class_id center_x center_y width height (normalized 0-1)
    Example: 0 0.4 0.7 0.3 0.4

    Empty annotation file = no objects in image (valid negative example)
    """

    def __init__(
        self,
        azurite_client: AzuriteClient,
        split: str = 'train',
        input_size: int = 320,
        augment: bool = True
    ):
        self.azurite = azurite_client
        self.config = azurite_client.config
        self.split = split
        self.input_size = input_size
        self.augment = augment and (split == 'train')

        # Find all images
        self.samples = self._find_samples()

        # Analyze and report dataset structure
        self._report_dataset_structure()

        # Setup transforms
        self._setup_transforms()

    def _report_dataset_structure(self):
        """Report details about the dataset structure"""
        if not self.samples:
            print(f"⚠ Warning: No samples found for {self.split} split!")
            return

        # Extract city folders from paths
        cities = set()
        for img_path, _ in self.samples:
            cities.add(Path(img_path).parent.name)

        print(f"Found {len(self.samples)} images for {self.split} split")
        print(f"  Cities: {len(cities)} folders")
        if len(cities) <= 15:
            print(f"  → {', '.join(sorted(cities))}")
        else:
            city_list = sorted(cities)
            print(f"  → {', '.join(city_list[:10])}... and {len(cities)-10} more")

        # Show example mappings for verification
        print(f"  Example mappings:")
        for i, (img_path, label_path) in enumerate(self.samples[:2]):
            print(f"    Image: {img_path}")
            print(f"    Label: {label_path}")
            if i < 1:
                print()

    def validate_samples(self, num_samples: int = 5) -> bool:
        """
        Validate that image-label pairs exist and are correctly mapped

        Args:
            num_samples: Number of random samples to validate

        Returns:
            True if validation passes
        """
        import random

        if not self.samples:
            print("✗ No samples to validate")
            return False

        # Check a few random samples
        samples_to_check = random.sample(
            self.samples,
            min(num_samples, len(self.samples))
        )

        valid = 0
        missing_labels = 0

        for img_path, label_path in samples_to_check:
            # Check if image exists
            img_data = self.azurite.get_object_bytes(
                self.config.azurite_data_bucket,
                img_path
            )

            if img_data is None:
                print(f"  ✗ Image not found: {img_path}")
                continue

            # Check if label exists (empty is OK, missing file is noted)
            label_data = self.azurite.get_object_bytes(
                self.config.azurite_data_bucket,
                label_path
            )

            if label_data is None:
                missing_labels += 1

            valid += 1

        print(f"  Validation: {valid}/{len(samples_to_check)} images accessible")
        if missing_labels > 0:
            print(f"  ✗ {missing_labels} required labels are missing")

        return valid > 0 and missing_labels == 0

    def _find_samples(self) -> List[Tuple[str, str]]:
        """
        Load image-label pairs from the published immutable split manifest.

        Expected Azurite structure:
            computer-vision-data/
            └── datasets/citypersons/<version>/
            ├── images/
            │   ├── train/
            │   │   ├── aachen/
            │   │   │   ├── aachen_000000_000019_leftImg8bit.png
            │   │   │   └── ...
            │   │   ├── bochum/
            │   │   └── ...
            │   ├── val/
            │   └── test/
            └── labels/yolo-person-v1/
                ├── train/
                │   ├── aachen/
                │   │   ├── aachen_000000_000019_leftImg8bit.txt
                │   │   └── ...
                │   └── ...
                └── val/
        """
        bucket = self.config.azurite_data_bucket
        read_blob = lambda name: self.azurite.get_object_bytes(bucket, name)
        prefix = resolve_citypersons_prefix(read_blob)
        records = load_citypersons_split(read_blob, prefix, self.split)
        samples = []
        for record in records:
            if record.get("labelStatus") not in TRAINABLE_LABEL_STATUSES:
                continue
            image = record.get("image")
            label = record.get("yoloLabel")
            if not isinstance(image, str) or not isinstance(label, str):
                raise RuntimeError(f"Invalid trainable record in {self.split} split")
            samples.append((version_blob(prefix, image), version_blob(prefix, label)))

        # Sort for reproducibility
        samples.sort(key=lambda x: x[0])

        return samples

    def _setup_transforms(self):
        """
        Setup augmentation pipeline using albumentations native YOLO format

        YOLO format in albumentations: [x_center, y_center, width, height] (all normalized 0-1)
        This matches our annotation format directly, no conversion needed!
        """
        if HAS_ALBUMENTATIONS:
            if self.augment:
                self.transform = A.Compose([
                    # Spatial transforms
                    A.LongestMaxSize(max_size=int(self.input_size * random.uniform(1.0, 1.5))),
                    A.PadIfNeeded(
                        int(self.input_size * 1.1),
                        int(self.input_size * 1.1),
                        border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.OneOf([
                        A.RandomCrop(self.input_size, self.input_size),
                        A.CenterCrop(int(self.input_size * 0.7), int(self.input_size * 0.7)),  # Tight crops
                    ], p=1.0),
                    A.Resize(self.input_size, self.input_size),
                    A.HorizontalFlip(p=0.5),
                    A.Perspective(scale=(0.05, 0.10), p=0.35),
                    A.Affine(
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.12, 0.03)},
                        scale=(0.85, 1.15),
                        rotate=(-6, 6),
                        p=0.3
                    ),

                    # Occlusion
                    A.CoarseDropout(
                        max_holes=6, max_height=40, max_width=40,
                        min_holes=1, min_height=10, min_width=10,
                        fill_value=0, p=0.3
                    ),

                    # Color/lighting
                    A.OneOf([
                        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                        A.CLAHE(clip_limit=2.0),
                        A.ToGray(p=0.15),
                    ], p=0.5),

                    # Noise/blur
                    A.OneOf([
                        A.GaussianBlur(blur_limit=3),
                        A.MotionBlur(blur_limit=3),
                        A.GaussNoise(var_limit=(10, 50)),
                        A.ImageCompression(quality_lower=65, quality_upper=95, p=0.25),
                    ], p=0.2),

                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['labels'],
                    min_area=0.002,
                    min_visibility=0.3
                ))
            else:
                self.transform = A.Compose([
                    A.Resize(self.input_size, self.input_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(
                    format='yolo',  # Native YOLO format
                    label_fields=['labels']
                ))
        else:
            self.transform = None

    def _parse_yolo_annotation(self, label_data: Optional[bytes]) -> Tuple[List, List]:
        """
        Parse YOLO format annotation

        YOLO format: class_id center_x center_y width height (all normalized 0-1)
        Example: 0 0.4 0.7 0.3 0.4

        Returns:
            boxes: List of [cx, cy, w, h] in normalized coordinates (0-1)
            labels: List of class labels (1 = person, 0 = background)
        """
        boxes = []
        labels = []

        if label_data is None:
            return boxes, labels

        try:
            content = label_data.decode('utf-8')
        except:
            return boxes, labels

        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                yolo_class = int(parts[0])  # 0 = person
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])
            except ValueError:
                continue

            # Validate normalized coordinates are in valid range
            if not (0 <= cx_norm <= 1 and 0 <= cy_norm <= 1 and
                    0 < w_norm <= 1 and 0 < h_norm <= 1):
                continue

            # Filter out tiny boxes (in normalized coords)
            # 5 pixels on 320px image = 5/320 = 0.0156
            # 10 pixels on 320px image = 10/320 = 0.0312
            min_w_norm = 2 / self.input_size
            min_h_norm = 4 / self.input_size

            if w_norm < min_w_norm or h_norm < min_h_norm:
                continue

            # Keep in YOLO format (normalized center format)
            # Albumentations will handle the transformation
            boxes.append([cx_norm, cy_norm, w_norm, h_norm])

            # YOLO class 0 -> our class 1 (0 is background)
            labels.append(yolo_class + 1)

        return boxes, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        img_data = self.azurite.get_object_bytes(
            self.config.azurite_data_bucket,
            img_path
        )
        if img_data is None:
            return self._get_blank_sample(idx)

        # Decode image
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return self._get_blank_sample(idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        # Load annotations (returns YOLO format: [cx, cy, w, h] normalized)
        label_data = self.azurite.get_object_bytes(
            self.config.azurite_data_bucket,
            label_path
        )
        boxes, labels = self._parse_yolo_annotation(label_data)

        # Apply transforms
        if self.transform is not None:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes if boxes else [],
                    labels=labels if labels else []
                )
                image = transformed['image']
                boxes = list(transformed['bboxes'])  # Still in YOLO format (normalized)
                for box in transformed['bboxes']:
                    cx, cy, w, h = box
                    assert 0 <= cx <= 1 and 0 <= cy <= 1, f"Invalid center: {box}"
                    assert 0 < w <= 1 and 0 < h <= 1, f"Invalid size: {box}"
                labels = list(transformed['labels'])
            except Exception as e:
                # Fallback if augmentation fails - apply basic preprocessing
                image = cv2.resize(image, (self.input_size, self.input_size))
                image = image.astype(np.float32) / 255.0
                # Apply ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                image = (image - mean) / std
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                # Boxes stay in YOLO format (already normalized)
        else:
            # Basic transform without albumentations - apply same preprocessing
            image = cv2.resize(image, (self.input_size, self.input_size))
            image = image.astype(np.float32) / 255.0
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            # Boxes stay in YOLO format (already normalized)

        # Convert boxes from YOLO format (normalized) to pascal_voc format (absolute pixels)
        # YOLO: [cx, cy, w, h] normalized -> Pascal VOC: [x1, y1, x2, y2] absolute
        if boxes:
            converted_boxes = []
            for box in boxes:
                cx, cy, w, h = box
                # Convert to absolute pixel coordinates on the input_size image
                x1 = (cx - w / 2) * self.input_size
                y1 = (cy - h / 2) * self.input_size
                x2 = (cx + w / 2) * self.input_size
                y2 = (cy + h / 2) * self.input_size

                # Clamp to image bounds
                x1 = max(0, min(x1, self.input_size))
                y1 = max(0, min(y1, self.input_size))
                x2 = max(0, min(x2, self.input_size))
                y2 = max(0, min(y2, self.input_size))

                converted_boxes.append([x1, y1, x2, y2])

            boxes = torch.as_tensor(converted_boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return image, {'boxes': boxes, 'labels': labels, 'image_id': idx}

    def _get_blank_sample(self, idx):
        """Return blank sample for failed loads"""
        image = torch.zeros(3, self.input_size, self.input_size)
        return image, {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': idx
        }

# ============================================================================
# ANCHOR GENERATOR
# ============================================================================

class PersonAnchorGenerator:
    """
    Anchor generator optimized for person detection

    People are typically:
    - Taller than wide (aspect ratios 1:2, 1:3)
    - Various sizes from distant to close
    """

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

        self.strides = [8, 16, 32, 64, 128]  # Fixed strides at each level

        # Feature map size = input_size / stride
        self.feature_map_sizes = [
            (input_size + stride - 1) // stride  # Ceiling division
            for stride in self.strides
        ]

        # In PersonAnchorGenerator.__init__
        self.scales = [
            [0.02, 0.04],           # P2: tiny distant people
            [0.06, 0.10],           # P3: small people
            [0.16, 0.24],           # P4: medium people
            [0.32, 0.48, 0.56],     # P5: large people - ADD intermediate
            [0.64, 0.80, 0.95],     # P6: very close - ADD near-full-frame
        ]

        # width / height ratios; pedestrian anchors must therefore be below 1.
        self.aspect_ratios = [
            [0.15, 0.25, 0.40],
            [0.15, 0.25, 0.40],
            [0.20, 0.33, 0.50],
            [0.25, 0.50, 1.00],
            [0.25, 0.50, 1.00],
        ]

        self.anchors = self._generate_anchors()
        self.num_anchors_per_level = self._count_anchors_per_level()
        print(f"Generated {len(self.anchors)} anchors across {len(self.feature_map_sizes)} levels")

    def _generate_anchors(self) -> torch.Tensor:
        """Generate all anchor boxes"""
        all_anchors = []

        for level_idx, fmap_size in enumerate(self.feature_map_sizes):
            scales = self.scales[level_idx]
            ratios = self.aspect_ratios[level_idx]

            for i in range(fmap_size):
                for j in range(fmap_size):
                    cx = (j + 0.5) / fmap_size
                    cy = (i + 0.5) / fmap_size

                    for scale in scales:
                        for ratio in ratios:
                            w = scale * np.sqrt(ratio)
                            h = scale / np.sqrt(ratio)
                            all_anchors.append([cx, cy, w, h])

        return torch.tensor(all_anchors, dtype=torch.float32)

    def _count_anchors_per_level(self) -> List[int]:
        """Count anchors per feature map level"""
        counts = []
        for level_idx, fmap_size in enumerate(self.feature_map_sizes):
            n_scales = len(self.scales[level_idx])
            n_ratios = len(self.aspect_ratios[level_idx])
            counts.append(fmap_size * fmap_size * n_scales * n_ratios)
        return counts

    def get_anchors(self) -> torch.Tensor:
        return self.anchors

    def get_num_anchors_per_location(self, level: int) -> int:
        return len(self.scales[level]) * len(self.aspect_ratios[level])

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficiency"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
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

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 convs to unify channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions (reduce aliasing after upsampling)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
        # self.p2_refine = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )

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
        # outputs[0] = self.p2_refine(outputs[0])  # Extra processing for highest res
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
    """Enhanced SSD with FPN and attention mechanisms"""

    def __init__(self, num_classes: int = 1, input_size: int = 640, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Backbone
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = backbone.features
        self.feature_indices = [3, 6, 12]

        # Extra layers
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(576, 128, kernel_size=1),  # Changed from 960
                nn.BatchNorm2d(128),
                nn.SiLU (inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True)
            ),
        ])

        # FPN to unify channels
        backbone_channels = [24, 40, 576, 256, 128]
        fpn_channels = 128
        self.fpn = FPN(backbone_channels, out_channels=fpn_channels)

        # Anchor generator
        self.anchor_generator = PersonAnchorGenerator(input_size)

        # Detection heads with attention (now all same channel dim due to FPN)
        self.detection_heads = nn.ModuleList([
            AttentionDetectionHead(
                fpn_channels,
                self.anchor_generator.get_num_anchors_per_location(i),
                num_classes
            )
            for i in range(5)
        ])

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)

        for extra_layer in self.extra_layers:
            x = extra_layer(x)
            features.append(x)

        # Apply FPN
        fpn_features = self.fpn(features)

        # Detection heads
        all_cls, all_bbox = [], []
        for feat, head in zip(fpn_features, self.detection_heads):
            cls, bbox = head(feat)
            all_cls.append(cls)
            all_bbox.append(bbox)

        return torch.cat(all_cls, dim=1), torch.cat(all_bbox, dim=1)


MODEL_FORMAT_VERSION = 2


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

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(target == 0, 1 - self.alpha, self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class LegacySSDLoss(nn.Module):
    """SSD Loss with hard negative mining and focal loss option"""

    def __init__(
        self,
        num_classes: int = 2,
        neg_pos_ratio: int = 3,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.35,
        focal_gamma: float = 1.5,
        input_size: int = 320
    ):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.use_focal_loss = use_focal_loss
        self.input_size = input_size

        if use_focal_loss:
            self.cls_loss_fn = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.cls_loss_fn = None

    def encode_boxes(self, gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / self.input_size
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / self.input_size
        gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]) / self.input_size
        gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]) / self.input_size

        dx = (gt_cx - anchors[:, 0]) / (anchors[:, 2] + 1e-6)
        dy = (gt_cy - anchors[:, 1]) / (anchors[:, 3] + 1e-6)
        dw = torch.log(gt_w / (anchors[:, 2] + 1e-6) + 1e-6)
        dh = torch.log(gt_h / (anchors[:, 3] + 1e-6) + 1e-6)

        return torch.stack([dx, dy, dw, dh], dim=1)

    @staticmethod
    def anchors_overlapping_ignore(anchor_boxes: torch.Tensor, ignore_boxes: torch.Tensor,
                                   threshold: float = 0.5) -> torch.Tensor:
        """Mask anchors whose area is substantially covered by an ignore region."""
        if ignore_boxes.numel() == 0:
            return torch.zeros(anchor_boxes.size(0), dtype=torch.bool, device=anchor_boxes.device)
        top_left = torch.maximum(anchor_boxes[:, None, :2], ignore_boxes[None, :, :2])
        bottom_right = torch.minimum(anchor_boxes[:, None, 2:], ignore_boxes[None, :, 2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
        anchor_area = ((anchor_boxes[:, 2] - anchor_boxes[:, 0]).clamp(min=1e-6) *
                       (anchor_boxes[:, 3] - anchor_boxes[:, 1]).clamp(min=1e-6))
        return (intersection / anchor_area[:, None]).amax(dim=1) >= threshold

    def match_anchors(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        anchors: torch.Tensor,
        ignore_boxes: Optional[torch.Tensor] = None,
        iou_threshold: float = 0.45,
        iou_threshold_neg: float = 0.35
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_anchors = anchors.size(0)
        device = anchors.device
        anchor_boxes = torch.zeros_like(anchors)
        anchor_boxes[:, 0] = (anchors[:, 0] - anchors[:, 2] / 2) * self.input_size
        anchor_boxes[:, 1] = (anchors[:, 1] - anchors[:, 3] / 2) * self.input_size
        anchor_boxes[:, 2] = (anchors[:, 0] + anchors[:, 2] / 2) * self.input_size
        anchor_boxes[:, 3] = (anchors[:, 1] + anchors[:, 3] / 2) * self.input_size
        ignored = self.anchors_overlapping_ignore(
            anchor_boxes,
            ignore_boxes if ignore_boxes is not None else anchor_boxes.new_zeros((0, 4)),
        )

        if gt_boxes.size(0) == 0:
            matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            matched_labels[ignored] = -1
            return (
                torch.zeros(num_anchors, 4, device=device),
                matched_labels,
                torch.zeros(num_anchors, dtype=torch.bool, device=device),
            )

        ious = box_iou(anchor_boxes, gt_boxes)

        best_gt_iou, best_gt_idx = ious.max(dim=1)
        best_anchor_iou, best_anchor_idx = ious.max(dim=0)

        for gt_idx, anchor_idx in enumerate(best_anchor_idx):
            best_gt_iou[anchor_idx] = 2.0
            best_gt_idx[anchor_idx] = gt_idx

        matched_labels = gt_labels[best_gt_idx]
        matched_boxes = gt_boxes[best_gt_idx]

        matched_labels[best_gt_iou < iou_threshold] = 0

        ignore_mask = (best_gt_iou >= iou_threshold_neg) & (best_gt_iou < iou_threshold)
        matched_labels[ignore_mask] = -1
        matched_labels[ignored & (matched_labels == 0)] = -1

        positive_mask = matched_labels > 0

        return matched_boxes, matched_labels, positive_mask

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

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        anchors: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        device = pred_cls.device
        batch_size = pred_cls.size(0)
        anchors = anchors.to(device)

        total_cls_loss = 0
        total_loc_loss = 0
        total_pos = 0

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            ignore_boxes = targets[i].get('ignore_regions', gt_boxes.new_zeros((0, 4))).to(device)

            matched_boxes, matched_labels, pos_mask = self.match_anchors(
                gt_boxes, gt_labels, anchors, ignore_boxes=ignore_boxes
            )

            valid_mask = matched_labels >= 0

            if self.use_focal_loss:
                cls_loss_per_anchor = self.cls_loss_fn(
                    pred_cls[i][valid_mask],
                    matched_labels[valid_mask]
                )
                cls_loss = cls_loss_per_anchor.sum()
            else:
                cls_loss_per_anchor = nn.functional.cross_entropy(
                    pred_cls[i], matched_labels.clamp(min=0),
                    reduction='none'
                )
                selected_mask = self.hard_negative_mining(
                    cls_loss_per_anchor, matched_labels, pos_mask
                )
                cls_loss = cls_loss_per_anchor[selected_mask].sum()

            total_cls_loss += cls_loss

            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                encoded_gt = self.encode_boxes(
                    matched_boxes[pos_mask],
                    anchors[pos_mask]
                )

                loc_loss = nn.functional.smooth_l1_loss(
                    pred_boxes[i][pos_mask],
                    encoded_gt,
                    reduction='sum'
                )
                total_loc_loss += loc_loss

            total_pos += num_pos

        num_pos_total = max(total_pos, 1)
        cls_loss = total_cls_loss / num_pos_total
        loc_loss = total_loc_loss / num_pos_total

        total_loss = cls_loss + loc_loss

        loss_dict = {
            'cls_loss': cls_loss.item(),
            'loc_loss': loc_loss if isinstance(loc_loss, float) else loc_loss.item(),
            'num_pos': total_pos
        }

        return total_loss, loss_dict


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
        input_size: int = 320,
        anchors_per_level: Optional[List[int]] = None,
        atss_topk: int = 9,
        giou_weight: float = 2.0,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("SSDLoss requires the one-logit binary detection head")
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.use_focal_loss = use_focal_loss
        self.input_size = input_size
        self.anchors_per_level = list(anchors_per_level or [])
        self.assigner = ATSSAnchorAssigner(top_k=atss_topk)
        self.giou_weight = giou_weight
        self.cls_loss_fn = QualityFocalLoss(focal_alpha, focal_gamma)

    def anchor_boxes(self, anchors: torch.Tensor) -> torch.Tensor:
        boxes = torch.zeros_like(anchors)
        boxes[:, 0] = (anchors[:, 0] - anchors[:, 2] / 2) * self.input_size
        boxes[:, 1] = (anchors[:, 1] - anchors[:, 3] / 2) * self.input_size
        boxes[:, 2] = (anchors[:, 0] + anchors[:, 2] / 2) * self.input_size
        boxes[:, 3] = (anchors[:, 1] + anchors[:, 3] / 2) * self.input_size
        return boxes

    def decode_boxes(self, offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        centers_x = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
        centers_y = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
        widths = offsets[:, 2].clamp(max=10).exp() * anchors[:, 2]
        heights = offsets[:, 3].clamp(max=10).exp() * anchors[:, 3]
        return torch.stack([
            (centers_x - widths / 2) * self.input_size,
            (centers_y - heights / 2) * self.input_size,
            (centers_x + widths / 2) * self.input_size,
            (centers_y + heights / 2) * self.input_size,
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
    ) -> Tuple[torch.Tensor, Dict]:
        if pred_cls.ndim != 3 or pred_cls.size(-1) != 1:
            raise ValueError("Quality-aware classification output must have shape [B, N, 1]")
        device = pred_cls.device
        anchors = anchors.to(device)
        total_cls_loss = pred_cls.new_zeros(())
        total_loc_loss = pred_cls.new_zeros(())
        total_quality = pred_cls.new_zeros(())
        total_pos = 0

        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            ignore_boxes = target.get("ignore_regions", gt_boxes.new_zeros((0, 4))).to(device)
            matched_boxes, matched_labels, positive_mask = self.match_anchors(
                gt_boxes, gt_labels, anchors, ignore_boxes=ignore_boxes
            )
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
                selected = LegacySSDLoss.hard_negative_mining(
                    self, losses, matched_labels, positive_mask
                )
                total_cls_loss += losses[selected & valid_mask].sum()
            total_pos += int(positive_mask.sum())

        normalizer = max(total_pos, 1)
        cls_loss = total_cls_loss / normalizer
        loc_loss = total_loc_loss / normalizer
        total_loss = cls_loss + loc_loss
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "loc_loss": loc_loss.item(),
            "mean_quality_target": (total_quality / normalizer).item(),
            "num_pos": total_pos,
        }

# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def decode_boxes(pred_offsets: torch.Tensor, anchors: torch.Tensor, input_size: int) -> torch.Tensor:
    """
    Decode predicted offsets to absolute box coordinates.

    Args:
        pred_offsets: [N, 4] tensor of (dx, dy, dw, dh)
        anchors: [N, 4] tensor of (cx, cy, w, h) in normalized coords
        input_size: Image size for converting to absolute coords

    Returns:
        boxes: [N, 4] tensor of (x1, y1, x2, y2) in absolute pixels
    """
    pred_cx = pred_offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    pred_cy = pred_offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    pred_w = torch.exp(pred_offsets[:, 2].clamp(max=10)) * anchors[:, 2]
    pred_h = torch.exp(pred_offsets[:, 3].clamp(max=10)) * anchors[:, 3]

    x1 = (pred_cx - pred_w / 2) * input_size
    y1 = (pred_cy - pred_h / 2) * input_size
    x2 = (pred_cx + pred_w / 2) * input_size
    y2 = (pred_cy + pred_h / 2) * input_size

    return torch.stack([x1, y1, x2, y2], dim=1)

def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-wise NMS."""
    from torchvision.ops import nms

    keep_boxes, keep_scores, keep_labels = [], [], []

    for class_id in labels.unique():
        if class_id == 0:  # Skip background
            continue

        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Score threshold
        score_mask = class_scores > score_threshold
        class_boxes = class_boxes[score_mask]
        class_scores = class_scores[score_mask]

        if len(class_boxes) == 0:
            continue

        # NMS
        keep_idx = nms(class_boxes, class_scores, iou_threshold)

        keep_boxes.append(class_boxes[keep_idx])
        keep_scores.append(class_scores[keep_idx])
        keep_labels.append(torch.full((len(keep_idx),), class_id, dtype=torch.long))

    if keep_boxes:
        boxes_out = torch.cat(keep_boxes)
        scores_out = torch.cat(keep_scores)
        labels_out = torch.cat(keep_labels)

        # Limit total detections
        if len(boxes_out) > max_detections:
            _, top_idx = scores_out.topk(max_detections)
            boxes_out = boxes_out[top_idx]
            scores_out = scores_out[top_idx]
            labels_out = labels_out[top_idx]

        return boxes_out, scores_out, labels_out
    else:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)

def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes."""
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

# ============================================================================
# OPENVINO EXPORT WITH mAP EVALUATION
# ============================================================================

class OpenVINOExporter:
    """Export models to OpenVINO IR format, benchmark performance, and calculate mAP"""

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
        input_size: int = 320,
        compress_to_fp16: bool = False
    ) -> Optional[str]:
        """Export PyTorch model to OpenVINO IR format."""
        if not HAS_OPENVINO:
            return None

        print(f"\nExporting model to OpenVINO IR: {output_path}")

        model = model.cpu()
        model.eval()

        dummy_input = torch.randn(1, 3, input_size, input_size)

        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            ov_model = ov.convert_model(
                model,
                example_input=dummy_input,
                input=[1, 3, input_size, input_size]
            )

            ov.save_model(ov_model, output_path, compress_to_fp16=compress_to_fp16)

            print(f"✓ Model exported to {output_path}")
            bin_path = output_path.replace('.xml', '.bin')
            print(f"  Weights saved to {bin_path}")

            return output_path

        except Exception as e:
            print(f"✗ Export failed: {e}")
            return None

    def run_inference(
        self,
        model_path: str,
        dataloader: DataLoader,
        anchors: torch.Tensor,
        input_size: int,
        device: str = 'CPU',
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on OpenVINO model and collect predictions.

        Returns:
            all_predictions: List of prediction dicts
            all_ground_truths: List of ground truth dicts
        """
        if not HAS_OPENVINO:
            return [], []

        # Load and compile model
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, device)

        # Get input/output info
        input_layer = compiled_model.input(0)
        output_cls = compiled_model.output(0)
        output_box = compiled_model.output(1)

        all_predictions = []
        all_ground_truths = []

        anchors_np = anchors.numpy()
        anchors_torch = anchors

        for images, targets in tqdm(dataloader, desc=f"Inference ({Path(model_path).stem})"):
            batch_size = images.size(0)

            for i in range(batch_size):
                image_np = images[i:i+1].numpy()
                image_id = targets[i]['image_id']

                # Run inference
                result = compiled_model([image_np])
                pred_cls = result[output_cls][0]  # [num_anchors, num_classes]
                pred_boxes = result[output_box][0]  # [num_anchors, 4]

                # Convert to torch for post-processing
                pred_cls_torch = torch.from_numpy(pred_cls)
                pred_boxes_torch = torch.from_numpy(pred_boxes)

                # Decode predictions
                max_scores = person_scores_from_logits(pred_cls_torch)
                boxes = decode_boxes(pred_boxes_torch, anchors_torch, input_size)

                pred_labels = torch.ones_like(max_scores, dtype=torch.long)

                # Apply NMS
                nms_boxes, nms_scores, nms_labels = apply_nms(
                    boxes, max_scores, pred_labels,
                    iou_threshold=nms_threshold,
                    score_threshold=score_threshold
                )

                # Store predictions
                for j in range(len(nms_boxes)):
                    all_predictions.append({
                        'image_id': image_id,
                        'class_id': nms_labels[j].item(),
                        'score': nms_scores[j].item(),
                        'box': nms_boxes[j].numpy()
                    })

                # Store ground truths
                gt_boxes = targets[i]['boxes'].numpy()
                gt_labels = targets[i]['labels'].numpy()

                for j in range(len(gt_boxes)):
                    all_ground_truths.append({
                        'image_id': image_id,
                        'class_id': int(gt_labels[j]),
                        'box': gt_boxes[j]
                    })

        return all_predictions, all_ground_truths

    def calculate_map(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_thresholds: List[float] = [0.5],
        verbose: bool = True
    ) -> Dict:
        """
        Calculate mAP from predictions and ground truths.

        Args:
            predictions: List of {'image_id', 'class_id', 'score', 'box'}
            ground_truths: List of {'image_id', 'class_id', 'box'}
            iou_thresholds: IoU thresholds for evaluation
            verbose: Print per-class results

        Returns:
            Dictionary with mAP metrics
        """
        if not predictions or not ground_truths:
            print("  No predictions or ground truths to evaluate")
            return {'mAP@0.50': 0.0}

        results = {}
        classes = sorted(set(gt['class_id'] for gt in ground_truths))

        for iou_thresh in iou_thresholds:
            aps = []

            for class_id in classes:
                if class_id == 0:  # Skip background
                    continue

                # Get predictions and GTs for this class
                class_preds = [p for p in predictions if p['class_id'] == class_id]
                class_gts = [g for g in ground_truths if g['class_id'] == class_id]

                if len(class_gts) == 0:
                    continue

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
                        fp[pred_idx] = 1
                        continue

                    img_gt_boxes = np.array(gt_by_image[pred_img_id])

                    # Compute IoU with all GTs in this image
                    ious = compute_iou_matrix(pred_box[None, :], img_gt_boxes)[0]

                    # Find best matching GT
                    best_iou_idx = np.argmax(ious)
                    best_iou = ious[best_iou_idx]

                    if best_iou >= iou_thresh and not gt_matched[pred_img_id][best_iou_idx]:
                        tp[pred_idx] = 1
                        gt_matched[pred_img_id][best_iou_idx] = True
                    else:
                        fp[pred_idx] = 1

                # Calculate precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                recalls = tp_cumsum / len(class_gts)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

                # Calculate AP
                ap = calculate_ap_voc(recalls, precisions)
                aps.append(ap)

                if verbose:
                    print(f"    Class {class_id}: AP@{iou_thresh:.2f} = {ap:.4f} "
                          f"(TP={int(tp.sum())}, FP={int(fp.sum())}, GT={len(class_gts)}, "
                          f"Preds={len(class_preds)})")

            mean_ap = np.mean(aps) if aps else 0.0
            results[f'mAP@{iou_thresh:.2f}'] = mean_ap

        # COCO-style mAP (average over IoU thresholds)
        if len(iou_thresholds) > 1:
            results['mAP@0.50:0.95'] = np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])

        return results

    def benchmark_model(
        self,
        model_path: str,
        device: str = 'CPU',
        duration_seconds: int = 15,
        num_threads: int = None
    ) -> Dict:
        """Benchmark OpenVINO model performance."""
        if not HAS_OPENVINO:
            return {}

        print(f"\n  Benchmarking latency on {device}...")

        try:
            model = self.core.read_model(model_path)

            config = {}
            if num_threads is not None:
                config["INFERENCE_NUM_THREADS"] = str(num_threads)
                config["NUM_STREAMS"] = "1"

            compiled_model = self.core.compile_model(model, device, config)

            input_layer = compiled_model.input(0)
            input_shape = input_layer.shape
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            # Warmup
            for _ in range(10):
                compiled_model([dummy_input])

            # Benchmark
            start_time = time.time()
            iterations = 0

            while time.time() - start_time < duration_seconds:
                compiled_model([dummy_input])
                iterations += 1

            elapsed_time = time.time() - start_time
            fps = iterations / elapsed_time
            latency_ms = (elapsed_time / iterations) * 1000

            results = {
                'fps': fps,
                'latency_ms': latency_ms,
                'iterations': iterations,
                'device': device,
                'threads': num_threads or 'auto'
            }

            print(f"    Threads: {num_threads or 'auto'}")
            print(f"    Throughput: {fps:.2f} FPS")
            print(f"    Latency: {latency_ms:.2f} ms")

            return results

        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
            return {}

    def evaluate_model(
        self,
        model_path: str,
        dataloader: DataLoader,
        anchors: torch.Tensor,
        input_size: int,
        device: str = 'CPU',
        iou_thresholds: List[float] = [0.5]
    ) -> Dict:
        """
        Run full evaluation: inference + mAP calculation.

        Args:
            model_path: Path to OpenVINO .xml file
            dataloader: Validation dataloader
            anchors: Anchor boxes
            input_size: Model input size
            device: OpenVINO device
            iou_thresholds: IoU thresholds for mAP

        Returns:
            Dictionary with mAP results
        """
        print(f"\n  Running evaluation on {Path(model_path).stem}...")

        # Run inference
        predictions, ground_truths = self.run_inference(
            model_path, dataloader, anchors, input_size, device
        )

        print(f"    Collected {len(predictions)} predictions, {len(ground_truths)} ground truths")

        # Calculate mAP
        map_results = self.calculate_map(predictions, ground_truths, iou_thresholds)

        return map_results

    def compare_models(
        self,
        fp32_path: str,
        int8_path: str,
        dataloader: DataLoader = None,
        anchors: torch.Tensor = None,
        input_size: int = 320,
        device: str = 'CPU',
        thread_counts: List[int] = None,
        evaluate_accuracy: bool = True,
        iou_thresholds: List[float] = None
    ) -> Dict:
        """
        Compare FP32 and INT8 models for both speed and accuracy.

        Args:
            fp32_path: Path to FP32 OpenVINO model
            int8_path: Path to INT8 OpenVINO model
            dataloader: Validation dataloader (required if evaluate_accuracy=True)
            anchors: Anchor boxes (required if evaluate_accuracy=True)
            input_size: Model input size
            device: OpenVINO device
            thread_counts: List of thread counts to benchmark
            evaluate_accuracy: Whether to calculate mAP
            iou_thresholds: IoU thresholds for mAP (default: [0.5] and COCO range)

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 70)
        print("Model Performance & Accuracy Comparison")
        print("=" * 70)

        if thread_counts is None:
            thread_counts = [None]

        if iou_thresholds is None:
            iou_thresholds = [0.5]  # Just mAP@0.50 by default

        results = {
            'fp32': {'speed': [], 'accuracy': {}},
            'int8': {'speed': [], 'accuracy': {}},
            'comparison': {}
        }

        # ====================================================================
        # Speed Benchmarking
        # ====================================================================
        print("\n" + "-" * 70)
        print("SPEED BENCHMARKING")
        print("-" * 70)

        for threads in thread_counts:
            thread_label = threads or 'auto'
            print(f"\n[Threads: {thread_label}]")

            print("\n  FP32 Model:")
            fp32_speed = self.benchmark_model(fp32_path, device, num_threads=threads)
            results['fp32']['speed'].append({'threads': thread_label, **fp32_speed})

            print("\n  INT8 Model:")
            int8_speed = self.benchmark_model(int8_path, device, num_threads=threads)
            results['int8']['speed'].append({'threads': thread_label, **int8_speed})

            if fp32_speed and int8_speed:
                speedup = int8_speed['fps'] / fp32_speed['fps']
                print(f"\n  → INT8 Speedup: {speedup:.2f}x")

        # ====================================================================
        # Accuracy Evaluation (mAP)
        # ====================================================================
        if evaluate_accuracy and dataloader is not None and anchors is not None:
            print("\n" + "-" * 70)
            print("ACCURACY EVALUATION (mAP)")
            print("-" * 70)

            # Evaluate FP32
            print("\n  FP32 Model:")
            fp32_map = self.evaluate_model(
                fp32_path, dataloader, anchors, input_size, device, iou_thresholds
            )
            results['fp32']['accuracy'] = fp32_map

            for key, value in fp32_map.items():
                print(f"    {key}: {value:.4f}")

            # Evaluate INT8
            print("\n  INT8 Model:")
            int8_map = self.evaluate_model(
                int8_path, dataloader, anchors, input_size, device, iou_thresholds
            )
            results['int8']['accuracy'] = int8_map

            for key, value in int8_map.items():
                print(f"    {key}: {value:.4f}")

            # Calculate accuracy drop
            print("\n  Accuracy Comparison:")
            for key in fp32_map.keys():
                if key in int8_map:
                    drop = fp32_map[key] - int8_map[key]
                    drop_pct = (drop / fp32_map[key] * 100) if fp32_map[key] > 0 else 0
                    results['comparison'][f'{key}_drop'] = drop
                    results['comparison'][f'{key}_drop_pct'] = drop_pct
                    print(f"    {key}: FP32={fp32_map[key]:.4f} → INT8={int8_map[key]:.4f} "
                          f"(Δ={drop:+.4f}, {drop_pct:+.2f}%)")

        # ====================================================================
        # Summary Table
        # ====================================================================
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Speed summary
        print("\nSpeed (best thread configuration):")
        if results['fp32']['speed'] and results['int8']['speed']:
            best_fp32 = max(results['fp32']['speed'], key=lambda x: x.get('fps', 0))
            best_int8 = max(results['int8']['speed'], key=lambda x: x.get('fps', 0))
            speedup = best_int8['fps'] / best_fp32['fps'] if best_fp32['fps'] > 0 else 0

            print(f"  FP32: {best_fp32['fps']:.2f} FPS ({best_fp32['latency_ms']:.2f} ms) @ {best_fp32['threads']} threads")
            print(f"  INT8: {best_int8['fps']:.2f} FPS ({best_int8['latency_ms']:.2f} ms) @ {best_int8['threads']} threads")
            print(f"  Speedup: {speedup:.2f}x")

            results['comparison']['best_speedup'] = speedup

        # Accuracy summary
        if evaluate_accuracy and results['fp32']['accuracy']:
            print("\nAccuracy:")
            main_metric = 'mAP@0.50'
            if main_metric in results['fp32']['accuracy']:
                fp32_map = results['fp32']['accuracy'][main_metric]
                int8_map = results['int8']['accuracy'].get(main_metric, 0)
                drop = fp32_map - int8_map

                print(f"  FP32 {main_metric}: {fp32_map:.4f}")
                print(f"  INT8 {main_metric}: {int8_map:.4f}")
                print(f"  Accuracy Drop: {drop:+.4f} ({drop/fp32_map*100:+.2f}%)" if fp32_map > 0 else "  N/A")

        print("\n" + "=" * 70)

        return results

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

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

    loss_meter = AverageMeter('Loss')
    cls_meter = AverageMeter('Cls')
    loc_meter = AverageMeter('Loc')

    start_time = time.time()

    # Wrap dataloader with tqdm - creates new progress bar each epoch
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch+1}", ncols=100)

    for batch_idx, (images, targets) in pbar:
        images = images.to(device)

        if config.use_amp and scaler is not None:
            with autocast(device_type=config.device):
                pred_cls, pred_boxes = model(images)
                loss, loss_dict = criterion(pred_cls, pred_boxes, targets, anchors)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_cls, pred_boxes = model(images)
            loss, loss_dict = criterion(pred_cls, pred_boxes, targets, anchors)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

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

    elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1} completed in {elapsed:.1f}s | "
          f"Avg Loss: {loss_meter.avg:.4f} | "
          f"Cls: {cls_meter.avg:.4f} | "
          f"Loc: {loc_meter.avg:.4f}\n")

    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_meter.avg,
        'loc_loss': loc_meter.avg
    }

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SSDLoss,
    anchors: torch.Tensor,
    device: str
) -> Dict:
    """Validation pass"""
    model.eval()

    loss_meter = AverageMeter('Loss')

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            pred_cls, pred_boxes = model(images)
            loss, _ = criterion(pred_cls, pred_boxes, targets, anchors)

            loss_meter.update(loss.item())

    return {'loss': loss_meter.avg}

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class DetectorTrainingPipeline:
    """Own the canonical train/validate/checkpoint/export workflow."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    @classmethod
    def from_environment(cls) -> "DetectorTrainingPipeline":
        return cls(TrainingConfig(
            azurite_endpoint=os.getenv("AZURITE_BLOB_ENDPOINT", "http://127.0.0.1:10000/devstoreaccount1"),
            azurite_access_key=os.getenv("AZURITE_ACCOUNT_NAME", "devstoreaccount1"),
            azurite_secret_key=os.getenv("AZURITE_ACCOUNT_KEY", ""),
            azurite_connection_string=os.getenv("AZURITE_CONNECTION_STRING", ""),
            azurite_data_bucket=os.getenv("AZURITE_DATA_CONTAINER", "computer-vision-data"),
            azurite_model_bucket=os.getenv("AZURITE_MODEL_CONTAINER", "computer-vision-models"),
            use_azurite=os.getenv("USE_AZURITE", "true").lower() == "true",
            data_root=os.getenv("DATA_ROOT", "./data"),
            enable_quantization=os.getenv("ENABLE_QUANTIZATION", "false").lower() == "true",
        ))

    def run(self) -> None:
        config = self.config
        if config.enable_quantization:
            raise RuntimeError(
                "The legacy in-training QAT path is retired. Train FP32 with "
                "ENABLE_QUANTIZATION=false, then run python -m person_detection.optimization.pipeline for manifest-driven "
                "accuracy-controlled INT8 calibration."
            )

        print("=" * 60)
        print("SSD Person Detection Training Pipeline")
        print("ATSS + quality-aware binary classification")
        print("=" * 60)
        print(f"Device: {config.device}")
        print(f"Input size: {config.input_size}")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Use AMP: {config.use_amp}")
        print(f"Use Focal Loss: {config.use_focal_loss}")
        print(f"Enable Quantization: {config.enable_quantization}")
        if config.use_azurite:
            print(f"Azurite endpoint: {config.azurite_endpoint}")
        else:
            print(f"Local data: {config.data_root}")
        print("=" * 60)

        # Initialize components
        azurite_client = AzuriteClient(config)
        exporter = OpenVINOExporter(config)

        # Check if FP32 checkpoint already exists
        checkpoint_path = 'best_model_fp32.pth'
        skip_fp32_training = False
        if os.path.exists(checkpoint_path):
            existing_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            skip_fp32_training = existing_checkpoint.get("modelFormatVersion") == MODEL_FORMAT_VERSION

        if skip_fp32_training:
            print("\n" + "=" * 60)
            print("Found compatible FP32 checkpoint - skipping FP32 training")
            print("=" * 60)

        elif os.path.exists(checkpoint_path):
            print("\n" + "=" * 60)
            print("Existing checkpoint uses the legacy softmax head; retraining the quality head")
            print("=" * 60)
        # Create datasets
        print("\nLoading datasets...")
        train_dataset = CanonicalPersonDetectionDataset(
            azurite_client,
            split='train',
            input_size=config.input_size,
            augment=True
        )

        val_dataset = CanonicalPersonDetectionDataset(
            azurite_client,
            split='val',
            input_size=config.input_size,
            augment=False
        )

        if skip_fp32_training:
            checkpoint_dataset = existing_checkpoint.get("dataset")
            expected_dataset = train_dataset.dataset_metadata
            provenance_keys = ("versionPrefix", "manifestSha256", "schemaVersion")
            mismatches = {
                key: {
                    "checkpoint": checkpoint_dataset.get(key) if isinstance(checkpoint_dataset, dict) else None,
                    "current": expected_dataset.get(key),
                }
                for key in provenance_keys
                if not isinstance(checkpoint_dataset, dict)
                or checkpoint_dataset.get(key) != expected_dataset.get(key)
            }
            if mismatches:
                skip_fp32_training = False
                print("Existing checkpoint dataset provenance differs; retraining")
                print(f"  Mismatches: {mismatches}")

        # Validate dataset structure
        print("\nValidating dataset structure...")
        train_dataset.validate_samples(num_samples=3)
        val_dataset.validate_samples(num_samples=3)

        # Create dataloaders
        train_sampler = None
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
            collate_fn=collate_fn,
            pin_memory=True if config.device == 'cuda' else False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if config.device == 'cuda' else False,
        )

        # Create model
        print("\nInitializing model...")
        model = SSDPersonDetector(
            num_classes=config.num_classes,
            input_size=config.input_size
        )
        model = model.to(config.device)

        # Get anchors
        anchors = model.anchor_generator.get_anchors()

        model_summary(
            model,
            model_type=ModelType.SINGLE_INPUT,
            input1_size=(config.batch_size, 3, config.input_size, config.input_size)
        )

        # ========================================================================
        # PHASE 1: FP32 Training (skip if checkpoint exists)
        # ========================================================================
        if not skip_fp32_training:
            print("\n" + "=" * 60)
            print("PHASE 1: FP32 Training")
            print("=" * 60)

            # Loss function
            criterion = SSDLoss(
                num_classes=config.num_classes,
                neg_pos_ratio=config.neg_pos_ratio,
                use_focal_loss=config.use_focal_loss,
                focal_alpha=config.focal_alpha,
                focal_gamma=config.focal_gamma,
                input_size=config.input_size,
                anchors_per_level=model.anchor_generator.num_anchors_per_level,
                atss_topk=config.atss_topk,
                giou_weight=config.giou_weight,
            )

            # Optimizer
            if config.use_varied_lr:
                # optimizer = optim.AdamW(
                #     [
                #         {'params': model.features.parameters(), 'lr': config.learning_rate * 0.1},      # Backbone frozen-ish
                #         {'params': model.fpn.parameters(), 'lr': config.learning_rate},           # FPN
                #         {'params': model.detection_heads.parameters(), 'lr': config.learning_rate}, # Heads
                #     ],
                #     weight_decay=config.weight_decay,  # AdamW handles this properly
                #     betas=(0.9, 0.999)
                # )
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': config.learning_rate * 0.1},      # Backbone frozen-ish
                    {'params': model.fpn.parameters(), 'lr': config.learning_rate},           # FPN
                    {'params': model.detection_heads.parameters(), 'lr': config.learning_rate}, # Heads
                ],
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
            else:
                # optimizer = optim.AdamW(
                #     model.parameters(),
                #     lr=config.learning_rate,  # Lower than SGD (start here)
                #     weight_decay=config.weight_decay,  # AdamW handles this properly
                #     betas=(0.9, 0.999)
                # )
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay
                )

            # Schedulers
            warmup = WarmupScheduler(optimizer, config.warmup_epochs, config.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs - config.warmup_epochs,
                eta_min=config.learning_rate * 0.1
            )

            # Mixed precision scaler
            scaler = GradScaler(device=config.device)

            best_val_loss = float('inf')

            for epoch in range(config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
                print("-" * 40)

                is_warmup = warmup.step(epoch)

                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, criterion,
                    anchors, config.device, epoch, config, scaler
                )

                val_metrics = validate(
                    model, val_loader, criterion,
                    anchors, config.device
                )

                if not is_warmup:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"LR: {current_lr:.6f}")

                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': config.__dict__,
                        'dataset': train_dataset.dataset_metadata,
                        'modelFormatVersion': MODEL_FORMAT_VERSION,
                        'assignment': {'name': 'ATSS', 'topKPerLevel': config.atss_topk},
                        'classificationHead': {'name': 'binary-quality-v1', 'target': 'predictedIoU'},
                    }

                    torch.save(checkpoint, checkpoint_path)

                    if config.use_azurite:
                        azurite_client.put_object(
                            config.azurite_model_bucket,
                            'person_detector_ssd/best_model_fp32.pth',
                            checkpoint_path
                        )

                    print(f"✓ Saved best FP32 model (loss: {best_val_loss:.4f})")

            print("\n✓ FP32 training complete!")

        # ========================================================================
        # Load best FP32 checkpoint before export/quantization
        # ========================================================================
        print("\n" + "=" * 60)
        print("Loading best FP32 checkpoint...")
        print("=" * 60)

        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        load_detector_state_dict(model, checkpoint['model_state_dict'])

        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
        checkpoint_epoch = checkpoint.get('epoch', 'N/A')
        print(f"✓ Loaded checkpoint from epoch {checkpoint_epoch} (val_loss: {best_val_loss})")

        # ========================================================================
        # PHASE 2: Export FP32 to OpenVINO
        # ========================================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Export FP32 Model to OpenVINO")
        print("=" * 60)

        exporter.export_to_openvino(
            model,
            'person_detector_fp32.xml',
            input_size=config.input_size,
            compress_to_fp16=False
        )

        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        print("\nOutput files:")
        print("  - best_model_fp32.pth (PyTorch FP32)")
        print("  - person_detector_fp32.xml/bin (OpenVINO FP32)")


def main() -> None:
    DetectorTrainingPipeline.from_environment().run()


if __name__ == "__main__":
    main()
