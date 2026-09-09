"""Canonical CityPersons dataset with geometry-consistent targets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from person_detection.data.annotations import parse_canonical_annotation
from person_detection.data.layout import (
    TRAINABLE_LABEL_STATUSES,
    load_citypersons_manifest,
    load_citypersons_split,
    resolve_citypersons_prefix,
    version_blob,
)
from person_detection.data.sampling import (
    DEFAULT_HARD_CASE_POLICY,
    GreedyStratifiedSelector,
    HardCaseSamplingPolicy,
)


try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
if HAS_ALBUMENTATIONS:
    class BoxPreservingCoarseDropout(A.CoarseDropout):
        """Add synthetic occlusion without deleting or shrinking detection targets."""

        def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
            return bboxes




IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PRODUCTION_PREPROCESSING = {
    "name": "citypersons-letterbox-imagenet-v1",
    "colorSpace": "RGB",
    "resize": "longest-side",
    "padding": "symmetric-zero",
    "normalization": {
        "scale": 255.0,
        "mean": IMAGENET_MEAN.tolist(),
        "std": IMAGENET_STD.tolist(),
    },
    "layout": "NCHW",
    "dtype": "float32",
}

DEFAULT_OVERSAMPLING_POLICY = DEFAULT_HARD_CASE_POLICY


def letterbox_image(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize and symmetric padding to a square canvas."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def map_letterbox_box(box: list[float], scale: float, pad_x: int, pad_y: int) -> list[float]:
    return [
        box[0] * scale + pad_x, box[1] * scale + pad_y,
        box[2] * scale + pad_x, box[3] * scale + pad_y,
    ]

def clip_box_to_image(box: list[float], width: int, height: int) -> list[float] | None:
    """Clip a raw CityPersons box at the model-transform boundary."""
    clipped = [
        max(float(box[0]), 0.0), max(float(box[1]), 0.0),
        min(float(box[2]), float(width)), min(float(box[3]), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def preprocess_rgb_image(
    image: np.ndarray,
    size: int,
    *,
    add_batch: bool = False,
    return_geometry: bool = False,
):
    """Apply the production letterbox/normalization contract to an RGB image."""
    image, scale, pad_x, pad_y = letterbox_image(image, size)
    image = image.astype(np.float32) / PRODUCTION_PREPROCESSING["normalization"]["scale"]
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    tensor = tensor[None, ...] if add_batch else tensor
    if return_geometry:
        return tensor, scale, pad_x, pad_y
    return tensor


def size_slice(box: list[float]) -> str:
    """Return a stable object-height slice in source-image pixels."""
    height = float(box[3] - box[1])
    if height < 50:
        return "small"
    if height < 100:
        return "medium"
    return "large"


def annotation_strata(annotation, *, status: str, image_blob: str) -> set[str]:
    """Describe an immutable record for sampling and calibration reports."""
    strata = {f"status:{status}", f"city:{Path(image_blob).parent.name}"}
    for obj in annotation.objects:
        strata.add(f"size:{size_slice(obj.full_box)}")
        strata.add(f"sourceLabel:{obj.source_label}")
        visibility = obj.attributes.get("visibility")
        posture = obj.attributes.get("posture")
        if isinstance(visibility, str):
            strata.add(f"visibility:{visibility}")
        if isinstance(posture, str):
            strata.add(f"posture:{posture}")
    return strata


def select_stratified_indices(
    strata_by_index: list[set[str]], sample_count: int, seed: int = 1337
) -> list[int]:
    """Compatibility wrapper around the configurable calibration strategy."""
    return GreedyStratifiedSelector().select(strata_by_index, sample_count, seed)


class CanonicalPersonDetectionDataset(Dataset):
    """Read image names and separate canonical JSON annotations from a split manifest."""

    def __init__(self, azurite_client: Any, split: str = "train", input_size: int = 320,
                 augment: bool = True):
        self.azurite = azurite_client
        self.config = azurite_client.config
        self.split = split
        self.input_size = input_size
        self.augment = augment and split == "train"
        self._annotation_cache = {}
        self._strata_cache = {}
        bucket = self.config.azurite_data_bucket
        read_blob = lambda name: self.azurite.get_object_bytes(bucket, name)
        self.version_prefix = resolve_citypersons_prefix(read_blob)
        manifest, manifest_checksum = load_citypersons_manifest(read_blob, self.version_prefix)
        self.dataset_metadata = {
            "dataset": "citypersons",
            "versionPrefix": self.version_prefix,
            "manifestSha256": manifest_checksum,
            "schemaVersion": manifest.get("schemaVersion"),
            "samplingPolicy": "natural" if split != "train" else "natural-v1",
        }
        self.samples = self._load_samples(read_blob)
        self._setup_transform()
        self._report()

    def _load_samples(self, read_blob):
        samples = []
        for manifest_index, record in enumerate(load_citypersons_split(read_blob, self.version_prefix, self.split)):
            status = record.get("labelStatus")
            if status not in TRAINABLE_LABEL_STATUSES:
                continue
            image, annotation = record.get("image"), record.get("annotation")
            checksums = record.get("checksums")
            if not isinstance(image, str) or not isinstance(annotation, str) or not isinstance(checksums, dict):
                raise RuntimeError(f"Invalid canonical record in {self.split} split")
            image_sha256 = checksums.get("image")
            annotation_sha256 = checksums.get("annotation")
            person_count, ignored_count = record.get("personCount"), record.get("ignoredCount")
            if (not isinstance(image_sha256, str) or len(image_sha256) != 64 or
                    not isinstance(annotation_sha256, str) or len(annotation_sha256) != 64 or
                    not isinstance(person_count, int) or person_count < 0 or
                    not isinstance(ignored_count, int) or ignored_count < 0):
                raise RuntimeError(f"Invalid checksums or counts in {self.split} split record")
            samples.append({
                "manifest_index": manifest_index,
                "image": version_blob(self.version_prefix, image),
                "image_relative": image,
                "annotation": version_blob(self.version_prefix, annotation),
                "annotation_relative": annotation,
                "image_sha256": image_sha256,
                "annotation_sha256": annotation_sha256,
                "person_count": person_count,
                "ignored_count": ignored_count,
                "status": status,
            })
        return sorted(samples, key=lambda sample: sample["image"])

    def _setup_transform(self):
        if not HAS_ALBUMENTATIONS:
            self.transform = None
            return
        geometry = [
            A.HorizontalFlip(p=0.5),
            A.Perspective(scale=(0.02, 0.06), p=0.2),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.08, 0.03)},
                     scale=(0.9, 1.1), rotate=(-5, 5), p=0.3),
        ] if self.augment else []
        appearance = [
            BoxPreservingCoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.02, 0.08),
                            hole_width_range=(0.02, 0.08), fill=0, p=0.3),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.CLAHE(clip_limit=2.0), A.ToGray(p=0.15),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3), A.MotionBlur(blur_limit=3),
                A.GaussNoise(std_range=(0.04, 0.20)),
                A.ImageCompression(quality_range=(65, 95)),
            ], p=0.2),
        ] if self.augment else []
        self.transform = A.Compose(
            geometry + [
                A.LongestMaxSize(max_size=self.input_size),
                A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size,
                              border_mode=cv2.BORDER_CONSTANT, fill=0),
            ] + appearance + [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_kinds", "bbox_indices"],
                                     min_area=1.0, min_visibility=0.0, clip=True),
        )

    def _report(self):
        cities = {Path(sample["image"]).parent.name for sample in self.samples}
        print(f"Found {len(self.samples)} canonical images for {self.split} split")
        print(f"  Dataset: {self.version_prefix} ({self.dataset_metadata['manifestSha256'][:12]}…)")
        print(f"  Cities: {len(cities)} folders")

    def validate_samples(self, num_samples: int = 5) -> bool:
        if not self.samples:
            print("✗ No samples to validate")
            return False
        for sample in random.sample(self.samples, min(num_samples, len(self.samples))):
            if self.azurite.get_object_bytes(self.config.azurite_data_bucket, sample["image"]) is None:
                raise RuntimeError(f"Image not found: {sample['image']}")
            self._load_annotation(sample)
        print(f"  Validation: {min(num_samples, len(self.samples))} canonical samples accessible")
        return True

    def _load_annotation(self, sample):
        key = sample["annotation"]
        if key not in self._annotation_cache:
            data = self.azurite.get_object_bytes(self.config.azurite_data_bucket, key)
            self._annotation_cache[key] = parse_canonical_annotation(
                data, expected_image_blob=sample["image_relative"],
                expected_image_sha256=sample["image_sha256"], expected_status=sample["status"],
                expected_sidecar_sha256=sample["annotation_sha256"],
                expected_person_count=sample["person_count"],
                expected_ignored_count=sample["ignored_count"],
            )
        return self._annotation_cache[key]

    def strata_for_sample(self, index: int) -> set[str]:
        """Return cached sampling strata derived from the canonical sidecar."""
        if index not in self._strata_cache:
            sample = self.samples[index]
            self._strata_cache[index] = annotation_strata(
                self._load_annotation(sample),
                status=sample["status"],
                image_blob=sample["image_relative"],
            )
        return set(self._strata_cache[index])

    def build_sampling_weights(self, policy: dict[str, Any] | None = None) -> torch.Tensor:
        """Build training-only weights without changing validation distribution."""
        if self.split != "train":
            raise ValueError("Oversampling is only valid for the training split")
        definition = DEFAULT_OVERSAMPLING_POLICY if policy is None else policy
        sampling_policy = HardCaseSamplingPolicy.from_mapping(definition)
        weights = [
            sampling_policy.weight(self.strata_for_sample(index))
            for index in range(len(self.samples))
        ]
        self.dataset_metadata["samplingPolicy"] = sampling_policy.to_mapping()
        return torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _tensor_boxes(boxes):
        return torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_data = self.azurite.get_object_bytes(self.config.azurite_data_bucket, sample["image"])
        if image_data is None:
            raise RuntimeError(f"Image not found: {sample['image']}")
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Image cannot be decoded: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = self._load_annotation(sample)
        if image.shape[1] != annotation.width or image.shape[0] != annotation.height:
            raise ValueError(f"Decoded image dimensions do not match sidecar: {sample['image']}")

        boxes, kinds, indices = [], [], []
        for item_index, obj in enumerate(annotation.objects):
            full_box = clip_box_to_image(obj.full_box, annotation.width, annotation.height)
            if full_box is None:
                raise ValueError(f"Person box does not intersect its image: {obj.object_id}")
            boxes.append(full_box)
            kinds.append(0)
            indices.append(item_index)
            visible_box = clip_box_to_image(
                obj.visible_box, annotation.width, annotation.height
            )
            if visible_box is not None:
                boxes.append(visible_box)
                kinds.append(1)
                indices.append(item_index)
        for item_index, box in enumerate(annotation.ignore_regions):
            clipped = clip_box_to_image(box, annotation.width, annotation.height)
            if clipped is not None:
                boxes.append(clipped)
                kinds.append(2)
                indices.append(item_index)

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, bbox_kinds=kinds, bbox_indices=indices)
            image = transformed["image"]
            transformed_items = zip(transformed["bboxes"], transformed["bbox_kinds"], transformed["bbox_indices"])
        else:
            image, scale, pad_x, pad_y = letterbox_image(image, self.input_size)
            transformed_items = ((map_letterbox_box(box, scale, pad_x, pad_y), kind, item_index)
                                 for box, kind, item_index in zip(boxes, kinds, indices))
            image = torch.from_numpy(
                ((image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
            ).permute(2, 0, 1).float()

        full_by_index, visible_by_index, ignore_boxes = {}, {}, []
        for box, kind, item_index in transformed_items:
            kind, item_index = int(kind), int(item_index)
            if kind == 0:
                full_by_index[item_index] = list(box)
            elif kind == 1:
                visible_by_index[item_index] = list(box)
            else:
                ignore_boxes.append(list(box))
        kept_indices = sorted(full_by_index)
        objects = [annotation.objects[item_index] for item_index in kept_indices]
        target = {
            "boxes": self._tensor_boxes([full_by_index[item_index] for item_index in kept_indices]),
            "labels": torch.ones(len(kept_indices), dtype=torch.int64),
            "visible_boxes": self._tensor_boxes(
                [visible_by_index[item_index] for item_index in kept_indices if item_index in visible_by_index]
            ),
            "visible_box_indices": torch.as_tensor(
                [position for position, item_index in enumerate(kept_indices) if item_index in visible_by_index],
                dtype=torch.int64,
            ),
            "ignore_regions": self._tensor_boxes(ignore_boxes),
            "source_labels": [obj.source_label for obj in objects],
            "source_class_ids": torch.as_tensor([obj.source_class_id for obj in objects], dtype=torch.int64),
            "attributes": [obj.attributes for obj in objects],
            "object_ids": [obj.object_id for obj in objects],
            "image_blob": sample["image_relative"],
            "image_id": index,
        }
        return image, target
