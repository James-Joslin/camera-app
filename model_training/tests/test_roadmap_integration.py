"""Focused regression tests for the detector roadmap implementation."""

import math
import warnings
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from person_detection.data.annotations import CanonicalAnnotation, CanonicalObject
import person_detection.data.dataset as canonical_data
from person_detection.data.dataset import (
    CanonicalPersonDetectionDataset,
    clip_box_to_image,
    preprocess_rgb_image,
    select_stratified_indices,
)
from person_detection.evaluation.citypersons import (
    OfficialCityPersonsAccumulator,
    run_official_citypersons_evaluator,
)
from person_detection.evaluation.inference import MAPCalculator
from person_detection.training.pipeline import (
    AttentionDetectionHead,
    PersonAnchorGenerator,
    QualityFocalLoss,
    SSDLoss,
    TrainingConfig,
    checkpoint_resume_mismatches,
    load_detector_state_dict,
    make_training_checkpoint,
)


class SamplingAndSliceTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_covers_rare_records(self):
        strata = [
            {"status:positive", "city:a", "size:large"},
            {"status:positive", "city:a", "size:large"},
            {
                "status:positive",
                "city:b",
                "size:small",
                "visibility:heavily_occluded",
            },
            {"status:verified_negative", "city:a"},
        ]
        first = select_stratified_indices(strata, 2, seed=17)
        second = select_stratified_indices(strata, 2, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {2, 3})

    def test_training_oversampling_weights_hard_cases_only(self):
        dataset = object.__new__(CanonicalPersonDetectionDataset)
        dataset.split = "train"
        dataset.samples = [{}, {}]
        dataset.dataset_metadata = {}
        strata = [
            {"status:positive", "size:large", "visibility:clear"},
            {"status:positive", "size:small", "visibility:heavily_occluded"},
        ]
        dataset.strata_for_sample = lambda index: strata[index]
        weights = dataset.build_sampling_weights()
        self.assertEqual(weights.tolist(), [1.0, 2.5])
        self.assertEqual(
            dataset.dataset_metadata["samplingPolicy"]["name"],
            "citypersons-hard-cases-v1",
        )

    def test_production_preprocessing_returns_letterbox_geometry(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, scale, pad_x, pad_y = preprocess_rgb_image(
            image, 90, 160, add_batch=True, return_geometry=True
        )
        self.assertEqual(tensor.shape, (1, 3, 90, 160))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual((scale, pad_x, pad_y), (0.8, 0, 5))
        self.assertEqual(
            clip_box_to_image([-10, 20, 20, 60], 100, 80), [0.0, 20.0, 20.0, 60.0]
        )

    @unittest.skipUnless(
        canonical_data.HAS_ALBUMENTATIONS, "Albumentations is not installed"
    )
    def test_coarse_dropout_preserves_tiny_boxes_without_runtime_warning(self):
        transform = canonical_data.A.Compose(
            [
                canonical_data.BoxPreservingCoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(0.5, 0.5),
                    hole_width_range=(0.5, 0.5),
                    fill=0,
                    p=1.0,
                )
            ],
            bbox_params=canonical_data.A.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_kinds", "bbox_indices"],
                min_area=0.0,
                min_visibility=0.0,
                clip=True,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            transformed = transform(
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                bboxes=[[1.0, 1.0, 1.2, 2.0]],
                bbox_kinds=[0],
                bbox_indices=[0],
            )
        self.assertEqual(len(transformed["bboxes"]), 1)


    def test_metric_slices_neutralize_out_of_slice_people(self):
        calculator = MAPCalculator([0.5])
        calculator.add_image(0)
        calculator.add_ground_truths(
            0,
            [[0, 0, 10, 40]],
            [1],
            [{"size": "small", "sourceLabel": "pedestrian", "visibility": "clear"}],
        )
        calculator.add_predictions(
            0, [{"class": 1, "score": 0.9, "box": [0, 0, 10, 40]}]
        )
        calculator.add_image(1)
        calculator.add_ground_truths(
            1,
            [[50, 0, 80, 120]],
            [1],
            [{"size": "large", "sourceLabel": "rider", "visibility": "occluded"}],
        )
        slices = calculator.compute_slices()
        self.assertAlmostEqual(slices["size/small"]["mAP@0.50"], 1.0)
        self.assertAlmostEqual(slices["size/large"]["mAP@0.50"], 0.0)
        self.assertIn("sourceLabel/rider", slices)
        self.assertIn("visibility/occluded", slices)


class AssignmentAndHeadTests(unittest.TestCase):
    def test_person_anchors_are_tall_at_fine_levels(self):
        generator = PersonAnchorGenerator(input_height=72, input_width=128)
        first_anchor = generator.get_anchors()[0]
        physical_ratio = float(first_anchor[2] * 128 / (first_anchor[3] * 72))
        self.assertAlmostEqual(physical_ratio, 0.15, places=5)

    def test_rectangular_anchors_use_actual_feature_map_height_and_width(self):
        generator = PersonAnchorGenerator(input_height=360, input_width=640)
        self.assertEqual(
            generator.feature_map_shapes,
            [(45, 80), (23, 40), (12, 20), (6, 10), (3, 5)],
        )
        self.assertEqual(tuple(generator.get_anchors().shape), (29235, 4))

    def test_atss_selects_adaptive_positive_and_respects_ignore(self):
        criterion = SSDLoss(
            input_height=100, input_width=100, anchors_per_level=[2, 2], atss_topk=1
        )
        anchors = torch.tensor([
            [0.50, 0.50, 0.20, 0.60],
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.30, 0.60],
            [0.85, 0.85, 0.10, 0.10],
        ])
        _, labels, positives = criterion.match_anchors(
            torch.tensor([[40.0, 20.0, 60.0, 80.0]]),
            torch.ones(1, dtype=torch.long),
            anchors,
            ignore_boxes=torch.tensor([[80.0, 80.0, 90.0, 90.0]]),
        )
        self.assertTrue(positives[0])
        self.assertEqual(int(positives.sum()), 1)
        self.assertEqual(labels.tolist(), [1, 0, 0, -1])

    def test_quality_focal_target_rewards_localization_quality(self):
        criterion = QualityFocalLoss(alpha=0.4, beta=2.0)
        target = torch.tensor([0.8])
        calibrated = torch.tensor([math.log(0.8 / 0.2)])
        uncalibrated = torch.tensor([0.0])
        self.assertLess(
            float(criterion(calibrated, target)),
            float(criterion(uncalibrated, target)),
        )

    def test_giou_is_zero_for_identical_boxes(self):
        box = torch.tensor([[10.0, 10.0, 30.0, 50.0]])
        disjoint = torch.tensor([[40.0, 40.0, 60.0, 80.0]])
        self.assertAlmostEqual(float(SSDLoss.aligned_giou_loss(box, box)), 0.0)
        self.assertGreater(
            float(SSDLoss.aligned_giou_loss(box, disjoint)),
            1.0,
        )

    def test_binary_head_shape_and_prior(self):
        head = AttentionDetectionHead(32, num_anchors=3, num_classes=1)
        logits, boxes = head(torch.zeros(2, 32, 4, 4))
        self.assertEqual(tuple(logits.shape), (2, 48, 1))
        self.assertEqual(tuple(boxes.shape), (2, 48, 4))
        self.assertAlmostEqual(
            float(head.cls_conv.bias[0].sigmoid()), 0.01, places=5
        )

    def test_legacy_softmax_head_migrates_to_log_odds(self):
        class WrappedHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.cls_conv = nn.Conv2d(1, 1, 1)

        model = nn.Module()
        model.head = WrappedHead()
        legacy = {
            "head.cls_conv.weight": torch.tensor([[[[1.0]]], [[[4.0]]]]),
            "head.cls_conv.bias": torch.tensor([2.0, 7.0]),
        }
        with self.assertWarns(UserWarning):
            load_detector_state_dict(model, legacy)
        self.assertEqual(model.head.cls_conv.weight.flatten().tolist(), [3.0])
        self.assertEqual(model.head.cls_conv.bias.tolist(), [5.0])


class ResumeContractTests(unittest.TestCase):
    def test_complete_resume_contract_matches_rectangular_run(self):
        config = TrainingConfig(input_height=360, input_width=640)
        dataset = {
            "versionPrefix": "datasets/citypersons/v1",
            "manifestSha256": "a" * 64,
            "schemaVersion": 1,
        }
        anchors = {"featureMapShapes": [[45, 80]]}
        checkpoint = {
            "modelFormatVersion": 3,
            "config": {"input_height": 360, "input_width": 640, "num_classes": 1},
            "dataset": dataset,
            "anchors": anchors,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "completedEpochs": 9,
            "best_val_loss": 1.2,
        }
        self.assertEqual(
            checkpoint_resume_mismatches(checkpoint, config, dataset, anchors), {}
        )

    def test_resume_rejects_canvas_mismatch(self):
        config = TrainingConfig(input_height=360, input_width=640)
        checkpoint = {
            "modelFormatVersion": 3,
            "config": {"input_height": 480, "input_width": 480, "num_classes": 1},
            "dataset": {},
            "anchors": {},
        }
        mismatches = checkpoint_resume_mismatches(checkpoint, config, {}, {})
        self.assertIn("inputHeight", mismatches)
        self.assertIn("inputWidth", mismatches)
        self.assertIn("resumeState", mismatches)


    def test_resumable_checkpoint_round_trips_all_training_state_weights_only(self):
        config = TrainingConfig(input_height=72, input_width=128, device="cpu")
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        generator = PersonAnchorGenerator(input_height=72, input_width=128)
        checkpoint = make_training_checkpoint(
            epoch=2,
            training_complete=False,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=0.5,
            config=config,
            dataset_metadata={},
            anchor_generator=generator,
            sampling_generator=torch.Generator().manual_seed(4),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, target)
            restored = torch.load(target, map_location="cpu", weights_only=True)
        self.assertEqual(restored["completedEpochs"], 3)
        self.assertFalse(restored["trainingComplete"])
        for key in (
            "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
            "scaler_state_dict", "sampler_generator_state", "torch_rng_state",
        ):
            self.assertIn(key, restored)


class MixedPrecisionLossTests(unittest.TestCase):
    def test_bfloat16_quality_targets_accept_float32_iou(self):
        criterion = SSDLoss(
            input_height=100, input_width=100, anchors_per_level=[2, 2], atss_topk=1
        )
        anchors = torch.tensor([
            [0.50, 0.50, 0.20, 0.60],
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.30, 0.60],
            [0.85, 0.85, 0.10, 0.10],
        ])
        logits = torch.zeros(
            1, 4, 1, dtype=torch.bfloat16, requires_grad=True
        )
        offsets = torch.zeros(1, 4, 4, dtype=torch.float32, requires_grad=True)
        targets = [{
            "boxes": torch.tensor([[40.0, 20.0, 60.0, 80.0]]),
            "labels": torch.ones(1, dtype=torch.long),
            "ignore_regions": torch.empty((0, 4)),
        }]
        loss, metrics = criterion(logits, offsets, targets, anchors)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["num_pos"], 0)
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(offsets.grad)


class OfficialEvaluatorAdapterTests(unittest.TestCase):
    def test_project_person_classes_become_official_neutral_regions(self):
        annotation = CanonicalAnnotation(
            image_blob="images/val/city/frame.png",
            width=100,
            height=100,
            image_sha256="0" * 64,
            objects=[
                CanonicalObject(
                    "pedestrian",
                    [0, 0, 10, 50],
                    [0, 0, 10, 40],
                    1,
                    "pedestrian",
                    {"visibility": "clear"},
                ),
                CanonicalObject(
                    "rider",
                    [20, 0, 40, 60],
                    [20, 0, 40, 30],
                    2,
                    "rider",
                    {"visibility": "occluded"},
                ),
            ],
            ignore_regions=[[50, 0, 80, 80]],
        )
        accumulator = OfficialCityPersonsAccumulator()
        accumulator.add_image(
            7,
            annotation.image_blob,
            annotation,
            [{"box": [0, 0, 10, 50], "score": 0.9, "class": 1}],
        )
        ground_truth, detections = accumulator.payloads()
        self.assertEqual([item["ignore"] for item in ground_truth["annotations"]], [0, 1, 1])
        self.assertAlmostEqual(ground_truth["annotations"][0]["vis_ratio"], 0.8)
        self.assertEqual(detections[0]["category_id"], 1)
        self.assertEqual(detections[0]["image_id"], 7)

    def test_modified_official_evaluator_is_rejected(self):
        accumulator = OfficialCityPersonsAccumulator()
        with tempfile.TemporaryDirectory() as directory:
            evaluator_dir = Path(directory)
            (evaluator_dir / "coco.py").write_text("# modified\n", encoding="utf-8")
            (evaluator_dir / "eval_MR_multisetup.py").write_text(
                "# modified\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                run_official_citypersons_evaluator(
                    accumulator, evaluator_dir, evaluator_dir / "output"
                )


if __name__ == "__main__":
    unittest.main()
