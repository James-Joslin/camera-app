"""Focused regression tests for the detector roadmap implementation."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from canonical_annotations import CanonicalAnnotation, CanonicalObject
from canonical_dataset import (
    CanonicalPersonDetectionDataset,
    clip_box_to_image,
    preprocess_rgb_image,
    select_stratified_indices,
)
from citypersons_evaluation import (
    OfficialCityPersonsAccumulator,
    run_official_citypersons_evaluator,
)
from test_inference import MAPCalculator
from train_tune_detector import (
    AttentionDetectionHead,
    PersonAnchorGenerator,
    QualityFocalLoss,
    SSDLoss,
    load_detector_state_dict,
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
            image, 100, add_batch=True, return_geometry=True
        )
        self.assertEqual(tensor.shape, (1, 3, 100, 100))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual((scale, pad_x, pad_y), (0.5, 0, 25))
        self.assertEqual(
            clip_box_to_image([-10, 20, 20, 60], 100, 80), [0.0, 20.0, 20.0, 60.0]
        )

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
        generator = PersonAnchorGenerator(input_size=128)
        first_anchor = generator.get_anchors()[0]
        self.assertLess(float(first_anchor[2]), float(first_anchor[3]))

    def test_atss_selects_adaptive_positive_and_respects_ignore(self):
        criterion = SSDLoss(
            input_size=100, anchors_per_level=[2, 2], atss_topk=1
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


class MixedPrecisionLossTests(unittest.TestCase):
    def test_bfloat16_quality_targets_accept_float32_iou(self):
        criterion = SSDLoss(
            input_size=100, anchors_per_level=[2, 2], atss_topk=1
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
