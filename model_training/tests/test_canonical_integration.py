"""Focused tests for canonical geometry and ignore-region semantics."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from person_detection.data.dataset import (
    CanonicalPersonDetectionDataset,
    clip_box_to_image,
    letterbox_image,
    map_letterbox_box,
)
from person_detection.evaluation.inference import MAPCalculator
from person_detection.training.pipeline import SSDLoss


class CanonicalIntegrationTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_maps_boxes(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        transformed, scale, pad_x, pad_y = letterbox_image(image, 100, 100)
        self.assertEqual(transformed.shape, (100, 100, 3))
        self.assertEqual((scale, pad_x, pad_y), (0.5, 0, 25))
        self.assertEqual(map_letterbox_box([20, 10, 100, 90], scale, pad_x, pad_y),
                         [10.0, 30.0, 50.0, 70.0])

    def test_degenerate_visible_box_is_omitted_from_transform_geometry(self):
        self.assertIsNone(clip_box_to_image([27, 54, 27, 74], 100, 80))

    def test_ignore_regions_neutralize_background_anchors(self):
        loss = SSDLoss(input_height=100, input_width=100)
        anchors = torch.tensor([
            [0.30, 0.30, 0.20, 0.20],
            [0.80, 0.80, 0.10, 0.10],
        ])
        _, labels, positives = loss.match_anchors(
            torch.empty((0, 4)),
            torch.empty((0,), dtype=torch.long),
            anchors,
            ignore_boxes=torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
        )
        self.assertEqual(labels.tolist(), [-1, 0])
        self.assertFalse(positives.any())

    def test_ignore_region_predictions_are_neutral_for_ap(self):
        calculator = MAPCalculator([0.5])
        calculator.add_ground_truths(0, [[0, 0, 10, 10]], [1])
        calculator.add_ignore_regions(0, [[20, 20, 40, 40]])
        calculator.add_predictions(0, [
            {"class": 1, "score": 0.9, "box": [0, 0, 10, 10]},
            {"class": 1, "score": 0.8, "box": [20, 20, 40, 40]},
        ])
        self.assertAlmostEqual(calculator.compute_map(verbose=False)["mAP@0.50"], 1.0)

    def test_empty_record_preserves_coco_metric_schema(self):
        calculator = MAPCalculator([0.5 + index * 0.05 for index in range(10)])
        calculator.add_image(0)

        metrics = calculator.compute_map(verbose=False)

        self.assertEqual(metrics["mAP@0.50:0.95"], 0.0)
        self.assertEqual(metrics["Recall@FPPI=0.10"], 0.0)


    def test_validate_samples_reads_image_and_annotation(self):
        class FakeClient:
            @staticmethod
            def get_object_bytes(bucket, name):
                return b"available"

        dataset = object.__new__(CanonicalPersonDetectionDataset)
        dataset.samples = [{"image": "image.png", "annotation": "annotation.json"}]
        dataset.config = SimpleNamespace(azurite_data_bucket="data")
        dataset.azurite = FakeClient()
        loaded = []
        dataset._load_annotation = loaded.append

        self.assertTrue(dataset.validate_samples(num_samples=1))
        self.assertEqual(loaded, dataset.samples)


if __name__ == "__main__":
    unittest.main()
