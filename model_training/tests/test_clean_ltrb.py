"""C/E geometry, gradient, checkpoint and real OpenVINO integration tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import openvino as ov

from person_detection.modeling.assignment import ATSSAnchorAssigner
from person_detection.modeling.clean_head import has_decoded_boxes, mark_openvino_outputs
from person_detection.training.pipeline import (
    SSDPersonDetector, SSDLoss, TrainingConfig, DetectorTrainingPipeline,
    decode_boxes, checkpoint_resume_mismatches, add_validation_map_batch,
)
from person_detection.evaluation.metrics import MAPCalculator
from person_detection.evaluation.inference import (
    DetectionPredictor, OpenVINOPredictor, InferenceConfig, load_model,
)
from person_detection.optimization.pipeline import decode_predictions, load_checkpoint_model


class CleanModelTests(unittest.TestCase):
    def test_ce_shapes_and_compute_structure_preserve_baseline_variant(self):
        models = {name: SSDPersonDetector(pretrained=False, model_variant=name).eval()
                  for name in ("anchor", "clean_anchor", "clean_ltrb")}
        with torch.no_grad():
            for name, model in models.items():
                cls, boxes = model(torch.zeros(1, 3, 360, 640))
                count = 4835 if name == "clean_ltrb" else 29235
                self.assertEqual(tuple(cls.shape), (1, count, 1))
                self.assertEqual(tuple(boxes.shape), (1, count, 4))
        baseline_params = sum(p.numel() for p in models["anchor"].parameters())
        for name in ("clean_anchor", "clean_ltrb"):
            self.assertLess(sum(p.numel() for p in models[name].parameters()), baseline_params)
            self.assertFalse(any(isinstance(m, torch.nn.GroupNorm) for m in models[name].modules()))
            self.assertEqual(models[name].fpn.output_convs[0].depthwise.groups, 128)

    def test_ltrb_graph_decodes_from_grid_and_nominal_stride(self):
        model = SSDPersonDetector(pretrained=False, model_variant="clean_ltrb").eval()
        for layer in model.detection_heads.box_outputs:
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.constant_(layer.bias, 1.0)
        with torch.no_grad():
            _, boxes = model(torch.zeros(1, 3, 360, 640))
        expected_distances = model.distance_scales * 1.001
        torch.testing.assert_close(boxes[0, :, :2], model.point_centers - expected_distances)
        torch.testing.assert_close(boxes[0, :, 2:], model.point_centers + expected_distances)
        # Odd-sized stride-16 map uses 360/23 grid spacing, not 16.
        self.assertAlmostEqual(model.point_centers[3600, 1].item(), 360 / 23 / 2, places=5)

    def test_backward_with_mixed_precision_and_empty_image(self):
        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                  model_variant="clean_ltrb").train()
        criterion = SSDLoss(input_height=72, input_width=128, model_variant="clean_ltrb",
                            anchors_per_level=model.anchor_generator.num_anchors_per_level)
        targets = [
            {"boxes": torch.tensor([[12., 8., 50., 65.]]), "labels": torch.ones(1, dtype=torch.long)},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
        ]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cls, boxes = model(torch.randn(2, 3, 72, 128))
            loss, metrics = criterion(cls, boxes, targets, model.anchor_generator.get_anchors())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["num_pos"], 0)
        loss.backward()
        for layer in (model.detection_heads.box_outputs[0], model.detection_heads.shared.pointwise):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_validation_uses_decoded_boxes_without_anchor_offset_transform(self):
        config = TrainingConfig(model_variant="clean_ltrb", input_height=72, input_width=128)
        calculator = MAPCalculator([0.5 + i * .05 for i in range(10)])
        target = {"image_id": 0, "boxes": torch.tensor([[10., 10., 30., 60.]]), "labels": torch.tensor([1])}
        add_validation_map_batch(calculator, torch.tensor([[[10.]]]), target["boxes"][None],
                                 [target], torch.zeros(1, 4), config)
        self.assertAlmostEqual(calculator.compute_map(False)["mAP@0.50:0.95"], 1.0)

    def test_checkpoint_round_trip_selects_variant_and_rejects_baseline_resume(self):
        config = TrainingConfig(model_variant="clean_ltrb", input_height=72, input_width=128)
        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                  model_variant=config.model_variant)
        checkpoint = {"config": vars(config), "model_state_dict": model.state_dict(),
                      "modelFormatVersion": 4, "boxEncoding": "xyxy_pixels",
                      "anchors": model.anchor_generator.specification()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save(checkpoint, path)
            loaded, anchors, kind = load_model(InferenceConfig(
                model_path=str(path), input_height=72, input_width=128, device="cpu"))
            self.assertEqual(loaded.model_variant, "clean_ltrb")
            self.assertEqual(kind, "pytorch")
            self.assertEqual(len(anchors), len(model.anchor_generator.get_anchors()))
        restored = load_checkpoint_model(checkpoint, 72, 128)
        self.assertEqual(restored.box_encoding, "xyxy_pixels")
        mismatches = checkpoint_resume_mismatches(checkpoint, TrainingConfig(model_variant="anchor"), {}, {})
        self.assertIn("modelVariant", mismatches)
        self.assertIn("modelFormatVersion", mismatches)

    def test_programmatic_defaults_use_ce(self):
        self.assertEqual(TrainingConfig().model_variant, "clean_ltrb")
        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128)
        self.assertEqual(model.model_variant, "clean_ltrb")
        self.assertEqual(model.box_encoding, "xyxy_pixels")
        self.assertEqual(SSDLoss().box_encoding, "xyxy_pixels")
        self.assertTrue(SSDLoss().assigner.point_regression)

    def test_new_training_defaults_to_ce(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(DetectorTrainingPipeline.from_environment().config.model_variant, "clean_ltrb")


class PointAssignmentTests(unittest.TestCase):
    def assign(self, refs, boxes, ignore=None):
        self.assigner = ATSSAnchorAssigner(top_k=1, point_regression=True)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return self.assigner.assign(torch.tensor(refs, dtype=torch.float32), [len(refs)],
                                    boxes, torch.arange(1, len(boxes) + 1),
                                    torch.tensor(ignore or [], dtype=torch.float32).reshape(-1, 4))

    def test_fallback_never_assigns_point_outside_box(self):
        result = self.assign([[0, 0, 2, 2]], [[5, 5, 6, 6]])
        self.assertFalse(result.positive_mask.any())
        self.assertEqual(self.assigner.unmatched_ground_truths, 1)

    def test_conflict_repair_keeps_both_people_when_points_exist(self):
        result = self.assign([[0, 0, 4, 4], [2, 0, 6, 4]],
                             [[0, 0, 5, 4], [1, 0, 5, 4]])
        self.assertEqual(set(result.matched_labels.tolist()), {1, 2})
        self.assertEqual(self.assigner.unmatched_ground_truths, 0)

    def test_augmenting_path_preserves_person_with_only_one_eligible_point(self):
        result = self.assign([[0, 0, 4, 4], [2, 0, 6, 4]],
                             [[0, 0, 5, 4], [1, 1, 3, 3]])
        self.assertEqual(result.matched_labels.tolist(), [2, 1])

    def test_impossible_crowd_conflict_is_counted(self):
        result = self.assign([[0, 0, 4, 4]], [[0, 0, 4, 4], [1, 1, 3, 3]])
        self.assertEqual(result.positive_mask.sum().item(), 1)
        self.assertEqual(self.assigner.unmatched_ground_truths, 1)

    def test_point_ignore_and_positive_precedence(self):
        refs = [[0, 0, 4, 4], [4, 4, 8, 8]]
        empty = self.assign(refs, [], [[1, 1, 3, 3]])
        self.assertEqual(empty.matched_labels.tolist(), [-1, 0])
        positive = self.assign(refs, [[0, 0, 4, 4]], [[1, 1, 3, 3]])
        self.assertEqual(positive.matched_labels[0].item(), 1)


class OpenVINOContractTests(unittest.TestCase):
    def test_int8_conversion_preserves_decoded_output_contract(self):
        import nncf
        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                  model_variant="clean_ltrb").eval()
        example = torch.zeros(1, 3, 72, 128)
        converted = ov.convert_model(model, example_input=example, input=[1, 3, 72, 128])
        mark_openvino_outputs(converted, model.model_variant)
        records = [np.random.default_rng(seed).normal(size=(1, 3, 72, 128)).astype(np.float32)
                   for seed in (1, 2)]
        quantized = nncf.quantize(converted, nncf.Dataset(records), subset_size=2,
                                  target_device=nncf.TargetDevice.CPU)
        self.assertTrue(has_decoded_boxes(quantized.outputs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "int8.xml"
            ov.save_model(quantized, path, compress_to_fp16=False)
            compiled = ov.Core().compile_model(path, "CPU")
            self.assertTrue(has_decoded_boxes(compiled.outputs))
            result = compiled([records[0]])
            boxes = result[compiled.output("boxes_xyxy_pixels")]
            self.assertEqual(boxes.shape, (1, 201, 4))
            self.assertTrue(np.isfinite(boxes).all())
            self.assertTrue((boxes[..., 2:] > boxes[..., :2]).all())

    def test_real_export_round_trip_and_all_evaluation_decoders(self):
        torch.manual_seed(17)
        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                  model_variant="clean_ltrb").eval()
        sample = torch.randn(1, 3, 72, 128)
        with torch.no_grad():
            expected_cls, expected_boxes = model(sample)
        converted = ov.convert_model(model, example_input=sample, input=[1, 3, 72, 128])
        mark_openvino_outputs(converted, model.model_variant)
        anchors = model.anchor_generator.get_anchors()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            ov.save_model(converted, path, compress_to_fp16=False)
            core = ov.Core()
            compiled = core.compile_model(core.read_model(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            self.assertTrue(has_decoded_boxes(compiled.outputs))
            result = compiled([sample.numpy()])
            np.testing.assert_allclose(result[compiled.output("boxes_xyxy_pixels")],
                                       expected_boxes.numpy(), atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(result[compiled.output("person_quality_logits")],
                                       expected_cls.numpy(), atol=1e-4, rtol=1e-4)
            config = InferenceConfig(model_path=str(path), input_height=72, input_width=128, device="cpu")
            ov_predictor = OpenVINOPredictor(str(path), anchors, config)
            torch_predictor = DetectionPredictor(model, anchors, config)
            test_box = torch.tensor([[10., 12., 30., 60.]])
            torch.testing.assert_close(torch_predictor.decode_boxes(test_box), test_box)
            np.testing.assert_array_equal(ov_predictor.decode_boxes(test_box.numpy()), test_box.numpy())
            torch.testing.assert_close(decode_boxes(test_box, anchors[:1], 72, 128,
                                                    encoding="xyxy_pixels"), test_box)
            detections = decode_predictions(compiled, sample.numpy(), anchors.numpy(), 72, 128,
                                             0., .5, 100, 1000)
            self.assertTrue(detections)
            self.assertTrue(all(np.isfinite(item["box"]).all() for item in detections))
            loaded, loaded_anchors, kind = load_model(config)
            self.assertEqual(kind, "openvino")
            self.assertEqual(loaded.box_encoding, "xyxy_pixels")
            self.assertEqual(len(loaded_anchors), len(anchors))


if __name__ == "__main__":
    unittest.main()
