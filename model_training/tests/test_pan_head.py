"""PAN geometry, gradient flow, normalization, checkpoint and export contracts."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import openvino as ov
from person_detection.training.pipeline import SSDPersonDetector, SSDLoss, TrainingConfig, checkpoint_resume_mismatches
from person_detection.optimization.pipeline import load_checkpoint_model


class PANHeadTests(unittest.TestCase):
    def model(self, **kwargs):
        return SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                 use_pan=True, regression_depth=2, **kwargs)

    def test_environment_defaults_and_explicit_baseline(self):
        from unittest.mock import patch
        from person_detection.training.pipeline import DetectorTrainingPipeline
        with patch.dict('os.environ', {}, clear=True):
            config = DetectorTrainingPipeline.from_environment().config
            self.assertTrue(config.use_pan)
            self.assertEqual(config.regression_depth, 2)
        with patch.dict('os.environ', {'TRAINING_USE_PAN': 'false', 'TRAINING_REGRESSION_DEPTH': '1'}, clear=True):
            config = DetectorTrainingPipeline.from_environment().config
            self.assertFalse(config.use_pan)
            self.assertEqual(config.regression_depth, 1)
        with self.assertRaises(ValueError):
            DetectorTrainingPipeline(TrainingConfig(regression_depth=3))

    def test_shapes_and_gradients_with_optional_stride4(self):
        for stride4 in (False, True):
            model = self.model(use_stride4=stride4).train()
            cls, boxes = model(torch.randn(2, 3, 72, 128))
            self.assertEqual(boxes.shape[1], len(model.point_centers))
            criterion = SSDLoss(input_height=72, input_width=128,
                                anchors_per_level=model.anchor_generator.num_anchors_per_level)
            targets = [{'boxes': torch.tensor([[10., 8., 60., 65.]]), 'labels': torch.tensor([1])}] * 2
            loss, _ = criterion(cls, boxes, targets, model.anchor_generator.get_anchors())
            loss.backward()
            for module in (model.fpn.pan_downsamples[0].pointwise,
                           model.fpn.pan_fusions[0].pointwise,
                           model.detection_heads.regression_extra.pointwise):
                self.assertIsNotNone(module.weight.grad)
                self.assertTrue(torch.isfinite(module.weight.grad).all())
                self.assertGreater(module.weight.grad.abs().sum().item(), 0)

    def test_batchnorm_statistics_are_independent_per_level(self):
        block = self.model().detection_heads.regression_extra.train()
        before = block.norms[1].running_mean.clone()
        block(torch.randn(2, 128, 8, 8) + 5, 0)
        torch.testing.assert_close(block.norms[1].running_mean, before)
        self.assertEqual(block.norms[0].num_batches_tracked.item(), 1)
        self.assertEqual(block.norms[1].num_batches_tracked.item(), 0)
        self.assertNotEqual(block.norms[0].weight.data_ptr(), block.norms[1].weight.data_ptr())

    def test_old_and_new_checkpoint_loading_and_resume_guard(self):
        for upgraded in (False, True):
            model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                                      use_pan=upgraded, regression_depth=2 if upgraded else 1).eval()
            config = dict(input_height=72, input_width=128, model_variant='clean_ltrb')
            if upgraded:
                config.update(use_pan=True, regression_depth=2)
            checkpoint = dict(config=config, model_state_dict=model.state_dict(),
                              modelFormatVersion=4, boxEncoding='xyxy_pixels')
            restored = load_checkpoint_model(checkpoint, 72, 128)
            sample = torch.randn(1, 3, 72, 128)
            with torch.no_grad():
                expected = model(sample)
                actual = restored(sample)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b)
            mismatches = checkpoint_resume_mismatches(checkpoint,
                TrainingConfig(input_height=72, input_width=128, use_pan=not upgraded,
                               regression_depth=1 if upgraded else 2), {}, {})
            self.assertIn('use_pan', mismatches)
            self.assertIn('regression_depth', mismatches)

    def test_openvino_export_keeps_two_decoded_outputs(self):
        model = self.model().eval()
        sample = torch.randn(1, 3, 72, 128)
        with torch.no_grad():
            expected = model(sample)
        converted = ov.convert_model(model, example_input=sample, input=[1, 3, 72, 128])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.xml'
            ov.save_model(converted, path, compress_to_fp16=False)
            compiled = ov.Core().compile_model(path, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
            actual = compiled([sample.numpy()])
            self.assertEqual(len(compiled.outputs), 2)
            for index, tensor in enumerate(expected):
                np.testing.assert_allclose(actual[compiled.output(index)], tensor.numpy(), atol=1e-4, rtol=1e-4)
