"""G loss identities, auxiliary training, checkpoint and deployed graph contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import openvino as ov

from person_detection.modeling.assignment import ATSSAnchorAssigner, AssignmentResult
from person_detection.modeling.occlusion import (
    OcclusionConfig, OcclusionLoss, auxiliary_scale, smooth_ln,
    repgt_sum, repbox_sum, select_repbox_predictions,
)
from person_detection.training.pipeline import (
    SSDPersonDetector, SSDLoss, TrainingConfig, DetectorTrainingPipeline,
    detector_state_dict, load_training_state, make_training_checkpoint,
    checkpoint_resume_mismatches, checkpoint_is_selected,
    training_forward_loss, log_epoch_to_tensorboard,
)
from person_detection.evaluation.inference import InferenceConfig, load_model
from person_detection.optimization.pipeline import load_checkpoint_model


def make_model(visible=True):
    return SSDPersonDetector(pretrained=False, input_height=72, input_width=128,
                             visible_auxiliary=visible)


def make_config(**kwargs):
    return TrainingConfig(input_height=72, input_width=128, visible_loss_weight=.25,
                          repgt_loss_weight=.05, repbox_loss_weight=.01,
                          checkpoint_selection='ap', validation_ap_every_n_epochs=1,
                          use_azurite=False, **kwargs)


def targets():
    return [dict(boxes=torch.tensor([[12., 8., 55., 65.], [30., 12., 70., 67.]]),
                 labels=torch.ones(2, dtype=torch.long),
                 visible_boxes=torch.tensor([[14., 9., 32., 30.], [42., 14., 68., 35.]]),
                 visible_box_indices=torch.tensor([0, 1]), image_id=0),
            dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long), image_id=1)]


class OcclusionTests(unittest.TestCase):
    def test_assignment_final_identity_and_empty_ignored(self):
        assigner = ATSSAnchorAssigner(point_regression=True)
        refs = torch.tensor([[0., 0., 10., 10.], [2., 0., 12., 10.], [40., 0., 50., 10.]])
        gt = torch.tensor([[0., 0., 14., 12.], [0., 0., 14., 12.]])
        result = assigner.assign(refs, [3], gt, torch.ones(2, dtype=torch.long), refs[2:])
        self.assertEqual(set(result.matched_gt_indices[result.positive_mask].tolist()), {0, 1})
        self.assertEqual(result.matched_gt_indices[2], -1)
        self.assertEqual(result.matched_labels[2], -1)
        empty = assigner.assign(refs, [3], gt[:0], torch.empty(0, dtype=torch.long), refs)
        self.assertTrue((empty.matched_gt_indices == -1).all())

    def test_repulsion_overlap_exclusion_sampling_and_chunks(self):
        gt = torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.]])
        boxes = gt.repeat_interleave(3, dim=0).requires_grad_()
        owners = torch.tensor([0, 0, 0, 1, 1, 1])
        self.assertEqual(select_repbox_predictions(boxes, owners, gt, 2).tolist(), [0, 1, 3, 4])
        small = repbox_sum(boxes, owners, gt, limit=2, chunk_size=1)
        large = repbox_sum(boxes, owners, gt, limit=2, chunk_size=64)
        torch.testing.assert_close(small[0], large[0])
        self.assertEqual(small[1:], (4, 4))
        repgt, active = repgt_sum(boxes, owners, gt, chunk_size=1)
        self.assertGreater(active, 0)
        torch.testing.assert_close(repgt, repgt_sum(boxes, owners, gt, chunk_size=64)[0])
        (repgt + small[0]).backward()
        self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertGreater(boxes.grad.abs().sum(), 0)
        same = repbox_sum(boxes[:3], owners[:3], gt)
        self.assertEqual(same[0].item(), 0)
        separate = torch.tensor([[0., 0., 1., 1.], [5., 5., 6., 6.]], requires_grad=True)
        self.assertEqual(repbox_sum(separate, torch.tensor([0, 1]), separate.detach())[1], 0)
        self.assertEqual(repgt_sum(separate, torch.tensor([0, 1]), separate.detach())[0].item(), 0)
        overlap = torch.tensor([0., .5, 1.], requires_grad=True)
        smooth_ln(overlap, .5).sum().backward()
        self.assertTrue(torch.isfinite(overlap.grad).all())
        torch.testing.assert_close(overlap.grad, torch.tensor([1., 2., 2.]))

    def test_visible_mapping_degenerate_and_noncontained(self):
        cfg = OcclusionConfig(visible_loss_weight=.25)
        loss = OcclusionLoss(cfg)
        gt = torch.tensor([[0., 0., 10., 20.], [10., 0., 20., 20.]])
        result = AssignmentResult(gt, torch.ones(2, dtype=torch.long), torch.tensor([True, True]), torch.tensor([0, 1]))
        target = dict(boxes=gt, visible_boxes=torch.tensor([[12., 1., 19., 8.], [0., 0., 0., 0.]]),
                      visible_box_indices=torch.tensor([1, 0]))
        visible = torch.tensor([[[1., 1., 5., 5.], [8., 10., 12., 14.]]], requires_grad=True)
        total, metrics = loss(gt[None], [target], [result], visible, torch.ones(2, 1))
        self.assertEqual(metrics['visible_valid_pairs'], 1)
        self.assertEqual(metrics['visible_skipped_pairs'], 1)
        self.assertEqual(metrics['visible_supervised'], 1)
        total.backward()
        self.assertEqual(visible.grad[0, 0].abs().sum(), 0)
        self.assertGreater(visible.grad[0, 1].abs().sum(), 0)
        target['visible_boxes'][0, 0] = 9
        self.assertEqual(loss(gt[None], [target], [result], visible, torch.ones(2, 1))[1]['visible_noncontained_pairs'], 1)
        target['visible_box_indices'] = torch.tensor([0, 0])
        with self.assertRaises(ValueError):
            loss(gt[None], [target], [result], visible, torch.ones(2, 1))

    def test_empty_supervision_and_zero_weights(self):
        predictions = torch.zeros(1, 2, 4, requires_grad=True)
        result = AssignmentResult(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long),
                                  torch.zeros(2, dtype=torch.bool), torch.full((2,), -1))
        loss, metrics = OcclusionLoss(make_config())(predictions, [targets()[1]], [result],
                                                     predictions, torch.ones(2, 1))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        with patch('person_detection.modeling.occlusion.repgt_sum', side_effect=AssertionError), \
             patch('person_detection.modeling.occlusion.repbox_sum', side_effect=AssertionError):
            zero, _ = OcclusionLoss(OcclusionConfig())(predictions, [targets()[1]], [result])
            self.assertEqual(zero.item(), 0)

    def test_combined_amp_backward_checkpoint_and_stripping(self):
        torch.manual_seed(7)
        model, config = make_model(), make_config()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        criterion = SSDLoss(input_height=72, input_width=128,
                            anchors_per_level=model.anchor_generator.num_anchors_per_level,
                            occlusion_config=config, distance_scales=model.distance_scales)
        images = torch.randn(2, 3, 72, 128)
        initial_aux = model.auxiliary_head.weight.detach().clone()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, _, loss, metrics = training_forward_loss(model, criterion, images, targets(),
                                                        model.anchor_generator.get_anchors())
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(metrics['visible_supervised'], 0)
        self.assertGreater(metrics['repbox_selected_count'], 0)
        loss.backward()
        for layer in (model.auxiliary_head, model.detection_heads.shared.pointwise,
                      model.detection_heads.box_outputs[0]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial_aux, model.auxiliary_head.weight))
        from person_detection.training.pipeline import validate
        validation = validate(model, [(images, targets())], criterion,
                              model.anchor_generator.get_anchors(), 'cpu', compute_ap=True, config=config)
        self.assertGreater(validation['visible_supervised'], 0)
        self.assertIn('mAP@0.50:0.95', validation)
        self.assertAlmostEqual(validation['loss'], validation['detection_loss'] +
                               sum(validation[key] for key in ('visible_loss_weighted', 'repgt_loss_weighted',
                                                              'repbox_loss_weighted')), places=5)
        checkpoint = make_training_checkpoint(epoch=5, training_complete=False, model=model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, best_val_loss=1., config=config,
            dataset_metadata={}, anchor_generator=model.anchor_generator, sampling_generator=None,
            best_val_map=.2)
        self.assertFalse(any(k.startswith('auxiliary_head.') for k in checkpoint['model_state_dict']))
        self.assertTrue(checkpoint['auxiliary_state_dict'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'g.pth'
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=True)
            inference_model, _, kind = load_model(InferenceConfig(
                model_path=str(path), input_height=72, input_width=128, device='cpu'))
            self.assertEqual(kind, 'pytorch')
            self.assertIsNone(inference_model.auxiliary_head)
            optimization_model = load_checkpoint_model(restored, 72, 128)
            self.assertIsNone(optimization_model.auxiliary_head)
            clone = make_model()
            load_training_state(clone, restored)
            other_optimizer = torch.optim.SGD(clone.parameters(), lr=.001, momentum=.9)
            other_optimizer.load_state_dict(restored['optimizer_state_dict'])
            self.assertEqual(len(optimizer.state), len(other_optimizer.state))
            self.assertFalse(checkpoint_resume_mismatches(restored, config, {}, model.anchor_generator.specification()))
            config.repgt_loss_weight = .1
            self.assertIn('occlusion', checkpoint_resume_mismatches(restored, config, {}, model.anchor_generator.specification()))
        plain = make_model(False).eval()
        plain.load_state_dict(detector_state_dict(model), strict=True)
        model.eval(); clone.eval()
        with torch.no_grad():
            expected = plain(images)
            ordinary = model(images)
            auxiliary = model.forward_auxiliary(images)
            resumed = clone(images)
        for i in range(2):
            torch.testing.assert_close(expected[i], ordinary[i], rtol=0, atol=0)
            torch.testing.assert_close(expected[i], auxiliary[i], rtol=0, atol=0)
            torch.testing.assert_close(expected[i], resumed[i], rtol=0, atol=0)
        self.assertTrue((auxiliary[2][..., 2:] > auxiliary[2][..., :2]).all())

    def test_openvino_graph_and_detection_parity(self):
        model = make_model().eval()
        plain = make_model(False).eval()
        plain.load_state_dict(detector_state_dict(model))
        example = torch.zeros(1, 3, 72, 128)
        exported = ov.convert_model(plain, example_input=example)
        baseline = ov.convert_model(make_model(False).eval(), example_input=example)
        self.assertEqual([op.get_type_name() for op in exported.get_ordered_ops()],
                         [op.get_type_name() for op in baseline.get_ordered_ops()])
        self.assertFalse(any('auxiliary' in op.get_friendly_name() for op in exported.get_ordered_ops()))
        self.assertEqual(sum(p.numel() for p in plain.parameters()),
                         sum(p.numel() for p in make_model(False).parameters()))
        compiled = ov.Core().compile_model(exported, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 1})
        outputs = compiled(example.numpy())
        with torch.no_grad():
            expected = model(example)
        for i in range(2):
            np.testing.assert_allclose(outputs[compiled.output(i)], expected[i].numpy(), rtol=1e-4, atol=1e-4)
        import nncf
        from person_detection.modeling.clean_head import mark_openvino_outputs, has_decoded_boxes
        mark_openvino_outputs(exported, plain.model_variant)
        samples = [np.random.default_rng(seed).normal(size=(1, 3, 72, 128)).astype(np.float32)
                   for seed in (1, 2)]
        quantized = nncf.quantize(exported, nncf.Dataset(samples), subset_size=2,
                                  target_device=nncf.TargetDevice.CPU)
        self.assertFalse(any('auxiliary' in op.get_friendly_name() for op in quantized.get_ordered_ops()))
        self.assertTrue(has_decoded_boxes(quantized.outputs))
        quantized_cpu = ov.Core().compile_model(quantized, 'CPU', {'INFERENCE_NUM_THREADS': 1})
        quantized_outputs = quantized_cpu(samples[0])
        self.assertEqual(len(quantized_outputs), 2)
        self.assertTrue(all(np.isfinite(value).all() for value in quantized_outputs.values()))

    def test_selection_configuration_ramp_and_logging(self):
        self.assertTrue(checkpoint_is_selected('ap', loss_improved=False, ap_improved=True, final_epoch=False))
        self.assertFalse(checkpoint_is_selected('ap', loss_improved=True, ap_improved=False, final_epoch=False))
        self.assertEqual([auxiliary_scale(e, 5) for e in (0, 1, 5, 59)], [0, .2, 1, 1])
        with patch.dict('os.environ', {'TRAINING_VISIBLE_LOSS_WEIGHT': '.25', 'TRAINING_REPGT_LOSS_WEIGHT': '.05',
                                      'TRAINING_REPBOX_LOSS_WEIGHT': '.01', 'TRAINING_CHECKPOINT_SELECTION': 'ap',
                                      'TRAINING_AP_EVERY_N_EPOCHS': '1'}):
            self.assertEqual(DetectorTrainingPipeline.from_environment().config.visible_loss_weight, .25)
        invalid = make_config(); invalid.validation_ap_every_n_epochs = 0
        with self.assertRaises(ValueError):
            DetectorTrainingPipeline(invalid)
        from unittest.mock import Mock
        writer = Mock()
        metrics = dict(loss=1., cls_loss=.2, loc_loss=.3, detection_loss=.5,
                       visible_loss=.1, repgt_overlap_count=2, duration_seconds=1.)
        log_epoch_to_tensorboard(writer, 1, metrics, metrics, .001)
        tags = [call.args[0] for call in writer.add_scalar.call_args_list]
        self.assertIn('Occlusion/train/visible_loss', tags)
        self.assertIn('Occlusion/validation/repgt_overlap_count', tags)


class OcclusionIntegrationTests(unittest.TestCase):
    def test_geometry_keeps_visible_identity_and_disables_dropout(self):
        import cv2
        from types import SimpleNamespace
        from person_detection.data import dataset as data
        from person_detection.data.annotations import CanonicalAnnotation, CanonicalObject
        source = np.full((72, 128, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode('.png', source)
        annotation = CanonicalAnnotation('test.png', 128, 72, 'x' * 64, [
            CanonicalObject('a', [10., 10., 30., 60.], [12., 12., 25., 30.], 1, 'pedestrian', {}),
            CanonicalObject('b', [50., 10., 70., 60.], [50., 10., 50., 10.], 1, 'pedestrian', {}),
        ], [])
        dataset = object.__new__(data.CanonicalPersonDetectionDataset)
        dataset.input_height, dataset.input_width = 72, 128
        dataset.augment, dataset.coarse_dropout = True, False
        dataset._setup_transform()
        self.assertEqual(dataset.appearance_transform.transforms[0].p, 0)
        dataset.geometry_transform = data.A.Compose([data.A.HorizontalFlip(p=1)],
            bbox_params=data.A.BboxParams(format='pascal_voc', label_fields=['bbox_kinds', 'bbox_indices']))
        dataset.appearance_transform = None
        dataset.samples = [dict(image='test.png', image_relative='test.png')]
        dataset.config = SimpleNamespace(azurite_data_bucket='test')
        dataset.azurite = SimpleNamespace(get_object_bytes=lambda *args: encoded.tobytes())
        dataset._load_annotation = lambda sample: annotation
        _, target = dataset[0]
        self.assertEqual(target['object_ids'], ['a', 'b'])
        self.assertEqual(target['visible_box_indices'].tolist(), [0])
        self.assertEqual(len(target['boxes']), 2)
        torch.testing.assert_close(target['visible_boxes'][0], torch.tensor([103., 12., 116., 30.]))

    def test_pipeline_selection_optimizers_and_resume(self):
        import os
        import contextlib
        import io
        from unittest.mock import Mock
        from person_detection.training import pipeline
        original_model = SSDPersonDetector

        class TinyDataset(torch.utils.data.Dataset):
            dataset_metadata = {}
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return torch.ones(3, 72, 128), targets()[index]
            def validate_samples(self, **kwargs):
                return True

        def factory(**kwargs):
            kwargs['pretrained'] = False
            return original_model(**kwargs)

        for varied in (False, True):
            with self.subTest(varied_lr=varied), tempfile.TemporaryDirectory() as directory:
                old_cwd = os.getcwd()
                try:
                    os.chdir(directory)
                    config = make_config(num_epochs=2, batch_size=2, num_workers=0,
                        tensorboard_enabled=False, use_stratified_oversampling=False,
                        use_varied_lr=varied, auxiliary_ramp_epochs=0, use_amp=False)
                    metrics = [dict(loss=9., detection_loss=2., **{'mAP@0.50': .4, 'mAP@0.50:0.95': .3, 'Recall@FPPI=0.10': .2}),
                               dict(loss=1., detection_loss=1., **{'mAP@0.50': .3, 'mAP@0.50:0.95': .2, 'Recall@FPPI=0.10': .1})]
                    exporter = Mock()
                    with patch.object(pipeline, 'CanonicalPersonDetectionDataset', side_effect=lambda *a, **kw: TinyDataset()), \
                         patch.object(pipeline, 'SSDPersonDetector', side_effect=factory), \
                         patch.object(pipeline, 'model_summary'), \
                         patch.object(pipeline, 'OpenVINOExporter', return_value=exporter), \
                         patch.object(pipeline, 'validate', side_effect=metrics), \
                         contextlib.redirect_stdout(io.StringIO()):
                        DetectorTrainingPipeline(config).run()
                        selected = torch.load('best_model_fp32.pth', weights_only=True)
                        by_loss = torch.load('best_model_loss.pth', weights_only=True)
                        last = torch.load('last_training_checkpoint.pth', weights_only=True)
                        self.assertEqual(selected['epoch'], 0)
                        self.assertEqual(by_loss['epoch'], 1)
                        self.assertEqual(last['completedEpochs'], 2)
                        self.assertEqual(by_loss['selection']['metric'], 'detection_loss')
                        self.assertTrue(last['trainingComplete'])
                        self.assertTrue(last['auxiliary_state_dict'])
                        self.assertIsNone(exporter.export_to_openvino.call_args.args[0].auxiliary_head)
                        # Completed run restores all training state and exports the same selected detector.
                        DetectorTrainingPipeline(config).run()
                        self.assertEqual(exporter.export_to_openvino.call_count, 2)
                finally:
                    os.chdir(old_cwd)


if __name__ == '__main__':
    unittest.main()
