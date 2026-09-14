"""Artifact gates and production shell orchestration without full training/network writes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from person_detection.core.artifacts import PRECISIONS, require_detector_artifacts

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_requires_every_nonempty_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'fp32.xml'):
                require_detector_artifacts(root)
            for precision in PRECISIONS:
                for suffix in ('xml', 'bin'):
                    (root / f'person_detector_{precision}.{suffix}').write_bytes(b'model')
            self.assertEqual(set(require_detector_artifacts(root)), set(PRECISIONS))
            for precision in PRECISIONS:
                for suffix in ('xml', 'bin'):
                    path = root / f'person_detector_{precision}.{suffix}'
                    path.write_bytes(b'')
                    with self.assertRaisesRegex(RuntimeError, path.name):
                        require_detector_artifacts(root)
                    path.write_bytes(b'model')
            with self.assertRaisesRegex(RuntimeError, 'Invalid fp32'):
                require_detector_artifacts(root, read_models=True)


FAKE_PYTHON = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
module = args[1]
def value(flag): return args[args.index(flag) + 1]
def write(path, data='artifact'):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(data)
def log(stage):
    with open(os.environ['CALL_LOG'], 'a') as stream: stream.write(stage + '\n')
mode = os.environ.get('FAIL_STAGE', '')
if module == 'scripts.data.validate_citypersons_azurite':
    log('dataset')
elif module == 'person_detection.training.pipeline':
    log('train')
    assert Path.cwd().name == 'checkpoints'
    assert os.environ['TRAINING_EXPORT_OPENVINO'] == 'false'
    assert os.environ['TRAINING_EPOCHS'] == '60'
    assert os.environ['TRAINING_VISIBLE_LOSS_WEIGHT'] == '0.25'
    write('best_model_fp32.pth'); write('last_training_checkpoint.pth')
elif module == 'scripts.data.download_citypersons_annotations':
    (Path(value('--output')) / 'evaluation/eval_script').mkdir(parents=True)
elif module == 'person_detection.evaluation.inference':
    precision = Path(value('--output-dir')).name
    log('evaluate:' + precision)
    if mode == precision: sys.exit(8)
    assert Path(value('--model-path')).is_file()
    write(Path(value('--output-dir')) / 'evaluation_metrics.json', '{}')
elif module == 'person_detection.optimization.pipeline':
    log('optimize')
    models = Path(value('--output-dir'))
    for precision in ('fp32', 'fp16', 'int8'):
        for ext in ('xml', 'bin'):
            if mode == 'missing' and precision == 'int8' and ext == 'bin': continue
            write(models / f'person_detector_{precision}.{ext}')
    write(models / 'calibration_manifest.json', '{}')
    write(models / 'optimization_report.json', '{}')
    write(models.parent / 'benchmarks/benchmark.json', '{}')
elif module == 'person_detection.core.artifacts':
    log('preflight')
    from person_detection.core.artifacts import require_detector_artifacts
    require_detector_artifacts(value('--models-dir'))
elif module == 'scripts.models.publish_release':
    log('publish')
    write(Path(value('--release-dir')) / 'release_manifest.json', '{}')
else:
    raise AssertionError(module)
'''


class ProductionWorkflowTests(unittest.TestCase):
    def run_workflow(self, failure=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'bin'; binary.mkdir()
            python = binary / 'python'
            python.write_text(f'#!{sys.executable}\n' + FAKE_PYTHON)
            python.chmod(0o755)
            output = root / 'output'
            log = root / 'calls'
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(('TRAINING_', 'MODEL_', 'PERSON_DETECTOR_', 'CITYPERSONS_'))}
            env.update(PATH=f'{binary}:{os.environ["PATH"]}', TRAINING_STATE_ROOT=str(output),
                       TRAINING_RUN_ID='test-run', MODEL_RELEASE_ID='test-release',
                       CALL_LOG=str(log), FAIL_STAGE=failure)
            result = subprocess.run(['bash', str(ROOT / 'runProductionTraining.sh')], env=env,
                                    capture_output=True, text=True)
            calls = log.read_text().splitlines() if log.exists() else []
            files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
            return result, calls, files

    def test_runs_training_evaluation_conversion_and_all_precision_evaluations(self):
        result, calls, files = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['dataset', 'train', 'evaluate:pytorch', 'optimize', 'preflight',
                                 'evaluate:fp32', 'evaluate:fp16', 'evaluate:int8', 'publish'])
        self.assertIn('runs/test-run/checkpoints/best_model_fp32.pth', files)
        self.assertIn('runs/test-run/logs/training.log', files)
        for precision in PRECISIONS:
            for suffix in ('xml', 'bin'):
                self.assertIn(f'releases/test-release/models/person_detector_{precision}.{suffix}', files)
                self.assertIn(f'active-models/person_detector_{precision}.{suffix}', files)
            self.assertIn(f'releases/test-release/evaluation/{precision}/evaluation_metrics.json', files)
        self.assertIn('releases/test-release/benchmarks/benchmark.json', files)

    def test_missing_int8_weights_prevent_evaluation_and_publication(self):
        result, calls, files = self.run_workflow('missing')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, ['dataset', 'train', 'evaluate:pytorch', 'optimize', 'preflight'])
        self.assertFalse(any(path.startswith('active-models/') for path in files))

    def test_failed_evaluation_prevents_publication(self):
        for failure in ('pytorch', 'fp16', 'int8'):
            with self.subTest(failure=failure):
                result, calls, files = self.run_workflow(failure)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('publish', calls)
                self.assertFalse(any(path.startswith('active-models/') for path in files))
                if failure == 'pytorch': self.assertNotIn('optimize', calls)


class OptimizerArtifactTests(unittest.TestCase):
    def test_complete_export_reports_and_missing_pair_blocks_timing(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        import torch
        from person_detection.optimization import pipeline
        from person_detection.training.pipeline import SSDPersonDetector

        model = SSDPersonDetector(pretrained=False, input_height=72, input_width=128).eval()
        record = Mock()
        record.identity.return_value = {"image": "synthetic.png"}
        record.input_tensor.return_value = np.zeros((1, 3, 72, 128), dtype=np.float32)
        accuracy = {"metrics": {"mAP@0.50": .4, "mAP@0.50:0.95": .2, "Recall@FPPI=0.10": .2}}
        original_save = pipeline.ov.save_model
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "model.pth"
                torch.save({"config": {"model_variant": "clean_ltrb", "backbone": model.backbone_name, "input_height": 72, "input_width": 128},
                            "model_state_dict": model.state_dict(), "modelFormatVersion": 4,
                            "boxEncoding": "xyxy_pixels", "anchors": model.anchor_generator.specification()}, checkpoint)
                with patch.object(sys, 'argv', ['optimize', '--checkpoint', str(checkpoint),
                                                '--output-dir', str(root / 'models'), '--no-azurite']):
                    args = pipeline.parse_args()
                def save(ir, path, **kwargs):
                    original_save(ir, path, **kwargs)
                    if missing and Path(path).stem == 'person_detector_int8':
                        Path(path).with_suffix('.bin').unlink()
                with patch.object(pipeline, 'CanonicalPersonDetectionDataset', return_value=SimpleNamespace(dataset_metadata={})), \
                     patch.object(pipeline, 'verify_checkpoint_dataset', return_value={"verified": True}), \
                     patch.object(pipeline, 'select_records', return_value=[record]), \
                     patch.object(pipeline.nncf, 'quantize_with_accuracy_control', side_effect=lambda ir, *a, **kw: ir), \
                     patch.object(pipeline.ov, 'save_model', side_effect=save), \
                     patch.object(pipeline, 'evaluate_compiled_model', return_value=accuracy), \
                     patch.object(pipeline, 'compile_for_cpu', return_value=Mock()), \
                     patch.object(pipeline, 'benchmark_core', return_value={"meanMs": 1.}) as timing, \
                     patch.object(pipeline, 'benchmark_pipeline', return_value={"meanMs": 2.}) as e2e, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if missing:
                        with self.assertRaisesRegex(RuntimeError, 'person_detector_int8.bin'):
                            pipeline.ManifestDrivenOpenVINOOptimizer(args).run()
                        timing.assert_not_called()
                        e2e.assert_not_called()
                    else:
                        report = pipeline.ManifestDrivenOpenVINOOptimizer(args).run()
                        self.assertTrue(report['release']['accepted'])
                        self.assertEqual(timing.call_count, 3)
                        self.assertTrue((root / 'benchmarks/benchmark.json').is_file())
                        for precision in PRECISIONS:
                            self.assertTrue((root / f'evaluation/{precision}/optimization_metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
