"""Backbone interchange, old checkpoint compatibility and loader IPC regression."""
import os
import subprocess
import sys
import unittest

import torch
from person_detection.training.pipeline import (
    SSDPersonDetector, TrainingConfig, checkpoint_resume_mismatches,
)
from person_detection.optimization.pipeline import load_checkpoint_model


class BackboneTests(unittest.TestCase):
    def test_defaults_and_both_backbones_round_trip(self):
        self.assertEqual(TrainingConfig().backbone, "mobilenetv4_conv_small")
        for backbone in ("mobilenetv3_small", "mobilenetv4_conv_small"):
            with self.subTest(backbone=backbone):
                model = SSDPersonDetector(backbone=backbone, pretrained=False).eval()
                sample = torch.zeros(1, 3, 360, 640)
                with torch.no_grad():
                    expected = model(sample)
                self.assertEqual(expected[0].shape, (1, 4835, 1))
                self.assertEqual(expected[1].shape, (1, 4835, 4))
                config = vars(TrainingConfig(backbone=backbone)).copy()
                if backbone == "mobilenetv3_small":
                    del config["backbone"]  # Real pre-selection checkpoint contract.
                checkpoint = {"config": config, "model_state_dict": model.state_dict(),
                              "modelFormatVersion": 4, "boxEncoding": "xyxy_pixels"}
                restored = load_checkpoint_model(checkpoint, 360, 640)
                self.assertEqual(restored.backbone_name, backbone)
                with torch.no_grad():
                    actual = restored(sample)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                other = "mobilenetv3_small" if backbone.endswith("conv_small") else "mobilenetv4_conv_small"
                mismatches = checkpoint_resume_mismatches(checkpoint, TrainingConfig(backbone=other), {}, {})
                self.assertIn("backbone", mismatches)

    def test_unknown_backbone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            SSDPersonDetector(backbone="typo", pretrained=False)

    def test_worker_batches_survive_low_descriptor_limit(self):
        # Isolate the reduced process limit from the test runner and export tools.
        code = '''
import resource
import torch
from torch.utils.data import DataLoader, Dataset
from person_detection.training.pipeline import initialize_loader_worker, collate_fn
class Samples(Dataset):
    def __len__(self): return 256
    def __getitem__(self, i):
        return torch.zeros(3, 8, 8), {str(k): torch.ones(4) * i for k in range(8)}
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(192, hard), hard))
loader = DataLoader(Samples(), batch_size=32, num_workers=1,
                    worker_init_fn=initialize_loader_worker, collate_fn=collate_fn)
for epoch in range(2):
    count = 0
    for images, targets in loader:
        assert images.shape == (32, 3, 8, 8)
        assert len(targets) == 32
        count += len(targets)
    assert count == 256
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=90, env=dict(os.environ, OMP_NUM_THREADS="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
