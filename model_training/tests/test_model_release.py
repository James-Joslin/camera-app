import json
import tempfile
import unittest
from pathlib import Path

from scripts.models.publish_release import (
    REQUIRED_ARTIFACTS,
    build_current_pointer,
    build_release_manifest,
    publish_release,
)
from person_detection.optimization.pipeline import (
    quality_gate,
    read_camera_metrics,
    release_is_accepted,
)


class ModelReleaseTests(unittest.TestCase):
    def test_builds_manifest_for_accepted_complete_release(self):
        with tempfile.TemporaryDirectory() as directory:
            release_dir = Path(directory)
            for relative_name in REQUIRED_ARTIFACTS:
                path = release_dir / relative_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            (release_dir / "models/optimization_report.json").write_text(
                json.dumps(
                    {
                        "accuracyControl": {"accepted": True, "measuredDrop": 0.001},
                        "release": {"status": "experimental", "accepted": True},
                        "topKValidation": {"accepted": True},
                        "checkpoint": {"sha256": "a" * 64},
                        "dataset": {"versionPrefix": "datasets/citypersons/v1"},
                        "preprocessing": {"inputHeight": 360, "inputWidth": 640},
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_release_manifest(
                release_dir,
                release_id="v1",
                container_name="models",
                prefix="person_detector_ssd/releases/v1",
            )

            self.assertEqual(manifest["releaseId"], "v1")
            self.assertTrue(manifest["accuracyControl"]["accepted"])
            self.assertEqual(manifest["release"]["status"], "experimental")
            self.assertEqual(
                {item["path"] for item in manifest["artifacts"]},
                set(REQUIRED_ARTIFACTS),
            )

    def test_rejects_unaccepted_int8_release(self):
        with tempfile.TemporaryDirectory() as directory:
            release_dir = Path(directory)
            for relative_name in REQUIRED_ARTIFACTS:
                path = release_dir / relative_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            (release_dir / "models/optimization_report.json").write_text(
                json.dumps({
                    "accuracyControl": {"accepted": False},
                    "release": {"status": "experimental", "accepted": False},
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "not accepted"):
                build_release_manifest(
                    release_dir,
                    release_id="v1",
                    container_name="models",
                    prefix="person_detector_ssd/releases/v1",
                )


    def test_rejects_release_when_quality_status_gate_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            release_dir = Path(directory)
            for relative_name in REQUIRED_ARTIFACTS:
                path = release_dir / relative_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            (release_dir / "models/optimization_report.json").write_text(
                json.dumps({
                    "accuracyControl": {"accepted": True},
                    "release": {"status": "production", "accepted": False},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "release gates"):
                build_release_manifest(
                    release_dir,
                    release_id="v1",
                    container_name="models",
                    prefix="person_detector_ssd/releases/v1",
                )

    def test_current_pointer_names_every_openvino_model_pair(self):
        pointer = build_current_pointer(
            {
                "releaseId": "v1",
                "release": {"status": "production", "accepted": True},
                "storage": {
                    "container": "models",
                    "prefix": "person_detector_ssd/releases/v1",
                },
            },
            current_pointer="person_detector_ssd/current.json",
        )

        self.assertEqual(pointer["releaseId"], "v1")
        self.assertEqual(pointer["releaseStatus"], "production")
        self.assertEqual(
            pointer["storage"]["releaseManifest"],
            "person_detector_ssd/releases/v1/release_manifest.json",
        )
        for precision in ("fp32", "fp16", "int8"):
            self.assertEqual(
                pointer["models"][precision],
                {
                    "xml": f"person_detector_ssd/releases/v1/models/person_detector_{precision}.xml",
                    "bin": f"person_detector_ssd/releases/v1/models/person_detector_{precision}.bin",
                },
            )

    def test_current_pointer_cannot_overlap_immutable_release(self):
        with self.assertRaisesRegex(ValueError, "immutable release prefix"):
            publish_release(
                Path("."),
                release_id="v1",
                container_name="models",
                prefix="person_detector_ssd/releases/v1",
                current_pointer="person_detector_ssd/releases/v1/current.json",
            )


class OptimizationReleaseGateTests(unittest.TestCase):
    def test_quality_gate_checks_every_required_metric(self):
        result = quality_gate(
            {
                "mAP@0.50": 0.30,
                "mAP@0.50:0.95": 0.09,
                "Recall@FPPI=0.10": 0.25,
            },
            {
                "mAP@0.50": 0.25,
                "mAP@0.50:0.95": 0.10,
                "Recall@FPPI=0.10": 0.20,
            },
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["mAP@0.50:0.95"])

    def test_experimental_can_publish_without_production_evidence(self):
        self.assertTrue(release_is_accepted(
            "experimental",
            quantization_accepted=True,
            topk_accepted=True,
            quality_accepted=False,
            camera_accepted=False,
        ))
        self.assertFalse(release_is_accepted(
            "production",
            quantization_accepted=True,
            topk_accepted=True,
            quality_accepted=False,
            camera_accepted=False,
        ))

    def test_camera_gate_reads_evaluator_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "camera.json"
            report_path.write_text(json.dumps({
                "projectBinary": {"metrics": {
                    "mAP@0.50": 0.30,
                    "mAP@0.50:0.95": 0.15,
                    "Recall@FPPI=0.10": 0.25,
                }}
            }), encoding="utf-8")
            result = read_camera_metrics(report_path, {
                "mAP@0.50": 0.25,
                "mAP@0.50:0.95": 0.10,
                "Recall@FPPI=0.10": 0.20,
            })
        self.assertTrue(result["evaluated"])
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
