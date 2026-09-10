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
                        "checkpoint": {"sha256": "a" * 64},
                        "dataset": {"versionPrefix": "datasets/citypersons/v1"},
                        "preprocessing": {"inputSize": 480},
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
                json.dumps({"accuracyControl": {"accepted": False}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "not accepted"):
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
                "storage": {
                    "container": "models",
                    "prefix": "person_detector_ssd/releases/v1",
                },
            },
            current_pointer="person_detector_ssd/current.json",
        )

        self.assertEqual(pointer["releaseId"], "v1")
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


if __name__ == "__main__":
    unittest.main()
