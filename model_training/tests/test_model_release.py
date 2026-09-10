import json
import tempfile
import unittest
from pathlib import Path

from scripts.models.publish_release import REQUIRED_ARTIFACTS, build_release_manifest


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


if __name__ == "__main__":
    unittest.main()
