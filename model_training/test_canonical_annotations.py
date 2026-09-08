"""Unit tests for canonical CityPersons sidecar ingestion."""

import json
import unittest

from canonical_annotations import parse_canonical_annotation


class CanonicalAnnotationTests(unittest.TestCase):
    def annotation(self):
        return {
            "schemaVersion": 1,
            "image": {"blob": "images/train/a/a.png", "width": 100, "height": 80, "sha256": "a" * 64},
            "objects": [{
                "id": "person-1", "detectionClass": "person", "sourceClassId": 3,
                "sourceLabel": "sitting person", "fullBoxXYWH": [10, 20, 20, 40],
                "visibleBoxXYWH": [12, 24, 15, 30], "ignored": False,
                "attributes": {"occlusion": 0.25, "visibility": "partially_occluded"},
            }],
            "ignoreRegions": [{"fullBoxXYWH": [60, 10, 20, 30]}],
        }

    def test_parses_full_visible_and_ignore_boxes(self):
        parsed = parse_canonical_annotation(
            json.dumps(self.annotation()).encode(), expected_image_blob="images/train/a/a.png",
            expected_image_sha256="a" * 64, expected_status="positive",
        )
        self.assertEqual(parsed.objects[0].full_box, [10.0, 20.0, 30.0, 60.0])
    def test_preserves_valid_boundary_crossing_source_box(self):
        value = self.annotation()
        value["objects"][0]["fullBoxXYWH"] = [-10, 20, 30, 40]
        parsed = parse_canonical_annotation(
            json.dumps(value).encode(), expected_image_blob="images/train/a/a.png",
            expected_image_sha256="a" * 64, expected_status="positive",
        )
        self.assertEqual(parsed.objects[0].full_box, [-10.0, 20.0, 20.0, 60.0])

        self.assertEqual(parsed.objects[0].visible_box, [12.0, 24.0, 27.0, 54.0])
        self.assertEqual(parsed.ignore_regions, [[60.0, 10.0, 80.0, 40.0]])

    def test_rejects_non_person_objects(self):
        value = self.annotation()
        value["objects"][0]["detectionClass"] = "car"
        with self.assertRaisesRegex(ValueError, "non-ignored person"):
            parse_canonical_annotation(
                json.dumps(value).encode(), expected_image_blob="images/train/a/a.png"
            )

    def test_rejects_status_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Verified-negative"):
            parse_canonical_annotation(
                json.dumps(self.annotation()).encode(), expected_image_blob="images/train/a/a.png",
                expected_status="verified_negative",
            )


if __name__ == "__main__":
    unittest.main()
