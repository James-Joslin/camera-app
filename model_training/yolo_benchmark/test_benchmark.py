import unittest

import numpy as np
import torch

from yolo_benchmark.benchmark import CanvasModel, decode, preprocess, verify_export


class BenchmarkTests(unittest.TestCase):
    def test_rgb_scale_and_letterbox(self):
        image = np.full((100, 200, 3), [255, 128, 0], dtype=np.uint8)
        tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 360, 640))
        np.testing.assert_allclose(tensor[0, :, 20, 0], [1, 128 / 255, 0])
        self.assertEqual(float(tensor[:, :, :20].sum()), 0)

    def test_padding_preserves_canvas(self):
        wrapper = CanvasModel(torch.nn.Identity())
        x = torch.ones(1, 3, 360, 640)
        y = wrapper(x)
        self.assertEqual(tuple(y.shape), (1, 3, 384, 640))
        torch.testing.assert_close(y[:, :, :360], x)
        self.assertEqual(y[:, :, 360:].sum(), 0)

    def test_person_probabilities_nms_and_canvas_clipping(self):
        output = np.zeros((1, 84, 5), dtype=np.float32)
        output[0, :4] = np.array([[10, 20, 40, 60], [11, 21, 41, 61],
                                  [50, 370, 70, 383], [-10, 300, 650, 380],
                                  [100, 100, 120, 150]]).T
        output[0, 4] = [.9, .8, .95, .7, .001]
        output[0, 5, 4] = .99  # A confident non-person must not become a person.
        detections = decode(output)
        self.assertEqual(len(detections), 2)
        self.assertAlmostEqual(detections[0]['score'], .9)
        self.assertEqual(detections[0]['box'], [10, 20, 40, 60])
        self.assertEqual(detections[1]['box'], [0, 300, 640, 360])
        self.assertEqual(detections[0]['class'], 1)

    def test_end2end_keeps_overlaps_without_nms_and_filters_classes(self):
        output = np.array([[[10, 20, 40, 60, .9, 0], [11, 21, 41, 61, .8, 0],
                            [100, 100, 120, 150, .99, 1], [20, 370, 40, 383, .95, 0],
                            [-10, 300, 650, 380, .7, 0], [10, 20, 40, 60, .001, 0]]], dtype=np.float32)
        detections = decode(output, end2end=True)
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0]['box'], [10, 20, 40, 60])
        self.assertEqual(detections[1]['box'], [11, 21, 41, 61])
        self.assertEqual(detections[2]['box'], [0, 300, 640, 360])
        self.assertEqual(len(decode(output, end2end=True, max_detections=1)), 1)
        self.assertEqual(decode(output, end2end=True, threshold=.99), [])

    def test_end2end_empty_and_invalid_shape(self):
        self.assertEqual(decode(np.zeros((1, 300, 6), dtype=np.float32), end2end=True), [])
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 84, 300), dtype=np.float32), end2end=True)

    def test_end2end_export_parity_allows_reordering_but_not_wrong_detections(self):
        reference = np.array([[[10, 20, 40, 60, .9, 0], [100, 20, 140, 60, .9, 1]]], dtype=np.float32)
        actual = reference[:, ::-1].copy()
        self.assertEqual(verify_export(actual, reference, end2end=True)['maxAbsoluteError'], 0)
        actual[0, 0, 0] += 10
        with self.assertRaises(AssertionError):
            verify_export(actual, reference, end2end=True)
        actual = reference.copy()
        actual[0, 0, 5] = 1
        with self.assertRaises(AssertionError):
            verify_export(actual, reference, end2end=True)

    def test_empty_and_wrong_output(self):
        self.assertEqual(decode(np.zeros((1, 84, 10), dtype=np.float32)), [])
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 10, 84), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
