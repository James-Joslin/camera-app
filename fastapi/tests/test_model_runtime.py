import numpy as np

from app.model_runtime import (
    classification_scores,
    generate_anchors,
    map_box_from_letterbox,
    preprocess,
)


def test_preprocess_uses_aspect_preserving_letterbox() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = preprocess(image, 360, 640)

    assert tensor.shape == (1, 3, 360, 640)
    assert tensor.dtype == np.float32
    assert (scale, pad_x, pad_y) == (3.2, 0, 20)


def test_maps_letterboxed_box_back_to_source_image() -> None:
    mapped = map_box_from_letterbox(
        np.array([32, 52, 320, 308], dtype=np.float32),
        scale=3.2,
        pad_x=0,
        pad_y=20,
        original_width=200,
        original_height=100,
    )

    assert np.allclose(mapped, [10, 10, 100, 90])


def test_supports_quality_logit_and_legacy_softmax_outputs() -> None:
    quality = classification_scores(np.array([[0.0], [2.0]], dtype=np.float32))
    legacy = classification_scores(np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    assert np.allclose(quality, [0.5, 0.880797], atol=1e-6)
    assert np.allclose(legacy, [0.5, 0.880797], atol=1e-6)


def test_serving_anchors_match_current_person_detector() -> None:
    anchors = generate_anchors(360, 640)

    assert anchors.shape == (29235, 4)
    physical_ratio = anchors[0, 2] * 640 / (anchors[0, 3] * 360)
    assert np.isclose(physical_ratio, 0.15)


def test_serving_accepts_named_decoded_boxes_without_anchors(tmp_path) -> None:
    import openvino as ov
    from openvino import opset13 as ops
    from app.model_runtime import OpenVinoPersonDetector

    image = ops.parameter([1, 3, 72, 128], np.float32)
    # Keep an input-dependent graph while supplying known pixel box geometry.
    zero = ops.multiply(ops.reduce_mean(image, ops.constant([0, 1, 2, 3]), False), ops.constant(0., np.float32))
    logits = ops.add(ops.constant(np.array([[[8.]]], np.float32)), zero)
    boxes = ops.add(ops.constant(np.array([[[16., 12., 48., 60.]]], np.float32)), zero)
    model = ov.Model([logits, boxes], [image])
    model.output(0).get_tensor().set_names({"person_quality_logits"})
    model.output(1).get_tensor().set_names({"boxes_xyxy_pixels"})
    path = tmp_path / "person.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    runtime = OpenVinoPersonDetector(path)
    detections, _ = runtime.predict(np.zeros((72, 128, 3), np.uint8))
    assert runtime.anchors is None
    assert len(detections) == 1
    assert detections[0]["box"] == [16, 12, 48, 60]
