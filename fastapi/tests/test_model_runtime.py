import numpy as np

from app.model_runtime import (
    classification_scores,
    generate_anchors,
    map_box_from_letterbox,
    preprocess,
)


def test_preprocess_uses_aspect_preserving_letterbox() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = preprocess(image, 480)

    assert tensor.shape == (1, 3, 480, 480)
    assert tensor.dtype == np.float32
    assert (scale, pad_x, pad_y) == (2.4, 0, 120)


def test_maps_letterboxed_box_back_to_source_image() -> None:
    mapped = map_box_from_letterbox(
        np.array([24, 144, 240, 336], dtype=np.float32),
        scale=2.4,
        pad_x=0,
        pad_y=120,
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
    anchors = generate_anchors(480)

    assert anchors.shape == (29070, 4)
    assert np.isclose(anchors[0, 2] / anchors[0, 3], 0.15)
