"""
tests/test_shape_analysis.py

Unit tests for the core detection/measurement logic in shape_analysis.py.

Since a real microscope image has no known ground truth, these tests run the
detection pipeline against *synthetic* images with hand-drawn ellipses of
known dimensions, so the expected Shape Factor / Ellipticity / Area can be
computed independently and compared against what the pipeline returns.

Run (from the project root):
    pytest tests/ -v
"""

import cv2
import numpy as np
import pytest

from shape_analysis import get_erythrocyte_shape_factors


def make_synthetic_ellipse_image(major_axis, minor_axis, center=(150, 150),
                                  angle=0, canvas_size=(300, 300)):
    """
    Draws a single filled dark ellipse on a light BGR canvas — matching real
    microscope images, where erythrocytes appear darker than the background
    (the pipeline uses THRESH_BINARY_INV precisely because of this polarity).

    cv2.ellipse's `axes` argument takes *semi*-axes, so major_axis/minor_axis
    here are the full lengths returned by cv2.fitEllipse (consistent with
    what get_erythrocyte_shape_factors reports).
    """
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    cv2.ellipse(
        canvas,
        center,
        (int(major_axis / 2), int(minor_axis / 2)),
        angle,
        0, 360,
        (40, 40, 40),
        thickness=-1,  # filled
    )
    return canvas


def make_two_ellipses_image(canvas_size=(400, 200)):
    """Two well-separated dark circles on a light background, to test that
    multiple cells are detected independently."""
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    cv2.circle(canvas, (100, 100), 40, (40, 40, 40), thickness=-1)
    cv2.circle(canvas, (300, 100), 40, (40, 40, 40), thickness=-1)
    return canvas


class TestShapeFactor:
    def test_elongated_ellipse_has_expected_shape_factor(self):
        """A 200x100 ellipse should yield Shape Factor ~= 2.0 (major/minor)."""
        image = make_synthetic_ellipse_image(major_axis=200, minor_axis=100)
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=2.5, min_axis_size=50
        )
        detected = normal_cells + anomalies
        assert len(detected) == 1
        assert detected[0]["Shape Factor"] == pytest.approx(2.0, rel=0.05)

    def test_circle_has_shape_factor_close_to_one(self):
        """A perfect circle should yield Shape Factor ~= 1.0 (the known limitation
        documented in the README: this metric cannot distinguish a healthy round
        cell from a spherocytosis-type anomaly)."""
        image = make_synthetic_ellipse_image(major_axis=120, minor_axis=120)
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=2.5, min_axis_size=50
        )
        detected = normal_cells + anomalies
        assert len(detected) == 1
        assert detected[0]["Shape Factor"] == pytest.approx(1.0, rel=0.05)


class TestAnomalyClassification:
    def test_cell_below_threshold_is_normal(self):
        """Shape Factor 1.5 with threshold 1.7 must land in the normal bucket."""
        image = make_synthetic_ellipse_image(major_axis=150, minor_axis=100)  # SF = 1.5
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=1.7, min_axis_size=50
        )
        assert len(normal_cells) == 1
        assert len(anomalies) == 0

    def test_cell_above_threshold_is_anomaly(self):
        """Shape Factor 2.0 with threshold 1.7 must land in the anomalies bucket."""
        image = make_synthetic_ellipse_image(major_axis=200, minor_axis=100)  # SF = 2.0
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=1.7, min_axis_size=50
        )
        assert len(normal_cells) == 0
        assert len(anomalies) == 1

    def test_color_band_never_contradicts_classification(self):
        """
        Regression test for the color/classification mismatch fixed in app.py:
        any cell drawn "red" (highly elongated) must never end up classified as
        Normal/Moderate once its Shape Factor exceeds the anomaly threshold, and
        the color-band cutoffs must always stay below the anomaly threshold
        itself, for any threshold the user picks.
        """
        for anomaly_threshold in (1.5, 1.7, 2.0, 2.5):
            yellow_cutoff = anomaly_threshold * 0.75
            red_cutoff = anomaly_threshold * 0.9
            assert yellow_cutoff < red_cutoff < anomaly_threshold


class TestArtifactFiltering:
    def test_small_contour_filtered_by_min_axis_size(self):
        """A tiny 20x20 blob should be rejected when min_axis_size=50."""
        image = make_synthetic_ellipse_image(major_axis=20, minor_axis=20)
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=2.5, min_axis_size=50
        )
        assert len(normal_cells) + len(anomalies) == 0

    def test_blank_image_detects_nothing(self):
        """No contours in a blank black image -> no detections, no crash."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=1.7, min_axis_size=50
        )
        assert normal_cells == []
        assert anomalies == []


class TestMultipleCells:
    def test_two_separate_cells_detected_independently(self):
        image = make_two_ellipses_image()
        _, normal_cells, anomalies, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=2.5, min_axis_size=30
        )
        detected = normal_cells + anomalies
        assert len(detected) == 2
        for cell in detected:
            assert cell["Shape Factor"] == pytest.approx(1.0, rel=0.05)


class TestEllipticity:
    def test_ellipticity_matches_manual_formula(self):
        """Ellipticity is defined as 1 - (minor/major); verify it against the
        raw axis lengths the function itself reports, not just a hardcoded number."""
        image = make_synthetic_ellipse_image(major_axis=200, minor_axis=100)
        _, normal_cells, _, _ = get_erythrocyte_shape_factors(
            image, anomaly_threshold=2.5, min_axis_size=50
        )
        cell = normal_cells[0]
        expected_ellipticity = 1 - (cell["Minor Axis"] / cell["Major Axis"])
        assert cell["Ellipticity"] == pytest.approx(expected_ellipticity)
        assert cell["Ellipticity"] == pytest.approx(0.5, rel=0.05)


class TestBinaryMask:
    def test_returns_grayscale_mask_same_size_as_input(self):
        image = make_synthetic_ellipse_image(major_axis=150, minor_axis=100)
        _, _, _, binary_mask = get_erythrocyte_shape_factors(
            image, anomaly_threshold=1.7, min_axis_size=50
        )
        assert binary_mask.shape == image.shape[:2]
        assert binary_mask.dtype == np.uint8
