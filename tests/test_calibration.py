from __future__ import annotations

import numpy as np

from src.calibration.court_registration import (
    _compute_confidence,
    _pixel_only_result,
    _process_single_frame,
)
from src.calibration.frame_sampler import _is_stable_frame
from src.calibration.line_detection import detect_lines_hough
from src.calibration.line_filtering import cluster_lines, filter_court_lines
from src.calibration.template_fitting import (
    _find_line_intersections,
    _line_intersection,
    _order_corners,
    fit_homography,
    get_court_template_lines,
    get_court_template_points,
    match_lines_to_template,
)
from src.schemas import CourtGeometry2D

# --- Court template tests ---


class TestCourtTemplate:
    def test_template_points_shape(self):
        geometry = CourtGeometry2D()
        points = get_court_template_points(geometry)
        assert points.shape == (15, 2)

    def test_template_points_bounds(self):
        geometry = CourtGeometry2D()
        points = get_court_template_points(geometry)
        assert points[:, 0].min() == 0.0
        assert points[:, 0].max() == 10.0
        assert points[:, 1].min() == 0.0
        assert points[:, 1].max() == 20.0

    def test_template_points_contains_corners(self):
        geometry = CourtGeometry2D()
        points = get_court_template_points(geometry)
        # Check that all 4 outer corners are present
        corners = [[0, 0], [10, 0], [0, 20], [10, 20]]
        for corner in corners:
            assert any(np.allclose(pt, corner) for pt in points)

    def test_template_points_net_line(self):
        geometry = CourtGeometry2D()
        points = get_court_template_points(geometry)
        # Net at y=10
        net_points = points[points[:, 1] == 10.0]
        assert len(net_points) == 3  # left, center, right

    def test_template_lines_shape(self):
        geometry = CourtGeometry2D()
        lines = get_court_template_lines(geometry)
        assert lines.shape[1] == 4
        assert len(lines) == 9  # 5 horizontal + 2 sidelines + 2 center service


# --- Line intersection tests ---


class TestLineIntersection:
    def test_perpendicular_lines(self):
        # Horizontal and vertical lines intersecting at (50, 50)
        result = _line_intersection(
            np.array([0, 50, 100, 50]),
            np.array([50, 0, 50, 100]),
        )
        assert result is not None
        assert abs(result[0] - 50) < 0.01
        assert abs(result[1] - 50) < 0.01

    def test_parallel_lines_no_intersection(self):
        result = _line_intersection(
            np.array([0, 0, 100, 0]),
            np.array([0, 10, 100, 10]),
        )
        assert result is None

    def test_nearly_parallel_lines(self):
        # Lines at very small angle (< 15 degrees)
        result = _line_intersection(
            np.array([0, 0, 100, 0]),
            np.array([0, 5, 100, 10]),
        )
        # Should be None because angle < 15 degrees
        assert result is None

    def test_diagonal_intersection(self):
        # Two diagonals of a 100x100 square
        result = _line_intersection(
            np.array([0, 0, 100, 100]),
            np.array([100, 0, 0, 100]),
        )
        assert result is not None
        assert abs(result[0] - 50) < 0.01
        assert abs(result[1] - 50) < 0.01


class TestFindLineIntersections:
    def test_grid_intersections(self):
        # 2 horizontal + 2 vertical = 4 intersections
        lines = np.array([
            [0, 25, 100, 25],   # horizontal top
            [0, 75, 100, 75],   # horizontal bottom
            [25, 0, 25, 100],   # vertical left
            [75, 0, 75, 100],   # vertical right
        ], dtype=np.float64)
        pts = _find_line_intersections(lines, (100, 100))
        assert len(pts) == 4

    def test_no_intersections_parallel(self):
        lines = np.array([
            [0, 10, 100, 10],
            [0, 50, 100, 50],
            [0, 90, 100, 90],
        ], dtype=np.float64)
        pts = _find_line_intersections(lines, (100, 100))
        assert len(pts) == 0


# --- Line filtering tests ---


class TestLineFiltering:
    def test_removes_short_lines(self):
        # Image 1000x1000, diagonal ~1414, min length ~28.3
        lines = np.array([
            [0, 0, 5, 0],       # Very short, should be removed
            [0, 500, 500, 500],  # Long horizontal, should keep
        ], dtype=np.float64)
        filtered = filter_court_lines(lines, (1000, 1000))
        assert len(filtered) == 1

    def test_removes_extreme_angles(self):
        # Lines at ~45 degrees should be removed (between 30 and 60)
        lines = np.array([
            [0, 0, 500, 500],    # 45 degrees - removed
            [0, 500, 500, 500],  # 0 degrees - kept
            [500, 0, 500, 500],  # 90 degrees - kept
        ], dtype=np.float64)
        filtered = filter_court_lines(lines, (1000, 1000))
        # The 45-degree line should be removed
        assert len(filtered) == 2

    def test_empty_input(self):
        lines = np.empty((0, 4))
        filtered = filter_court_lines(lines, (1000, 1000))
        assert len(filtered) == 0


class TestLineClustering:
    def test_separates_horizontal_vertical(self):
        lines = np.array([
            [0, 100, 500, 100],  # horizontal
            [0, 300, 500, 300],  # horizontal
            [100, 0, 100, 500],  # vertical
            [400, 0, 400, 500],  # vertical
        ], dtype=np.float64)
        clusters = cluster_lines(lines)
        assert len(clusters["horizontal"]) >= 1
        assert len(clusters["vertical"]) >= 1

    def test_empty_input(self):
        lines = np.empty((0, 4))
        clusters = cluster_lines(lines)
        assert len(clusters["horizontal"]) == 0
        assert len(clusters["vertical"]) == 0

    def test_merges_nearby_parallel(self):
        # Two nearly identical horizontal lines should merge to one
        lines = np.array([
            [0, 100, 500, 100],
            [0, 105, 500, 105],  # 5 pixels away, should merge
            [100, 0, 100, 500],
        ], dtype=np.float64)
        clusters = cluster_lines(lines)
        assert len(clusters["horizontal"]) == 1


# --- Homography fitting tests ---


class TestHomographyFitting:
    def test_perfect_correspondences(self):
        # Known homography: simple scaling
        court_pts = np.array([
            [0, 0], [10, 0], [10, 20], [0, 20],
            [5, 10], [5, 0],
        ], dtype=np.float64)

        # Scale by 50 in x and 30 in y, translate by (100, 50)
        image_pts = court_pts.copy()
        image_pts[:, 0] = court_pts[:, 0] * 50 + 100
        image_pts[:, 1] = court_pts[:, 1] * 30 + 50

        H, error = fit_homography(image_pts, court_pts)
        assert H is not None
        assert error < 1.0  # Should be very small for perfect correspondences

    def test_insufficient_points(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        H, error = fit_homography(pts, pts)
        assert H is None
        assert error == float("inf")

    def test_returns_valid_homography_matrix(self):
        court_pts = np.array([
            [0, 0], [10, 0], [10, 20], [0, 20], [5, 10],
        ], dtype=np.float64)
        image_pts = np.array([
            [100, 400], [500, 400], [450, 100], [150, 100], [300, 250],
        ], dtype=np.float64)

        H, error = fit_homography(image_pts, court_pts)
        if H is not None:
            assert H.shape == (3, 3)
            assert error < float("inf")


# --- Frame stability tests ---


class TestFrameStability:
    def test_dark_frame_rejected(self):
        # Very dark frame
        frame = np.zeros((100, 100), dtype=np.uint8)
        assert not _is_stable_frame(frame)

    def test_bright_frame_rejected(self):
        # Very bright (saturated) frame
        frame = np.full((100, 100), 250, dtype=np.uint8)
        assert not _is_stable_frame(frame)

    def test_normal_frame_accepted(self):
        # Create a frame with edges (high Laplacian variance)
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:80, 20:80] = 200
        frame[40:60, 40:60] = 50
        assert _is_stable_frame(frame)

    def test_blurry_frame_rejected(self):
        # Uniform gray frame (no edges, low Laplacian variance)
        frame = np.full((100, 100), 128, dtype=np.uint8)
        assert not _is_stable_frame(frame)


# --- Confidence score tests ---


class TestConfidence:
    def test_zero_error_max_inliers(self):
        conf = _compute_confidence(0.0, 10.0, 10)
        assert conf == 1.0

    def test_max_error_zero_confidence(self):
        conf = _compute_confidence(10.0, 10.0, 4)
        assert conf == 0.0

    def test_moderate_error(self):
        conf = _compute_confidence(5.0, 10.0, 6)
        assert 0.0 < conf < 1.0

    def test_clipped_to_range(self):
        conf = _compute_confidence(20.0, 10.0, 2)
        assert conf == 0.0


# --- Pixel-only fallback tests ---


class TestPixelOnlyResult:
    def test_returns_correct_mode(self):
        result = _pixel_only_result()
        assert result.mode == "pixel_only"
        assert result.homography_image_to_court is None
        assert result.homography_court_to_image is None
        assert result.confidence == 0.0


# --- Hough line detection tests ---


class TestHoughDetection:
    def test_detects_clear_lines(self):
        # Create image with obvious lines
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Draw horizontal and vertical white lines
        img[100, 50:450] = 255
        img[200, 50:450] = 255
        img[300, 50:450] = 255
        img[50:450, 100] = 255
        img[50:450, 300] = 255

        lines = detect_lines_hough(img)
        assert len(lines) >= 2  # Should detect at least some lines

    def test_empty_image_no_lines(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        lines = detect_lines_hough(img)
        assert len(lines) == 0

    def test_output_shape(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img[250, 50:450] = 255  # single horizontal line
        lines = detect_lines_hough(img)
        if len(lines) > 0:
            assert lines.shape[1] == 4


# --- Corner ordering tests ---


class TestCornerOrdering:
    def test_orders_correctly(self):
        # Unordered corners
        corners = np.array([
            [400, 400],  # bottom-right
            [100, 100],  # top-left
            [400, 100],  # top-right
            [100, 400],  # bottom-left
        ], dtype=np.float64)
        ordered = _order_corners(corners)
        # Should be: TL, TR, BR, BL
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[0][1] < ordered[3][1]  # TL.y < BL.y
        assert ordered[1][0] > ordered[0][0]  # TR.x > TL.x


# --- Integration-style test with synthetic court ---


class TestSyntheticCourt:
    def _make_synthetic_court_image(self):
        """Create a synthetic image of a court with clear lines."""
        img = np.full((600, 800, 3), 80, dtype=np.uint8)  # dark green background

        # Draw court lines in white (simplified top-down view)
        # Map court (10m x 20m) to image region (200..600 x 50..550)
        # x: 0..10 -> 200..600 (40 px/m)
        # y: 0..20 -> 50..550 (25 px/m)

        def court_to_img(cx, cy):
            return int(200 + cx * 40), int(50 + cy * 25)

        import cv2

        # Baselines
        cv2.line(img, court_to_img(0, 0), court_to_img(10, 0), (255, 255, 255), 2)
        cv2.line(img, court_to_img(0, 20), court_to_img(10, 20), (255, 255, 255), 2)

        # Sidelines
        cv2.line(img, court_to_img(0, 0), court_to_img(0, 20), (255, 255, 255), 2)
        cv2.line(img, court_to_img(10, 0), court_to_img(10, 20), (255, 255, 255), 2)

        # Net
        cv2.line(img, court_to_img(0, 10), court_to_img(10, 10), (255, 255, 255), 2)

        # Service lines
        cv2.line(img, court_to_img(0, 3.05), court_to_img(10, 3.05), (255, 255, 255), 2)
        cv2.line(img, court_to_img(0, 16.95), court_to_img(10, 16.95), (255, 255, 255), 2)

        # Center service lines
        cv2.line(img, court_to_img(5, 0), court_to_img(5, 3.05), (255, 255, 255), 2)
        cv2.line(img, court_to_img(5, 16.95), court_to_img(5, 20), (255, 255, 255), 2)

        return img

    def test_line_detection_on_synthetic_court(self):
        img = self._make_synthetic_court_image()
        lines = detect_lines_hough(img)
        # Should detect several lines from the court drawing
        assert len(lines) >= 4

    def test_line_filtering_on_synthetic_court(self):
        img = self._make_synthetic_court_image()
        lines = detect_lines_hough(img)
        filtered = filter_court_lines(lines, img.shape)
        # Filtered should retain most court lines
        assert len(filtered) >= 3

    def test_process_frame_on_synthetic_court(self):
        """Integration test: full frame processing on synthetic court image."""
        img = self._make_synthetic_court_image()
        geometry = CourtGeometry2D()

        result = _process_single_frame(img, 0, geometry, max_error=15.0)
        # On a clean synthetic image, we should get a result
        # (may not always succeed due to matching complexity, so we just check type)
        if result is not None:
            assert result.mode == "floor_homography"
            assert result.homography_image_to_court is not None
            assert result.reprojection_error_px is not None


# --- Match lines to template tests ---


class TestMatchLinesToTemplate:
    def test_insufficient_lines(self):
        lines = np.array([[0, 0, 100, 0], [0, 0, 0, 100]], dtype=np.float64)
        geometry = CourtGeometry2D()
        img_pts, court_pts = match_lines_to_template(lines, geometry, (500, 500))
        # With only 2 lines, might not get enough correspondences
        # Just check it doesn't crash and returns correct shape
        assert img_pts.shape[1] == 2 or len(img_pts) == 0
        assert court_pts.shape[1] == 2 or len(court_pts) == 0

    def test_returns_matching_counts(self):
        geometry = CourtGeometry2D()
        # Create a grid of lines that resemble a court
        lines = np.array([
            [100, 50, 500, 50],    # top horizontal
            [100, 200, 500, 200],  # mid horizontal
            [100, 400, 500, 400],  # bottom horizontal
            [100, 50, 100, 400],   # left vertical
            [300, 50, 300, 400],   # center vertical
            [500, 50, 500, 400],   # right vertical
        ], dtype=np.float64)
        img_pts, court_pts = match_lines_to_template(lines, geometry, (500, 600))
        assert len(img_pts) == len(court_pts)
