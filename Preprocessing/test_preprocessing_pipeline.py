import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

try:
    from . import low_light_detection as pipeline
except ImportError:
    import low_light_detection as pipeline


class PreprocessingPipelineTests(unittest.TestCase):
    def test_prepare_frame_enhances_only_low_light_images(self) -> None:
        dark_frame = np.full((8, 8, 3), 30, dtype=np.uint8)
        enhanced, brightness, status = pipeline.prepare_frame(dark_frame)

        self.assertEqual(brightness, 30)
        self.assertEqual(status, "Low Lighting")
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(dark_frame)))

        normal_frame = np.full((8, 8, 3), 160, dtype=np.uint8)
        unchanged, brightness, status = pipeline.prepare_frame(normal_frame)

        self.assertEqual(brightness, 160)
        self.assertEqual(status, "Normal Lighting")
        np.testing.assert_array_equal(unchanged, normal_frame)

    def test_save_hand_png_writes_transparent_png(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        expected = np.zeros((2, 2, 4), dtype=np.uint8)
        expected[:, :, 3] = 255

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "hand.png"
            with patch.object(pipeline, "remove_background", return_value=expected):
                pipeline.save_hand_png(frame, [], output_path)

            saved = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

        self.assertIsNotNone(saved)
        self.assertEqual(saved.shape[2], 4)
        np.testing.assert_array_equal(saved, expected)

    def test_save_webcam_capture_writes_full_frame_png(self) -> None:
        frame = np.full((2, 3, 3), (20, 100, 220), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "webcam.png"
            pipeline.save_webcam_capture(frame, output_path)
            saved = cv2.imread(str(output_path))

        self.assertIsNotNone(saved)
        np.testing.assert_array_equal(saved, frame)

    def test_save_webcam_capture_rejects_empty_frame(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.save_webcam_capture(np.empty((0, 0, 3), dtype=np.uint8))

    def test_background_removal_keeps_landmark_seeds(self) -> None:
        frame = np.full((100, 100, 3), (20, 40, 60), dtype=np.uint8)
        cv2.rectangle(frame, (35, 30), (65, 80), (100, 170, 210), -1)
        pixel_points = [(50, 30), (40, 42), (60, 42), (50, 60), (50, 78)]
        landmarks = [
            SimpleNamespace(x=x / 100, y=y / 100)
            for x, y in pixel_points
        ]

        result = pipeline.remove_background(frame, landmarks)

        self.assertEqual(result.shape, (100, 100, 4))
        for x, y in pixel_points:
            self.assertGreater(result[y, x, 3], 0)


if __name__ == "__main__":
    unittest.main()
