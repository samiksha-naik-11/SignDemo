import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .sign_dataset import (
        build_sign_catalog,
        extract_landmark_features,
        match_landmark_features,
    )
except ImportError:
    from sign_dataset import (
        build_sign_catalog,
        extract_landmark_features,
        match_landmark_features,
    )


class SignDatasetTests(unittest.TestCase):
    def test_catalog_lists_ordered_image_frames_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sign_folder = root / "washroom"
            sign_folder.mkdir()
            (sign_folder / "02.jpg").touch()
            (sign_folder / "01.jpg").touch()
            (sign_folder / "notes.txt").touch()

            catalog = build_sign_catalog(root)

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["label"], "washroom")
        self.assertEqual(catalog[0]["title"], "Washroom")
        self.assertEqual(
            catalog[0]["frames"],
            ["/dataset/washroom/01.jpg", "/dataset/washroom/02.jpg"],
        )

    def test_landmark_features_are_hand_order_invariant(self) -> None:
        first_hand = [
            SimpleNamespace(x=(index % 5) / 10, y=(index // 5) / 10, z=index / 100)
            for index in range(21)
        ]
        second_hand = [
            SimpleNamespace(x=0.6 + (index % 5) / 10, y=(index // 5) / 10, z=index / 100)
            for index in range(21)
        ]

        forward = extract_landmark_features([first_hand, second_hand])
        reverse = extract_landmark_features([second_hand, first_hand])

        self.assertIsNotNone(forward)
        self.assertEqual(forward.shape, (126,))
        np.testing.assert_array_equal(forward, reverse)

    def test_match_rejects_an_ambiguous_out_of_dataset_query(self) -> None:
        features = np.vstack(
            [
                np.zeros((5, 126), dtype=np.float32),
                np.ones((5, 126), dtype=np.float32),
            ]
        )
        labels = np.asarray(["help"] * 5 + ["water"] * 5)

        recognized = match_landmark_features(
            np.zeros(126, dtype=np.float32), features, labels
        )
        unknown = match_landmark_features(
            np.full(126, 0.5, dtype=np.float32), features, labels
        )

        self.assertEqual(recognized["label"], "help")
        self.assertTrue(recognized["recognized"])
        self.assertIsNone(unknown["label"])
        self.assertFalse(unknown["recognized"])


if __name__ == "__main__":
    unittest.main()
