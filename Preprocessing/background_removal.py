

import cv2
import numpy as np


def remove_background(frame, hand_landmarks):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image")
    if not hand_landmarks:
        raise ValueError("At least one hand landmark is required")

    height, width = frame.shape[:2]
    hands = (
        [hand_landmarks]
        if hasattr(hand_landmarks[0], "x")
        else hand_landmarks
    )
    hand_points = [
        np.array(
            [
                (
                    np.clip(int(landmark.x * width), 0, width - 1),
                    np.clip(int(landmark.y * height), 0, height - 1),
                )
                for landmark in hand
            ],
            dtype=np.int32,
        )
        for hand in hands
        if hand
    ]
    if not hand_points:
        raise ValueError("At least one hand landmark is required")

    points = np.concatenate(hand_points, axis=0)
    x_values = points[:, 0]
    y_values = points[:, 1]
    hand_width = int(x_values.max() - x_values.min())
    hand_height = int(y_values.max() - y_values.min())
    if hand_width < 2 or hand_height < 2:
        raise ValueError("Hand landmarks do not span a valid image region")

    padding = max(20, int(max(hand_width, hand_height) * 0.18))
    x_min = max(0, int(x_values.min()) - padding)
    y_min = max(0, int(y_values.min()) - padding)
    x_max = min(width, int(x_values.max()) + padding + 1)
    y_max = min(height, int(y_values.max()) + padding + 1)
    roi = frame[y_min:y_max, x_min:x_max]

    # Use landmark locations as definite foreground seeds and the hand hull
    # as probable foreground so GrabCut can keep the full hand silhouette.
    mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    seed_radius = max(2, int(min(hand_width, hand_height) * 0.025))
    roi_origin = np.array([x_min, y_min], dtype=np.int32)
    for hand in hand_points:
        roi_points = hand - roi_origin
        hull = cv2.convexHull(roi_points)
        cv2.fillConvexPoly(mask, hull, cv2.GC_PR_FGD)
        for point in roi_points:
            cv2.circle(
                mask,
                tuple(point),
                seed_radius,
                cv2.GC_FGD,
                thickness=-1,
            )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        roi,
        mask,
        None,
        bg_model,
        fg_model,
        7,
        cv2.GC_INIT_WITH_MASK,
    )

    hand_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    hand_mask = cv2.morphologyEx(
        hand_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    hand_mask = cv2.GaussianBlur(hand_mask, (3, 3), 0)

    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[y_min:y_max, x_min:x_max, :3] = roi
    result[y_min:y_max, x_min:x_max, 3] = hand_mask

    return result


def main() -> None:
    try:
        from .low_light_detection import main as run_preprocessing
    except ImportError:
        from low_light_detection import main as run_preprocessing

    run_preprocessing()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped.")
