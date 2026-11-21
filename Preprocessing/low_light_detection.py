from pathlib import Path
from time import sleep

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .background_removal import remove_background
    from .test_hand import (
        MODEL_PATH,
        draw_hand_landmarks,
        ensure_model,
        open_camera,
    )
except ImportError:
    from background_removal import remove_background
    from test_hand import (
        MODEL_PATH,
        draw_hand_landmarks,
        ensure_model,
        open_camera,
    )


LOW_LIGHT_THRESHOLD = 80
OUTPUT_PATH = Path(__file__).resolve().parent / "processed_hand.png"
WEBCAM_OUTPUT_PATH = Path(__file__).resolve().parent / "webcam_capture.png"


def detect_low_light(frame: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    status = (
        "Low Lighting"
        if brightness < LOW_LIGHT_THRESHOLD
        else "Normal Lighting"
    )
    return brightness, status


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    gamma = 1.5
    lookup_table = np.array(
        [((value / 255.0) ** (1 / gamma)) * 255 for value in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(frame, lookup_table)


def prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, float, str]:
    brightness, status = detect_low_light(frame)
    processed = enhance_low_light(frame) if status == "Low Lighting" else frame.copy()
    return processed, brightness, status


def save_hand_png(
    frame: np.ndarray,
    hand_landmarks: list,
    output_path: Path = OUTPUT_PATH,
) -> np.ndarray:
    result = remove_background(frame, hand_landmarks)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Unable to save processed hand image: {output_path}")
    return result


def save_webcam_capture(
    frame: np.ndarray,
    output_path: Path = WEBCAM_OUTPUT_PATH,
) -> None:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot save an empty webcam frame.")
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Unable to save webcam image: {output_path}")


def _show_saved_image(result: np.ndarray) -> None:
    alpha = result[:, :, 3:4].astype(np.float32) / 255.0
    preview = (
        result[:, :, :3].astype(np.float32) * alpha
        + 255 * (1 - alpha)
    ).astype(np.uint8)
    cv2.imshow("Background Removed", preview)


def main() -> None:
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap, _ = open_camera()
    if cap is None or not cap.isOpened():
        raise RuntimeError(
            "The camera did not provide any images. Close other camera apps, "
            "allow camera access in Windows Settings > Privacy & security > "
            "Camera (including desktop apps), and test the camera in the "
            "Windows Camera app."
        )

    timestamp_ms = 0
    missed_frames = 0
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                if not success or frame is None or frame.size == 0:
                    missed_frames += 1
                    if missed_frames >= 10:
                        cap.release()
                        sleep(1)
                        cap, _ = open_camera()
                        if cap is None or not cap.isOpened():
                            raise RuntimeError(
                                "Camera reconnection failed. Please check "
                                "your webcam."
                            )
                        missed_frames = 0
                    else:
                        sleep(0.1)
                    continue

                missed_frames = 0
                frame = cv2.flip(frame, 1)
                processed, brightness, status = prepare_frame(frame)

                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )
                timestamp_ms += 33
                detection = landmarker.detect_for_video(image, timestamp_ms)

                display = processed.copy()
                if detection.hand_landmarks:
                    for hand_landmarks in detection.hand_landmarks:
                        draw_hand_landmarks(display, hand_landmarks)
                    prompt = "Hand detected | C: capture image, S: save hand PNG"
                else:
                    prompt = "No hand detected | C: capture image"

                cv2.putText(
                    display,
                    f"{status} | Brightness: {brightness:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255) if status == "Low Lighting" else (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    prompt,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("SignBridge Hand Preprocessing", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    try:
                        save_webcam_capture(processed)
                    except (cv2.error, ValueError, RuntimeError) as error:
                        print(f"Unable to capture webcam image: {error}")
                    else:
                        print(f"Webcam image saved as PNG: {WEBCAM_OUTPUT_PATH}")

                if key == ord("s"):
                    if not detection.hand_landmarks:
                        print("No hand detected. Try again.")
                        continue

                    try:
                        result = save_hand_png(
                            processed,
                            detection.hand_landmarks[0],
                        )
                    except (cv2.error, ValueError, RuntimeError) as error:
                        print(f"Unable to process hand image: {error}")
                    else:
                        print(f"Processed hand saved as PNG: {OUTPUT_PATH}")
                        _show_saved_image(result)

                if key == ord("q"):
                    break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
