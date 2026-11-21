from pathlib import Path
from time import sleep
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")






def open_camera():

    import time

    # Try MSMF first, then DirectShow
    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DSHOW", cv2.CAP_DSHOW)
    ]

    for name, backend in backends:

        print("Trying camera with:", name)

        cap = cv2.VideoCapture(0, backend)

        if not cap.isOpened():
            print(name, "could not open camera")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for camera initialization
        time.sleep(2)

        # Try reading camera frames
        for attempt in range(30):

            success, frame = cap.read()

            if success and frame is not None and frame.size > 0:

                print("SignBridge camera ready using", name)

                return cap, frame

            time.sleep(0.1)

        print(name, "opened but could not capture frames")

        cap.release()

    print("ERROR: No camera backend could capture frames!")

    return None, None




def ensure_model() -> None:
    if MODEL_PATH.exists():
        return

    print("Downloading MediaPipe hand landmark model...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(
            f"Unable to download the hand landmark model to {MODEL_PATH}: {error}"
        ) from error


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]

    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (0, 255, 0),
            2,
        )

    for point in points:
        cv2.circle(frame, point, 4, (255, 0, 0), -1)

    x_values, y_values = zip(*points)
    x_min = max(0, min(x_values))
    x_max = min(width - 1, max(x_values))
    y_min = max(0, min(y_values))
    y_max = min(height - 1, max(y_values))

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Hand Detected",
        (x_min, max(25, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


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

    cap, frame = open_camera()
    if cap is None:
        raise RuntimeError(
            "The camera did not provide any images. Close other camera apps, "
            "allow camera access in Windows Settings > Privacy & security > "
            "Camera (including desktop apps), and test the camera in the "
            "Windows Camera app."
        )

    timestamp_ms = 0
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                if frame is None:
                    success, frame = cap.read()

                    if not success or frame is None:
                        print("Camera frame failed. Reconnecting...")

                        cap.release()
                        sleep(1)

                        cap, frame = open_camera()

                        if cap is None:
                            print("Camera reconnection failed")
                            break

                        continue

                # Mirror camera
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms += 33

                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        draw_hand_landmarks(frame, hand_landmarks)
                else:
                    cv2.putText(
                        frame,
                        "No Hand Detected",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("SignBridge Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame = None
    finally:
        if cap is not None:
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
