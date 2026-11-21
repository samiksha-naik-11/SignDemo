import base64
import binascii
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Preprocessing.background_removal import remove_background
from Preprocessing.low_light_detection import prepare_frame
from Preprocessing.test_hand import MODEL_PATH, draw_hand_landmarks


MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 720
_landmarker = None
_landmarker_lock = threading.RLock()
_last_timestamp_ms = 0
STATIC_FILES = {
    "/": ROOT / "index.html",
    "/index.html": ROOT / "index.html",
    "/styles.css": ROOT / "styles.css",
    "/app.js": ROOT / "app.js",
}


def decode_image_data(image_data: str) -> np.ndarray:
    header, separator, encoded = image_data.partition(",")
    if (
        not separator
        or not header.startswith("data:image/")
        or ";base64" not in header
    ):
        raise ValueError("Expected a base64-encoded image.")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("The uploaded image could not be decoded.") from error

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("The uploaded file is not a readable image.")

    height, width = frame.shape[:2]
    scale = min(MAX_FRAME_WIDTH / width, MAX_FRAME_HEIGHT / height, 1)
    if scale < 1:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def get_landmarker():
    global _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            if not MODEL_PATH.is_file():
                raise RuntimeError(
                    f"Hand landmark model not found: {MODEL_PATH}"
                )
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _landmarker = vision.HandLandmarker.create_from_options(options)
        return _landmarker


def encode_image(image: np.ndarray, image_format: str) -> str:
    success, encoded = cv2.imencode(image_format, image)
    if not success:
        raise RuntimeError("Unable to encode the processed camera frame.")
    mime_type = "image/png" if image_format == ".png" else "image/jpeg"
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def process_frame(frame: np.ndarray, remove_hand_background: bool) -> dict:
    global _last_timestamp_ms

    prepared, brightness, lighting_status = prepare_frame(frame)
    rgb_frame = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    with _landmarker_lock:
        _last_timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            _last_timestamp_ms + 1,
        )
        detection = get_landmarker().detect_for_video(image, _last_timestamp_ms)

    landmarks = detection.hand_landmarks[0] if detection.hand_landmarks else None
    result = {
        "hand_detected": landmarks is not None,
        "brightness": round(brightness, 1),
        "low_light": lighting_status == "Low Lighting",
        "lighting_status": lighting_status,
        "processed_image": None,
        "background_removed": False,
    }

    if remove_hand_background and landmarks:
        rgba = remove_background(prepared, landmarks)
        result["processed_image"] = encode_image(rgba, ".png")
        result["background_removed"] = True
    elif landmarks:
        preview = prepared.copy()
        draw_hand_landmarks(preview, landmarks)
        result["processed_image"] = encode_image(preview, ".jpg")

    return result


class SignBridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.send_json(
                200,
                {
                    "ok": True,
                    "hand_detection": "ready" if MODEL_PATH.is_file() else "model_missing",
                    "background_removal": "ready",
                    "translation": "not_configured",
                },
            )
            return
        static_path = STATIC_FILES.get(path)
        if static_path is None:
            self.send_json(404, {"error": "Resource not found."})
            return

        body = static_path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(str(static_path))[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/process-frame":
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid request length."})
            return

        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "Image is empty or larger than 8 MB."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
                raise ValueError("An image is required.")
            frame = decode_image_data(payload["image"])
            result = process_frame(frame, payload.get("remove_background") is True)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.send_json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self.send_json(503, {"error": str(error)})
            return
        except Exception as error:
            self.log_error("Frame processing failed: %s", error)
            self.send_json(500, {"error": "Frame processing failed. Check the server terminal."})
            return

        self.send_json(200, result)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), SignBridgeHandler)
    print("SignBridge is running at http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SignBridge server.")
    finally:
        server.server_close()
        with _landmarker_lock:
            if _landmarker is not None:
                _landmarker.close()


if __name__ == "__main__":
    main()
