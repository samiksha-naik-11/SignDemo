import os, time, cv2
import numpy as np
from collections import deque
from predictor import predict, MODE
from tts import Speaker

# ---------- settings ----------
CONF_THRESHOLD = 0.70    # minimum confidence to accept a prediction
SMOOTH_FRAMES  = 8       # same label this many frames in a row
PREDICT_EVERY  = 2       # run the model every Nth frame (speed)
EMOJI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emoji")

speaker = Speaker()
speech_on = True
history = deque(maxlen=SMOOTH_FRAMES)
stable_label = ""          # currently confirmed sign
sentence = []              # words confirmed so far
last_added = ""
latency_ms = 0.0
label, conf = None, 0.0
frame_no = 0

def load_emoji(name):
    path = os.path.join(EMOJI_DIR, name.lower().replace(" ", "_") + ".png")
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return cv2.resize(img, (64, 64)) if img is not None else None
    return None

def overlay(frame, img, x, y):
    """Draw a PNG (with or without alpha) on the frame."""
    h, w = img.shape[:2]
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return
    if img.shape[2] == 4:
        a = img[:, :, 3:4] / 255.0
        frame[y:y+h, x:x+w] = (img[:, :, :3] * a + frame[y:y+h, x:x+w] * (1 - a)).astype("uint8")
    else:
        frame[y:y+h, x:x+w] = img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not found. Try VideoCapture(1).")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame_no += 1

    # 1) predict (every Nth frame, measure latency)
    if frame_no % PREDICT_EVERY == 0:
        t0 = time.perf_counter()
        label, conf = predict(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        # 2) threshold
        accepted = label if (label is not None and conf >= CONF_THRESHOLD) else None
        history.append(accepted)

        # 3) smoothing: whole window must agree
        if len(history) == SMOOTH_FRAMES:
            if len(set(history)) == 1 and history[0] is not None:
                stable_label = history[0]
            elif all(h is None for h in history):
                stable_label = ""     # nothing recognized / no hand

            # 4) add to sentence + speak only when the sign changes
            if stable_label and stable_label != last_added:
                sentence.append(stable_label)
                last_added = stable_label
                if speech_on:
                    speaker.say(stable_label)
            if not stable_label:
                last_added = ""       # allow the same sign again after a gap

    # ---------- draw UI ----------
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 110), (30, 30, 30), -1)
    shown = stable_label if stable_label else "Sign not recognized"
    colour = (0, 255, 0) if stable_label else (0, 0, 255)
    cv2.putText(frame, f"Sign: {shown}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    cv2.putText(frame, f"Confidence: {conf*100:.0f}%", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"{MODE} | {latency_ms:.0f} ms | Speech {'ON' if speech_on else 'OFF'}",
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if stable_label:
        em = load_emoji(stable_label)
        if em is not None:
            overlay(frame, em, w - 80, 20)

    # sentence bar at the bottom
    cv2.rectangle(frame, (0, h - 45), (w, h), (30, 30, 30), -1)
    text = " ".join(sentence[-6:])
    cv2.putText(frame, f"Text: {text}", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "s=speech  c=clear  q=quit", (w - 260, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("SignBridge", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        speech_on = not speech_on
    elif key == ord('c'):
        sentence.clear(); last_added = ""

cap.release()
cv2.destroyAllWindows()