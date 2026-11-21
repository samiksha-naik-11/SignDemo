import os, sys, json, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ---------- DUMMY (works today) ----------
DUMMY_LABELS = ["Hello", "Help", "Thank You", "Yes", "No"]
_state = {"label": "Hello", "count": 0}

def _dummy_predict(frame):
    # changes label every ~60 frames so you can test smoothing and speech
    _state["count"] += 1
    if _state["count"] % 60 == 0:
        _state["label"] = random.choice(DUMMY_LABELS)
    return _state["label"], random.uniform(0.75, 0.98)

# ---------- REAL MODEL (edit when teammates finish) ----------
_model = None
_labels = None

def _load_real():
    global _model, _labels
    import tensorflow as tf
    _model = tf.keras.models.load_model(os.path.join(ROOT, "signbridge_model.keras"))
    _labels = json.load(open(os.path.join(ROOT, "labels.json")))

def _real_predict(frame):
    from preprocessing.preprocess import preprocess_hand   # Samiksha's function
    img = preprocess_hand(frame)                           # 224x224x3, 0-1, or None
    if img is None:
        return None, 0.0
    probs = _model.predict(img[np.newaxis, ...], verbose=0)[0]
    i = int(np.argmax(probs))
    return _labels[str(i)], float(probs[i])

# ---------- what app.py calls ----------
USE_REAL = os.path.exists(os.path.join(ROOT, "signbridge_model.keras"))
if USE_REAL:
    _load_real()

def predict(frame):
    """Returns (label or None, confidence 0-1). None = no hand found."""
    return _real_predict(frame) if USE_REAL else _dummy_predict(frame)

MODE = "REAL MODEL" if USE_REAL else "DUMMY MODE"