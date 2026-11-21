from __future__ import annotations

import hashlib
import threading
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from threading import Thread
from typing import Protocol, TypedDict
from urllib.parse import quote

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NEIGHBOR_COUNT = 5
MINIMUM_MARGIN = 0.5
MINIMUM_VOTE_SHARE = 0.6


class LandmarkPoint(Protocol):
    x: float
    y: float
    z: float


class SignEntry(TypedDict):
    label: str
    title: str
    frames: list[str]


def _image_files(root: Path) -> list[tuple[str, Path]]:
    files = []
    if not root.is_dir():
        return files

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        label = folder.name.lower()
        files.extend(
            (label, path)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return files


def build_sign_catalog(root: Path) -> list[SignEntry]:
    catalog: list[SignEntry] = []
    for label, path in _image_files(root):
        if not catalog or catalog[-1]["label"] != label:
            catalog.append(
                {"label": label, "title": label.replace("_", " ").title(), "frames": []}
            )
        catalog[-1]["frames"].append(
            f"/dataset/{quote(label, safe='')}/{quote(path.name, safe='')}"
        )
    return catalog


def extract_landmark_features(
    hands: Sequence[Sequence[LandmarkPoint]],
) -> np.ndarray | None:
    if not hands:
        return None

    hand_points = []
    for hand in hands[:2]:
        if len(hand) != 21:
            raise ValueError("Each hand must contain exactly 21 landmarks.")
        hand_points.append(
            np.asarray(
                [(point.x, point.y, point.z) for point in hand],
                dtype=np.float32,
            )
        )
    hand_points.sort(key=lambda points: float(points[:, 0].mean()))

    all_points = np.concatenate(hand_points, axis=0)
    lower = all_points[:, :2].min(axis=0)
    upper = all_points[:, :2].max(axis=0)
    scale = max(float((upper - lower).max()), 0.02)
    center = (lower + upper) / 2

    features = np.zeros((2, 21, 3), dtype=np.float32)
    for index, points in enumerate(hand_points):
        normalized = points.copy()
        normalized[:, :2] = (normalized[:, :2] - center) / scale
        normalized[:, 2] /= scale
        features[index] = normalized
    return features.reshape(-1)


def match_landmark_features(
    query: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    if features.ndim != 2 or features.shape[1] != query.size:
        raise ValueError("Dataset features do not match the query shape.")
    if labels.ndim != 1 or labels.size != features.shape[0] or labels.size == 0:
        raise ValueError("Dataset labels do not match the available features.")

    distances = np.mean((features - query) ** 2, axis=1)
    neighbor_count = min(NEIGHBOR_COUNT, len(labels))
    nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    votes = Counter(str(labels[index]) for index in nearest)

    class_distances = {
        str(label): float(
            np.mean(
                np.sort(distances[labels == label])[
                    : min(3, int(np.count_nonzero(labels == label)))
                ]
            )
        )
        for label in np.unique(labels)
    }
    ranked_labels = sorted(class_distances, key=lambda label: class_distances[label])
    candidate = ranked_labels[0]
    best_distance = class_distances[candidate]
    second_distance = (
        class_distances[ranked_labels[1]]
        if len(ranked_labels) > 1
        else float("inf")
    )
    margin = max(0.0, 1.0 - best_distance / max(second_distance, 1e-9))
    vote_share = votes[candidate] / neighbor_count
    match_score = round(100 * (margin + vote_share) / 2)
    accepted = margin >= MINIMUM_MARGIN and vote_share >= MINIMUM_VOTE_SHARE

    return {
        "state": "ready",
        "recognized": accepted,
        "label": candidate if accepted else None,
        "candidate": candidate,
        "match_score": match_score,
    }


class DatasetLandmarkRecognizer:
    def __init__(self, dataset_root: Path, cache_path: Path, model_path: Path) -> None:
        self.dataset_root = dataset_root
        self.cache_path = cache_path
        self.model_path = model_path
        self.catalog = build_sign_catalog(dataset_root)
        self._files = _image_files(dataset_root)
        self._frame_paths = {
            f"/dataset/{quote(label, safe='')}/{quote(path.name, safe='')}": path
            for label, path in self._files
        }
        self._lock = threading.Lock()
        self._state = "not_started"
        self._error: str | None = None
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._state != "not_started":
                return
            self._state = "loading"
            self._thread = Thread(target=self._load_features, daemon=True)
            self._thread.start()

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state,
                "sign_count": len(self.catalog),
                "error": self._error,
            }

    def classify(
        self, hands: Sequence[Sequence[LandmarkPoint]]
    ) -> dict[str, object]:
        query = extract_landmark_features(hands)
        if query is None:
            return {
                "state": "no_hand",
                "recognized": False,
                "label": None,
                "candidate": None,
                "match_score": 0,
            }

        with self._lock:
            state = self._state
            features = self._features
            labels = self._labels
        if state != "ready" or features is None or labels is None:
            return {
                "state": state,
                "recognized": False,
                "label": None,
                "candidate": None,
                "match_score": 0,
            }
        return match_landmark_features(query, features, labels)

    def frame_path(self, url_path: str) -> Path | None:
        return self._frame_paths.get(url_path)

    def _dataset_signature(self) -> str:
        digest = hashlib.sha256()
        for label, path in self._files:
            stat = path.stat()
            digest.update(
                f"{label}\0{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    def _read_cache(self, signature: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.cache_path.is_file():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                if str(cached["signature"].item()) != signature:
                    return None
                features = cached["features"]
                labels = cached["labels"].astype(str)
        except (OSError, ValueError, KeyError) as error:
            print(f"Sign feature cache is invalid; rebuilding it: {error}")
            return None
        if (
            features.ndim != 2
            or features.shape[1] != 126
            or features.shape[0] != labels.size
        ):
            print("Sign feature cache has inconsistent dimensions; rebuilding it.")
            return None
        return features, labels

    def _build_features(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._files:
            raise RuntimeError(f"No sign images found in {self.dataset_root}.")
        if not self.model_path.is_file():
            raise RuntimeError(f"Hand landmark model not found: {self.model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        features = []
        labels = []
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            for label, path in self._files:
                frame = cv2.imread(str(path))
                if frame is None:
                    print(f"Skipping unreadable sign dataset image: {path}")
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                feature = extract_landmark_features(result.hand_landmarks)
                if feature is None:
                    print(f"Skipping dataset image without detected hands: {path}")
                    continue
                features.append(feature)
                labels.append(label)

        if not features:
            raise RuntimeError("No hand landmarks could be extracted from the dataset.")
        return np.vstack(features), np.asarray(labels, dtype=str)

    def _load_features(self) -> None:
        try:
            signature = self._dataset_signature()
            cached = self._read_cache(signature)
            if cached is None:
                print("Preparing the sign dataset for recognition...")
                features, labels = self._build_features()
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.cache_path.with_name(
                    f"{self.cache_path.stem}.tmp.npz"
                )
                np.savez_compressed(
                    temporary_path,
                    signature=np.asarray(signature),
                    features=features,
                    labels=labels,
                )
                temporary_path.replace(self.cache_path)
            else:
                features, labels = cached

            with self._lock:
                self._features = features
                self._labels = labels
                self._state = "ready"
        except Exception as error:
            with self._lock:
                self._error = str(error)
                self._state = "error"
            print(f"Sign dataset recognition could not start: {error}")
