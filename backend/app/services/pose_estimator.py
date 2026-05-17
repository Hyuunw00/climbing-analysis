"""Pose estimation service using MediaPipe Pose."""

import logging

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.models.schemas import PoseKeypoint

logger = logging.getLogger(__name__)

# Landmark names (33 landmarks)
_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Model download URL
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
_MODEL_PATH = "models/pose_landmarker_heavy.task"


def _ensure_model() -> str | None:
    """Download pose landmarker model if not present."""
    import os
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    try:
        import urllib.request
        logger.info("Downloading pose landmarker model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model downloaded to %s", _MODEL_PATH)
        return _MODEL_PATH
    except Exception as e:
        logger.error("Failed to download pose model: %s", e)
        return None


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for single-frame inference.

    Initializes the model once and reuses across frames.
    """

    def __init__(self) -> None:
        model_path = _ensure_model()
        self._available = False

        if model_path is None:
            logger.warning("Pose estimation unavailable — model not found")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._available = True
        except Exception as e:
            logger.error("Failed to initialize PoseLandmarker: %s", e)

    def estimate(self, frame_rgb: np.ndarray) -> list[PoseKeypoint]:
        """Run pose estimation on a single RGB frame.

        Args:
            frame_rgb: numpy array in RGB format (H, W, 3).

        Returns:
            List of 33 PoseKeypoint items, or empty list if detection fails.
        """
        if not self._available:
            return []

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return []

            landmarks = result.pose_landmarks[0]
            keypoints: list[PoseKeypoint] = []
            for idx, landmark in enumerate(landmarks):
                name = _LANDMARK_NAMES[idx] if idx < len(_LANDMARK_NAMES) else f"landmark_{idx}"
                keypoints.append(
                    PoseKeypoint(
                        name=name,
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility,
                        confidence=landmark.visibility,
                    )
                )
            return keypoints
        except Exception as e:
            logger.error("Pose estimation failed: %s", e)
            return []

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._available:
            self._landmarker.close()

    def __del__(self) -> None:
        self.close()
