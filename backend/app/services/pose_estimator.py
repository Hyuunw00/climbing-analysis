"""Pose estimation service using MediaPipe Pose."""

import numpy as np
import mediapipe as mp

from app.models.schemas import PoseKeypoint


# MediaPipe landmark indices for hands and feet
_HAND_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]  # wrists, pinkies, indexes, thumbs
_FOOT_INDICES = [27, 28, 29, 30, 31, 32]  # ankles, heels, foot indexes


class PoseEstimator:
    """Wraps MediaPipe Pose for single-frame inference.

    Initializes the model once and reuses across frames.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmark_names: list[str] = [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]

    def estimate(self, frame_rgb: np.ndarray) -> list[PoseKeypoint]:
        """Run pose estimation on a single RGB frame.

        Args:
            frame_rgb: numpy array in RGB format (H, W, 3).

        Returns:
            List of 33 PoseKeypoint items, or empty list if detection fails.
        """
        results = self._pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return []

        keypoints: list[PoseKeypoint] = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append(
                PoseKeypoint(
                    name=self._landmark_names[idx],
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                    confidence=landmark.visibility,
                )
            )
        return keypoints

    def extract_hands(self, keypoints: list[PoseKeypoint]) -> list[PoseKeypoint]:
        """Extract hand/wrist keypoints from full pose."""
        if not keypoints:
            return []
        return [keypoints[i] for i in _HAND_INDICES if i < len(keypoints)]

    def extract_feet(self, keypoints: list[PoseKeypoint]) -> list[PoseKeypoint]:
        """Extract foot/ankle keypoints from full pose."""
        if not keypoints:
            return []
        return [keypoints[i] for i in _FOOT_INDICES if i < len(keypoints)]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    def __del__(self) -> None:
        self.close()
