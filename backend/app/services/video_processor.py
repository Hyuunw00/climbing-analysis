import logging

import cv2
from dataclasses import dataclass

from app.config import settings
from app.services.hold_detector import hold_detector
from app.services.pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int


def extract_metadata(file_path: str) -> VideoMetadata | None:
    """Extract video metadata using OpenCV."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def extract_frames(file_path: str, target_fps: int | None = None) -> list[tuple[int, float]]:
    """
    Extract frames at the configured frame rate.
    Returns list of (frame_number, timestamp) tuples.
    Actual frame data processing is a stub for now.
    """
    fps = target_fps or settings.frame_rate
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if video_fps <= 0:
        return []

    # Calculate which frames to sample
    interval = max(1, int(video_fps / fps))
    frames: list[tuple[int, float]] = []

    for i in range(0, frame_count, interval):
        timestamp = i / video_fps
        frames.append((i, timestamp))

    return frames


def process_video(file_path: str) -> dict:
    """
    Full video processing pipeline.
    Runs pose estimation on sampled frames using MediaPipe.
    """
    metadata = extract_metadata(file_path)
    if metadata is None:
        return {"error": "Failed to read video"}

    frame_indices = extract_frames(file_path)

    # Open video and run pose estimation on each sampled frame
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {"error": "Failed to open video for processing"}

    estimator = PoseEstimator()
    frame_results: list[dict] = []

    for frame_number, timestamp in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()

        pose_keypoints = []
        detected_holds = []
        if ret and frame_bgr is not None:
            # MediaPipe expects RGB input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            keypoints = estimator.estimate(frame_rgb)
            pose_keypoints = [kp.model_dump() for kp in keypoints]

            # Hold detection (operates on BGR)
            raw_holds = hold_detector.detect(frame_bgr)
            for i, h in enumerate(raw_holds):
                detected_holds.append(
                    {
                        "id": f"hold_{frame_number}_{i}",
                        "x": h["x"],
                        "y": h["y"],
                        "width": h["width"],
                        "height": h["height"],
                        "confidence": h["confidence"],
                    }
                )

        frame_results.append(
            {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "holds": detected_holds,
                "pose": pose_keypoints,
                "predictions": [],
            }
        )

    cap.release()
    estimator.close()

    return {
        "metadata": {
            "duration": metadata.duration,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
        },
        "frames": frame_results,
    }
