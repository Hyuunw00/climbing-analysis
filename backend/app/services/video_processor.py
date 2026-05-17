import logging
from collections import Counter

import cv2
import numpy as np
from dataclasses import dataclass

from app.config import settings
from app.services.hold_detector import hold_detector
from app.services.path_predictor import predict_next_holds
from app.services.force_estimator import estimate_force_distribution
from app.services.performance_scorer import compute_frame_metrics
from app.services.pose_estimator import PoseEstimator


# ── Hold color utilities ─────────────────────────────────────────────

# HSV-based color classification
_COLOR_RANGES: list[tuple[str, tuple, tuple]] = [
    ("red",    (0, 80, 80),    (10, 255, 255)),
    ("red2",   (170, 80, 80),  (180, 255, 255)),  # red wraps around
    ("orange", (10, 80, 80),   (25, 255, 255)),
    ("yellow", (25, 80, 80),   (35, 255, 255)),
    ("green",  (35, 80, 80),   (85, 255, 255)),
    ("blue",   (85, 80, 80),   (130, 255, 255)),
    ("purple", (130, 80, 80),  (160, 255, 255)),
    ("pink",   (160, 80, 80),  (170, 255, 255)),
    ("white",  (0, 0, 180),    (180, 40, 255)),
    ("black",  (0, 0, 0),      (180, 80, 80)),
]


def _classify_color(hsv_roi: np.ndarray) -> str:
    """Classify a region's dominant color from HSV values."""
    if hsv_roi.size == 0:
        return "unknown"

    for name, lower, upper in _COLOR_RANGES:
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        ratio = np.count_nonzero(mask) / max(mask.size, 1)
        if ratio > 0.3:
            return name.replace("2", "")  # merge red2 → red

    return "unknown"


def _extract_hold_color(frame_bgr: np.ndarray, hold: dict, img_w: int, img_h: int) -> str:
    """Extract dominant color from a hold's bounding box region."""
    # Convert normalized coords back to pixels
    cx = int(hold["x"] * img_w)
    cy = int(hold["y"] * img_h)
    w = int(hold["width"] * img_w)
    h = int(hold["height"] * img_h)

    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(img_w, cx + w // 2)
    y2 = min(img_h, cy + h // 2)

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return _classify_color(hsv)

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


def process_video(file_path: str, on_progress: callable = None) -> dict:
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
    hand_proximity_colors: list[str] = []
    total_frames = len(frame_indices)

    for idx, (frame_number, timestamp) in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()

        # Report progress
        if on_progress and total_frames > 0:
            on_progress((idx + 1) / total_frames * 100)

        pose_keypoints = []
        detected_holds = []
        if ret and frame_bgr is not None:
            # MediaPipe expects RGB input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            keypoints = estimator.estimate(frame_rgb)
            pose_keypoints = [kp.model_dump() for kp in keypoints]

            # Hold detection (operates on BGR)
            raw_holds = hold_detector.detect(frame_bgr)
            img_h, img_w = frame_bgr.shape[:2]
            for i, h in enumerate(raw_holds):
                # Normalize pixel coords to 0-1 (matching pose coords)
                hold_data = {
                    "id": f"hold_{frame_number}_{i}",
                    "x": h["x"] / img_w,
                    "y": h["y"] / img_h,
                    "width": h["width"] / img_w,
                    "height": h["height"] / img_h,
                    "confidence": h["confidence"],
                }
                hold_data["color"] = _extract_hold_color(
                    frame_bgr, hold_data, img_w, img_h
                )
                detected_holds.append(hold_data)

        # Collect hand-proximity colors for route detection (pass 1)
        if pose_keypoints and detected_holds:
            from app.models.schemas import PoseKeypoint as PK
            kps = [PK(**kp) for kp in pose_keypoints]
            lwrist = next((k for k in kps if k.name == "left_wrist" and k.confidence >= 0.3), None)
            rwrist = next((k for k in kps if k.name == "right_wrist" and k.confidence >= 0.3), None)

            for wrist in [lwrist, rwrist]:
                if not wrist:
                    continue
                for h in detected_holds:
                    dist = ((wrist.x - h["x"]) ** 2 + (wrist.y - h["y"]) ** 2) ** 0.5
                    if dist < 0.08:
                        c = h.get("color", "unknown")
                        if c != "unknown":
                            hand_proximity_colors.append(c)

        # Force distribution estimation
        force_dist = None
        if pose_keypoints:
            from app.models.schemas import PoseKeypoint as PK
            kps_for_force = [PK(**kp) for kp in pose_keypoints]
            force_dist = estimate_force_distribution(kps_for_force)

        frame_results.append(
            {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "holds": detected_holds,
                "pose": pose_keypoints,
                "predictions": [],  # filled in pass 2
                "force_distribution": force_dist,
            }
        )

    cap.release()
    estimator.close()

    # Pass 2: determine route color from all hand-proximity samples
    route_color = None
    if hand_proximity_colors:
        route_color = Counter(hand_proximity_colors).most_common(1)[0][0]
        logger.info("Detected route color: %s (%d samples)", route_color, len(hand_proximity_colors))

    # Filter holds + compute predictions per frame
    for frame in frame_results:
        if route_color:
            frame["holds"] = [h for h in frame["holds"] if h.get("color") == route_color]
        frame["route_color"] = route_color

        if frame["pose"] and frame["holds"]:
            from app.models.schemas import PoseKeypoint as PK
            kps = [PK(**kp) for kp in frame["pose"]]
            raw_preds = predict_next_holds(kps, frame["holds"])
            frame["predictions"] = [p.model_dump() for p in raw_preds]

    # Pass 3: compute performance metrics across frame sequence
    effective_fps = metadata.fps / max(1, int(metadata.fps / (settings.frame_rate or metadata.fps)))
    metrics_list = compute_frame_metrics(frame_results, effective_fps)
    for frame, m in zip(frame_results, metrics_list):
        frame["metrics"] = {
            "smoothness": m.smoothness,
            "technique": m.technique,
            "stable": m.stable,
            "adhesion": m.adhesion,
            "score": m.score,
            "vertical_velocity": m.vertical_velocity,
        }

    return {
        "metadata": {
            "duration": metadata.duration,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "route_color": route_color,
        },
        "frames": frame_results,
    }
