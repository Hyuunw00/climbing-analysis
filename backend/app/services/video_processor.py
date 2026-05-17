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
from app.models.schemas import PoseKeypoint as PK

logger = logging.getLogger(__name__)


# ── Hold color utilities ─────────────────────────────────────────────

# HSV ranges: H=0-180, S=0-255, V=0-255 (OpenCV convention)
_COLOR_RANGES: list[tuple[str, tuple, tuple]] = [
    ("red",      (0, 70, 70),     (8, 255, 255)),
    ("red2",     (170, 70, 70),   (180, 255, 255)),
    ("orange",   (8, 70, 70),     (20, 255, 255)),
    ("yellow",   (20, 70, 70),    (35, 255, 255)),
    ("lime",     (35, 70, 70),    (50, 255, 255)),
    ("green",    (50, 70, 70),    (75, 255, 255)),
    ("mint",     (75, 40, 70),    (95, 255, 255)),
    ("blue",     (95, 70, 70),    (130, 255, 255)),
    ("purple",   (130, 50, 70),   (155, 255, 255)),
    ("pink",     (155, 40, 70),   (170, 255, 255)),
    ("white",    (0, 0, 200),     (180, 30, 255)),
    ("gray",     (0, 0, 100),     (180, 30, 200)),
    ("black",    (0, 0, 0),       (180, 50, 100)),
    ("brown",    (8, 30, 50),     (25, 70, 150)),
    ("beige",    (15, 20, 150),   (35, 70, 230)),
]

_COLOR_ALIASES = {"red2": "red", "lime": "green"}


def _classify_color(hsv_roi: np.ndarray) -> str:
    """Classify dominant color from center 50% of HSV region (avoids edges/shadows)."""
    if hsv_roi.size == 0:
        return "unknown"

    # Use center 50% to avoid edge shadows
    h, w = hsv_roi.shape[:2]
    margin_y, margin_x = h // 4, w // 4
    center = hsv_roi[margin_y:h - margin_y, margin_x:w - margin_x]
    if center.size == 0:
        center = hsv_roi

    best_name = "unknown"
    best_ratio = 0.15

    for name, lower, upper in _COLOR_RANGES:
        mask = cv2.inRange(center, np.array(lower), np.array(upper))
        ratio = np.count_nonzero(mask) / max(mask.size, 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = _COLOR_ALIASES.get(name, name)

    return best_name


def _extract_hold_color(frame_bgr: np.ndarray, cx_px: int, cy_px: int, w_px: int, h_px: int) -> str:
    """Extract dominant color from a hold's bounding box center region."""
    img_h, img_w = frame_bgr.shape[:2]
    x1 = max(0, cx_px - w_px // 2)
    y1 = max(0, cy_px - h_px // 2)
    x2 = min(img_w, cx_px + w_px // 2)
    y2 = min(img_h, cy_px + h_px // 2)

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return _classify_color(hsv)


# ── Hold map: merge detections across frames ────────────────────────

def _merge_hold_detections(
    all_detections: list[list[dict]],
    merge_dist: float = 0.05,
) -> list[dict]:
    """Merge hold detections across multiple frames into a stable hold map.

    Holds detected at similar positions in different frames are merged.
    Returns deduplicated holds with averaged positions and best confidence.
    """
    merged: list[dict] = []

    for frame_holds in all_detections:
        for h in frame_holds:
            # Find existing merged hold close enough
            found = False
            for m in merged:
                dist = ((m["x"] - h["x"]) ** 2 + (m["y"] - h["y"]) ** 2) ** 0.5
                if dist < merge_dist:
                    # Update with running average
                    n = m["count"]
                    m["x"] = (m["x"] * n + h["x"]) / (n + 1)
                    m["y"] = (m["y"] * n + h["y"]) / (n + 1)
                    m["width"] = (m["width"] * n + h["width"]) / (n + 1)
                    m["height"] = (m["height"] * n + h["height"]) / (n + 1)
                    m["confidence"] = max(m["confidence"], h["confidence"])
                    m["count"] += 1
                    # Accumulate color votes
                    if h.get("color") and h["color"] != "unknown":
                        m["color_votes"].append(h["color"])
                    found = True
                    break
            if not found:
                merged.append({
                    "x": h["x"],
                    "y": h["y"],
                    "width": h["width"],
                    "height": h["height"],
                    "confidence": h["confidence"],
                    "count": 1,
                    "color_votes": [h["color"]] if h.get("color") and h["color"] != "unknown" else [],
                })

    # Finalize: assign IDs, determine color by majority vote
    result: list[dict] = []
    for i, m in enumerate(merged):
        if m["count"] < 1:
            continue
        color = "unknown"
        if m["color_votes"]:
            color = Counter(m["color_votes"]).most_common(1)[0][0]
        result.append({
            "id": f"hold_{i}",
            "x": round(m["x"], 5),
            "y": round(m["y"], 5),
            "width": round(m["width"], 5),
            "height": round(m["height"], 5),
            "confidence": round(m["confidence"], 3),
            "color": color,
        })

    return result


# ── Video metadata ──────────────────────────────────────────────────

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
    """Extract frame indices at the configured frame rate."""
    fps = target_fps or settings.frame_rate
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if video_fps <= 0:
        return []

    interval = max(1, int(video_fps / fps))
    frames: list[tuple[int, float]] = []
    for i in range(0, frame_count, interval):
        timestamp = i / video_fps
        frames.append((i, timestamp))

    return frames


# ── Training data collection ───────────────────────────────────────

_TRAINING_DATA_DIR = "data/processed"
_TRAINING_DATA_FILE = "user_training_data.jsonl"


def _collect_training_samples(frame_results: list[dict]) -> None:
    """Extract move events from analyzed frames and append to training data file."""
    import json, os
    from app.ml.labeler import detect_move_events
    from app.ml.features import extract_features

    events = detect_move_events(frame_results)
    if not events:
        return

    os.makedirs(_TRAINING_DATA_DIR, exist_ok=True)
    filepath = os.path.join(_TRAINING_DATA_DIR, _TRAINING_DATA_FILE)

    samples_written = 0
    with open(filepath, "a") as f:
        for event in events:
            frame = frame_results[event.frame_idx]
            pose = frame.get("pose", [])
            holds = frame.get("holds", [])
            if not pose or not holds:
                continue

            kps = [PK(**kp) if isinstance(kp, dict) else kp for kp in pose]
            for hold in holds:
                feats = extract_features(kps, holds, event.hand, hold)
                row = feats.copy()
                row["label"] = 1.0 if hold.get("id") == event.target_hold_id else 0.0
                row["hand"] = event.hand
                row["target_hold_id"] = event.target_hold_id
                row["candidate_hold_id"] = hold.get("id", "")
                f.write(json.dumps(row) + "\n")
                samples_written += 1

    if samples_written > 0:
        logger.info("Collected %d training samples from %d move events", samples_written, len(events))


# ── Main pipeline ───────────────────────────────────────────────────

# Number of frames to scan for building the hold map
_HOLDMAP_SCAN_FRAMES = 20


def process_video(file_path: str, on_progress: callable = None) -> dict:
    """Full video processing pipeline.

    1. Scan initial frames to build a stable hold map (merged across frames)
    2. Determine route color from climber hand proximity
    3. Filter holds by route color
    4. Run per-frame pose + path prediction + force estimation
    5. Compute performance metrics
    """
    metadata = extract_metadata(file_path)
    if metadata is None:
        return {"error": "Failed to read video"}

    frame_indices = extract_frames(file_path)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {"error": "Failed to open video for processing"}

    img_h, img_w = metadata.height, metadata.width
    total_frames = len(frame_indices)

    # ── Phase 1: Build hold map from scan frames ────────────────────
    scan_indices = frame_indices[:_HOLDMAP_SCAN_FRAMES]
    if len(frame_indices) > _HOLDMAP_SCAN_FRAMES:
        # Also sample some frames from middle and end
        mid = len(frame_indices) // 2
        extra = frame_indices[mid:mid + 5] + frame_indices[-5:]
        scan_indices = scan_indices + extra

    all_hold_detections: list[list[dict]] = []
    scan_frames_bgr: dict[int, np.ndarray] = {}

    for frame_number, _ in scan_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        scan_frames_bgr[frame_number] = frame_bgr
        raw_holds = hold_detector.detect(frame_bgr)
        frame_holds = []
        for h in raw_holds:
            hold_data = {
                "x": h["x"] / img_w,
                "y": h["y"] / img_h,
                "width": h["width"] / img_w,
                "height": h["height"] / img_h,
                "confidence": h["confidence"],
            }
            hold_data["color"] = _extract_hold_color(
                frame_bgr,
                int(h["x"]), int(h["y"]),
                int(h["width"]), int(h["height"]),
            )
            frame_holds.append(hold_data)

        all_hold_detections.append(frame_holds)

    # Merge all detections into stable hold map
    hold_map = _merge_hold_detections(all_hold_detections)
    logger.info("Hold map: %d stable holds from %d scan frames", len(hold_map), len(scan_indices))

    # ── Phase 2: Determine route color from early contacts ──────────
    # Scan early frames, collect the color of the closest hold to each hand,
    # then vote — the most frequent contact color = route color.
    estimator = PoseEstimator()
    route_color = None
    contact_colors: list[str] = []

    for frame_number, _ in frame_indices[:40]:  # scan first ~20 seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints = estimator.estimate(frame_rgb)
        if not keypoints:
            continue

        lwrist = next((k for k in keypoints if k.name == "left_wrist" and k.confidence >= 0.3), None)
        rwrist = next((k for k in keypoints if k.name == "right_wrist" and k.confidence >= 0.3), None)

        for wrist in [lwrist, rwrist]:
            if not wrist:
                continue
            # Find the single closest hold to this hand
            best_dist = float("inf")
            best_color = None
            for h in hold_map:
                dist = ((wrist.x - h["x"]) ** 2 + (wrist.y - h["y"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_color = h.get("color", "unknown")
            # Only count if very close (touching)
            if best_dist < 0.05 and best_color and best_color != "unknown":
                contact_colors.append(best_color)

    if contact_colors:
        route_color = Counter(contact_colors).most_common(1)[0][0]
        logger.info("Route color from contacts: %s (votes: %s)", route_color, Counter(contact_colors).most_common(3))

    # Filter hold map by route color
    route_holds = hold_map
    if route_color:
        route_holds = [h for h in hold_map if h.get("color") == route_color]
        logger.info("Route holds: %d/%d (color=%s)", len(route_holds), len(hold_map), route_color)

    # ── Phase 3: Per-frame analysis with fixed hold map ─────────────
    frame_results: list[dict] = []

    for idx, (frame_number, timestamp) in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()

        if on_progress and total_frames > 0:
            on_progress((idx + 1) / total_frames * 100)

        pose_keypoints = []
        if ret and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            keypoints = estimator.estimate(frame_rgb)
            pose_keypoints = [kp.model_dump() for kp in keypoints]

        # Path prediction with fixed route holds
        predictions = []
        if pose_keypoints and route_holds:
            kps = [PK(**kp) for kp in pose_keypoints]
            raw_preds = predict_next_holds(kps, route_holds)
            predictions = [p.model_dump() for p in raw_preds]

        # Force distribution
        force_dist = None
        if pose_keypoints:
            kps_for_force = [PK(**kp) for kp in pose_keypoints]
            force_dist = estimate_force_distribution(kps_for_force)

        frame_results.append({
            "frame_number": frame_number,
            "timestamp": timestamp,
            "holds": route_holds,  # Same holds for every frame
            "pose": pose_keypoints,
            "predictions": predictions,
            "force_distribution": force_dist,
            "route_color": route_color,
        })

    cap.release()
    estimator.close()

    # ── Phase 4: Performance metrics ────────────────────────────────
    effective_fps = metadata.fps / max(1, int(metadata.fps / (settings.frame_rate or metadata.fps)))
    metrics_list = compute_frame_metrics(frame_results, effective_fps, img_height=metadata.height)
    for frame, m in zip(frame_results, metrics_list):
        frame["metrics"] = {
            "smoothness": m.smoothness,
            "technique": m.technique,
            "stable": m.stable,
            "adhesion": m.adhesion,
            "score": m.score,
            "vertical_velocity": m.vertical_velocity,
        }

    # ── Phase 5: Collect training data from user uploads ──────────
    _collect_training_samples(frame_results)

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
