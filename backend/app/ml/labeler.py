"""Auto-labeling pipeline for ML path prediction training data.

Processes climbing videos through the existing pipeline (hold detection + pose estimation)
and detects "move events" — moments when a hand moves from one hold to another.
Each move event produces a training sample: (pose, holds, target_hold, hand).
"""

import logging
from dataclasses import dataclass

import cv2

from app.config import settings
from app.services.hold_detector import hold_detector
from app.services.pose_estimator import PoseEstimator
from app.services.video_processor import (
    _merge_hold_detections,
    _extract_hold_color,
    extract_frames,
    extract_metadata,
    _HOLDMAP_SCAN_FRAMES,
)
from app.models.schemas import PoseKeypoint

logger = logging.getLogger(__name__)

# Move event detection thresholds (normalized coords)
_MOVE_THRESHOLD = 0.04      # wrist displacement to count as "moving"
_STABLE_THRESHOLD = 0.02    # wrist displacement to count as "stabilized"
_CONTACT_THRESHOLD = 0.05   # max distance to nearest hold for contact


@dataclass
class MoveEvent:
    frame_idx: int        # index in frame_sequence (before the move)
    hand: str             # "left" or "right"
    target_hold_id: str   # hold the hand moved to


@dataclass
class TrainingSample:
    pose: list[dict]      # PoseKeypoint dicts at source frame
    holds: list[dict]     # all route holds
    target_hold_id: str   # which hold was grabbed
    hand: str             # "left" or "right"


def detect_move_events(
    frame_sequence: list[dict],
) -> list[MoveEvent]:
    """Detect hand move events from a sequence of analyzed frames.

    A move event is detected when:
    1. Wrist displacement between frame t and t+1 > MOVE_THRESHOLD
    2. Wrist then stabilizes (displacement < STABLE_THRESHOLD in next frame)
    3. Stabilized wrist is within CONTACT_THRESHOLD of a hold

    Args:
        frame_sequence: List of frame dicts with 'pose' and 'holds' fields.

    Returns:
        List of MoveEvent objects.
    """
    if len(frame_sequence) < 3:
        return []

    events: list[MoveEvent] = []

    # Extract wrist positions per frame
    wrist_positions: list[dict[str, tuple[float, float] | None]] = []
    for frame in frame_sequence:
        pose = frame.get("pose", [])
        positions: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        for kp_data in pose:
            kp = kp_data if isinstance(kp_data, dict) else kp_data.model_dump() if hasattr(kp_data, "model_dump") else {}
            name = kp.get("name", "")
            conf = kp.get("confidence", 0.0)
            if conf < 0.3:
                continue
            if name == "left_wrist":
                positions["left"] = (kp["x"], kp["y"])
            elif name == "right_wrist":
                positions["right"] = (kp["x"], kp["y"])
        wrist_positions.append(positions)

    holds = frame_sequence[0].get("holds", [])

    for hand in ["left", "right"]:
        for i in range(len(frame_sequence) - 2):
            pos_prev = wrist_positions[i].get(hand)
            pos_curr = wrist_positions[i + 1].get(hand)
            pos_next = wrist_positions[i + 2].get(hand)

            if not pos_prev or not pos_curr or not pos_next:
                continue

            # Check movement: prev → curr
            disp_move = ((pos_curr[0] - pos_prev[0]) ** 2 + (pos_curr[1] - pos_prev[1]) ** 2) ** 0.5
            if disp_move < _MOVE_THRESHOLD:
                continue

            # Check stabilization: curr → next
            disp_stable = ((pos_next[0] - pos_curr[0]) ** 2 + (pos_next[1] - pos_curr[1]) ** 2) ** 0.5
            if disp_stable > _STABLE_THRESHOLD:
                continue

            # Find nearest hold to stabilized position
            best_hold_id = None
            best_dist = float("inf")
            for h in holds:
                dist = ((pos_next[0] - h["x"]) ** 2 + (pos_next[1] - h["y"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_hold_id = h.get("id")

            if best_dist < _CONTACT_THRESHOLD and best_hold_id:
                events.append(MoveEvent(
                    frame_idx=i,
                    hand=hand,
                    target_hold_id=best_hold_id,
                ))

    return events


def label_video(video_path: str) -> list[TrainingSample]:
    """Process a video and extract labeled training samples.

    Runs hold detection + pose estimation (skipping force/metrics),
    then detects move events to produce training data.

    Args:
        video_path: Path to the video file.

    Returns:
        List of TrainingSample objects.
    """
    metadata = extract_metadata(video_path)
    if metadata is None:
        logger.warning("Failed to read video: %s", video_path)
        return []

    frame_indices = extract_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    img_h, img_w = metadata.height, metadata.width

    # ── Phase 1: Build hold map ──
    scan_indices = frame_indices[:_HOLDMAP_SCAN_FRAMES]
    if len(frame_indices) > _HOLDMAP_SCAN_FRAMES:
        mid = len(frame_indices) // 2
        extra = frame_indices[mid:mid + 5] + frame_indices[-5:]
        scan_indices = scan_indices + extra

    all_hold_detections: list[list[dict]] = []

    for frame_number, _ in scan_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

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
                frame_bgr, int(h["x"]), int(h["y"]),
                int(h["width"]), int(h["height"]),
            )
            frame_holds.append(hold_data)
        all_hold_detections.append(frame_holds)

    hold_map = _merge_hold_detections(all_hold_detections)
    if not hold_map:
        cap.release()
        logger.warning("No holds detected in %s", video_path)
        return []

    logger.info("Label pipeline: %d holds from %s", len(hold_map), video_path)

    # ── Phase 2: Pose estimation per frame ──
    estimator = PoseEstimator()
    frame_sequence: list[dict] = []

    for frame_number, timestamp in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            frame_sequence.append({"pose": [], "holds": hold_map, "timestamp": timestamp})
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints = estimator.estimate(frame_rgb)
        pose_dicts = [kp.model_dump() for kp in keypoints]

        frame_sequence.append({
            "pose": pose_dicts,
            "holds": hold_map,
            "timestamp": timestamp,
        })

    cap.release()
    estimator.close()

    # ── Phase 3: Detect move events → training samples ──
    events = detect_move_events(frame_sequence)
    logger.info("Detected %d move events in %s", len(events), video_path)

    samples: list[TrainingSample] = []
    for event in events:
        frame = frame_sequence[event.frame_idx]
        if not frame.get("pose"):
            continue
        samples.append(TrainingSample(
            pose=frame["pose"],
            holds=hold_map,
            target_hold_id=event.target_hold_id,
            hand=event.hand,
        ))

    return samples
