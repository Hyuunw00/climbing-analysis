"""Performance scoring service for climbing video analysis.

Calculates per-frame metrics from pose keypoints and hold detections:
- Smoothness: joint movement jerk (lower jerk = higher score)
- Technique: center-of-mass path efficiency
- Stable: joint displacement variance (lower = more stable)
- Adhesion: hand/foot proximity to hold bounding boxes
- Vertical velocity: COM y-axis movement between frames
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

from app.models.schemas import PoseKeypoint

logger = logging.getLogger(__name__)

# Joint names used for center-of-mass calculation
_COM_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]

# Joint names used for smoothness / stability (major body joints)
_TRACKED_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Hand/foot keypoint names for adhesion scoring
_HAND_NAMES = ["left_wrist", "right_wrist", "left_index", "right_index"]
_FOOT_NAMES = ["left_ankle", "right_ankle", "left_foot_index", "right_foot_index"]

# Adhesion: max distance threshold for "inside hold" (normalized coords)
_ADHESION_DIST_THRESHOLD = 0.06

# Default score when insufficient data
_DEFAULT_SCORE = 50.0

# Min confidence to consider a keypoint valid
_MIN_CONFIDENCE = 0.3


@dataclass
class FrameMetrics:
    smoothness: float
    technique: float
    stable: float
    adhesion: float
    score: float
    vertical_velocity: float


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _get_keypoint(keypoints: list[PoseKeypoint], name: str) -> PoseKeypoint | None:
    """Find a keypoint by name with sufficient confidence."""
    for kp in keypoints:
        if kp.name == name and kp.confidence >= _MIN_CONFIDENCE:
            return kp
    return None


def _compute_com(keypoints: list[PoseKeypoint]) -> tuple[float, float] | None:
    """Compute center of mass from major joints (normalized coords)."""
    xs, ys = [], []
    for name in _COM_JOINTS:
        kp = _get_keypoint(keypoints, name)
        if kp is not None:
            xs.append(kp.x)
            ys.append(kp.y)
    if len(xs) < 2:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _get_joint_positions(keypoints: list[PoseKeypoint]) -> dict[str, tuple[float, float]]:
    """Extract tracked joint positions as {name: (x, y)}."""
    positions: dict[str, tuple[float, float]] = {}
    for name in _TRACKED_JOINTS:
        kp = _get_keypoint(keypoints, name)
        if kp is not None:
            positions[name] = (kp.x, kp.y)
    return positions


# ── Metric calculators ──────────────────────────────────────────────


def _calc_smoothness(
    prev_positions: dict[str, tuple[float, float]] | None,
    curr_positions: dict[str, tuple[float, float]],
    next_positions: dict[str, tuple[float, float]] | None,
) -> float:
    """Smoothness from jerk (second derivative of position).

    Lower jerk → smoother movement → higher score.
    Needs prev and next frames; returns default if unavailable.
    """
    if prev_positions is None or next_positions is None:
        return _DEFAULT_SCORE

    jerks: list[float] = []
    for name in _TRACKED_JOINTS:
        if name in prev_positions and name in curr_positions and name in next_positions:
            px, py = prev_positions[name]
            cx, cy = curr_positions[name]
            nx, ny = next_positions[name]
            # Acceleration ≈ (next - curr) - (curr - prev) = next - 2*curr + prev
            jx = nx - 2 * cx + px
            jy = ny - 2 * cy + py
            jerks.append(math.sqrt(jx ** 2 + jy ** 2))

    if not jerks:
        return _DEFAULT_SCORE

    avg_jerk = float(np.mean(jerks))
    # Map jerk to score: jerk=0 → 100, jerk>=0.05 → 0
    score = 100.0 * (1.0 - min(avg_jerk / 0.05, 1.0))
    return _clamp(score)


def _calc_technique(
    com_history: list[tuple[float, float]],
    window_start: int,
    window_end: int,
) -> float:
    """Technique = path efficiency of COM over a sliding window.

    Ratio of straight-line distance to actual path length.
    Closer to 1.0 → more efficient → higher score.
    """
    segment = com_history[window_start:window_end + 1]
    if len(segment) < 2:
        return _DEFAULT_SCORE

    # Actual path length
    path_length = 0.0
    for i in range(1, len(segment)):
        dx = segment[i][0] - segment[i - 1][0]
        dy = segment[i][1] - segment[i - 1][1]
        path_length += math.sqrt(dx ** 2 + dy ** 2)

    if path_length < 1e-6:
        return 100.0  # No movement = perfectly efficient

    # Straight-line distance start→end
    straight = math.sqrt(
        (segment[-1][0] - segment[0][0]) ** 2
        + (segment[-1][1] - segment[0][1]) ** 2
    )

    efficiency = straight / path_length  # 0..1
    return _clamp(efficiency * 100.0)


def _calc_stability(
    positions_window: list[dict[str, tuple[float, float]]],
) -> float:
    """Stability from joint displacement standard deviation.

    Lower std → more stable → higher score.
    """
    if len(positions_window) < 2:
        return _DEFAULT_SCORE

    displacements: list[float] = []
    for name in _TRACKED_JOINTS:
        coords = [p[name] for p in positions_window if name in p]
        if len(coords) < 2:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        displacements.append(float(np.std(xs)))
        displacements.append(float(np.std(ys)))

    if not displacements:
        return _DEFAULT_SCORE

    avg_std = float(np.mean(displacements))
    # Map: std=0 → 100, std>=0.03 → 0
    score = 100.0 * (1.0 - min(avg_std / 0.03, 1.0))
    return _clamp(score)


def _calc_adhesion(
    keypoints: list[PoseKeypoint],
    holds: list[dict],
) -> float:
    """Adhesion: ratio of hand/foot keypoints within hold bounding boxes."""
    if not holds:
        return _DEFAULT_SCORE

    contact_names = _HAND_NAMES + _FOOT_NAMES
    total = 0
    inside = 0

    for name in contact_names:
        kp = _get_keypoint(keypoints, name)
        if kp is None:
            continue
        total += 1

        for h in holds:
            hx, hy = h["x"], h["y"]
            hw, hh = h["width"] / 2, h["height"] / 2
            # Check if keypoint is within hold bbox (with threshold margin)
            if (abs(kp.x - hx) <= hw + _ADHESION_DIST_THRESHOLD
                    and abs(kp.y - hy) <= hh + _ADHESION_DIST_THRESHOLD):
                inside += 1
                break  # counted once per keypoint

    if total == 0:
        return _DEFAULT_SCORE

    return _clamp((inside / total) * 100.0)


def _calc_vertical_velocity(
    com_prev: tuple[float, float] | None,
    com_curr: tuple[float, float] | None,
    dt: float,
) -> float:
    """Vertical velocity of COM between frames.

    Positive = moving upward (y decreasing in image coords).
    Returns 0 if data insufficient.
    """
    if com_prev is None or com_curr is None or dt <= 0:
        return 0.0
    # In image coords, y increases downward, so negate for "upward = positive"
    return -(com_curr[1] - com_prev[1]) / dt


# ── Main entry point ────────────────────────────────────────────────

# Sliding window size for technique and stability
_WINDOW_SIZE = 5


def compute_frame_metrics(
    frame_sequence: list[dict],
    fps: float,
) -> list[FrameMetrics]:
    """Compute performance metrics for each frame in the sequence.

    Args:
        frame_sequence: list of frame dicts with 'pose' and 'holds' fields.
            Each pose entry is a dict matching PoseKeypoint fields.
        fps: video frame rate for velocity calculation.

    Returns:
        List of FrameMetrics, one per input frame.
    """
    n = len(frame_sequence)
    if n == 0:
        return []

    dt = 1.0 / fps if fps > 0 else 1.0

    # Pre-compute keypoints, joint positions, and COMs for all frames
    all_keypoints: list[list[PoseKeypoint]] = []
    all_positions: list[dict[str, tuple[float, float]]] = []
    all_coms: list[tuple[float, float] | None] = []

    for frame in frame_sequence:
        raw_pose = frame.get("pose", [])
        if raw_pose and isinstance(raw_pose[0], dict):
            kps = [PoseKeypoint(**kp) for kp in raw_pose]
        elif raw_pose and isinstance(raw_pose[0], PoseKeypoint):
            kps = raw_pose
        else:
            kps = []

        all_keypoints.append(kps)
        all_positions.append(_get_joint_positions(kps))
        all_coms.append(_compute_com(kps))

    # Build COM history (only non-None entries with their indices)
    com_history: list[tuple[float, float]] = []
    com_index_map: dict[int, int] = {}  # frame_idx → com_history idx
    for i, com in enumerate(all_coms):
        if com is not None:
            com_index_map[i] = len(com_history)
            com_history.append(com)

    results: list[FrameMetrics] = []

    for i in range(n):
        kps = all_keypoints[i]
        holds = frame_sequence[i].get("holds", [])

        # Smoothness: needs prev and next frame positions
        prev_pos = all_positions[i - 1] if i > 0 else None
        next_pos = all_positions[i + 1] if i < n - 1 else None
        smoothness = _calc_smoothness(prev_pos, all_positions[i], next_pos)

        # Technique: sliding window over COM history
        if i in com_index_map:
            ci = com_index_map[i]
            w_start = max(0, ci - _WINDOW_SIZE // 2)
            w_end = min(len(com_history) - 1, ci + _WINDOW_SIZE // 2)
            technique = _calc_technique(com_history, w_start, w_end)
        else:
            technique = _DEFAULT_SCORE

        # Stability: sliding window over joint positions
        w_start_idx = max(0, i - _WINDOW_SIZE // 2)
        w_end_idx = min(n, i + _WINDOW_SIZE // 2 + 1)
        stability = _calc_stability(all_positions[w_start_idx:w_end_idx])

        # Adhesion
        adhesion = _calc_adhesion(kps, holds)

        # Composite score
        score = _clamp((smoothness + technique + stability + adhesion) / 4.0)

        # Vertical velocity
        com_prev = all_coms[i - 1] if i > 0 else None
        com_curr = all_coms[i]
        vertical_velocity = _calc_vertical_velocity(com_prev, com_curr, dt)

        results.append(FrameMetrics(
            smoothness=round(smoothness, 1),
            technique=round(technique, 1),
            stable=round(stability, 1),
            adhesion=round(adhesion, 1),
            score=round(score, 1),
            vertical_velocity=round(vertical_velocity, 4),
        ))

    return results
