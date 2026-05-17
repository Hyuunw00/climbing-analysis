"""Feature extraction for ML path prediction.

Converts raw pose keypoints + hold coordinates into a feature vector
for each (hand, candidate_hold) pair. Used for both training and inference.
"""

import math
from typing import Optional

from app.models.schemas import PoseKeypoint

# Keypoint names used for feature extraction
_WRIST = {"left": "left_wrist", "right": "right_wrist"}
_ELBOW = {"left": "left_elbow", "right": "right_elbow"}
_SHOULDER = {"left": "left_shoulder", "right": "right_shoulder"}
_HIP = {"left": "left_hip", "right": "right_hip"}
_ANKLE = {"left": "left_ankle", "right": "right_ankle"}

# COM joints (same as performance_scorer)
_COM_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]

MAX_REACH = 0.35
_MIN_CONF = 0.3


def _find_kp(keypoints: list[PoseKeypoint], name: str) -> Optional[PoseKeypoint]:
    for kp in keypoints:
        if kp.name == name and kp.confidence >= _MIN_CONF:
            return kp
    return None


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _angle_rad(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def _compute_com(keypoints: list[PoseKeypoint]) -> Optional[tuple[float, float]]:
    xs, ys = [], []
    for name in _COM_JOINTS:
        kp = _find_kp(keypoints, name)
        if kp:
            xs.append(kp.x)
            ys.append(kp.y)
    if len(xs) < 2:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _elbow_angle(
    shoulder: Optional[PoseKeypoint],
    elbow: Optional[PoseKeypoint],
    wrist: Optional[PoseKeypoint],
) -> float:
    """Angle at the elbow joint in radians (0 = fully extended, pi = fully bent)."""
    if not shoulder or not elbow or not wrist:
        return math.pi / 2  # default: 90 degrees
    v1x, v1y = shoulder.x - elbow.x, shoulder.y - elbow.y
    v2x, v2y = wrist.x - elbow.x, wrist.y - elbow.y
    dot = v1x * v2x + v1y * v2y
    mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = math.sqrt(v2x ** 2 + v2y ** 2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return math.pi / 2
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_angle)


def extract_features(
    keypoints: list[PoseKeypoint],
    holds: list[dict],
    hand: str,
    candidate_hold: dict,
) -> dict[str, float]:
    """Extract features for a single (hand, candidate_hold) pair.

    Args:
        keypoints: Full pose keypoints (33 MediaPipe landmarks).
        holds: All route holds (list of dicts with x, y, width, height, confidence).
        hand: "left" or "right".
        candidate_hold: The specific hold to score.

    Returns:
        Feature dict with ~36 float values.
    """
    features: dict[str, float] = {}
    other = "right" if hand == "left" else "left"

    wrist = _find_kp(keypoints, _WRIST[hand])
    other_wrist = _find_kp(keypoints, _WRIST[other])
    elbow = _find_kp(keypoints, _ELBOW[hand])
    shoulder = _find_kp(keypoints, _SHOULDER[hand])
    hip = _find_kp(keypoints, _HIP[hand])
    other_hip = _find_kp(keypoints, _HIP[other])
    ankle = _find_kp(keypoints, _ANKLE[hand])
    other_ankle = _find_kp(keypoints, _ANKLE[other])

    hx, hy = candidate_hold["x"], candidate_hold["y"]
    hw, hh = candidate_hold.get("width", 0.05), candidate_hold.get("height", 0.05)

    # Defaults when keypoints missing
    wx = wrist.x if wrist else 0.5
    wy = wrist.y if wrist else 0.5

    # ── Spatial features ──
    distance = _dist(wx, wy, hx, hy)
    features["distance"] = distance
    features["dx"] = hx - wx  # positive = hold is to the right
    features["dy"] = hy - wy  # positive = hold is below (image coords)
    features["angle"] = _angle_rad(wx, wy, hx, hy)

    # ── Reachability ──
    features["reach_ratio"] = distance / MAX_REACH if MAX_REACH > 0 else 1.0
    features["is_above"] = 1.0 if hy < wy else 0.0
    # Does the hold's lateral direction match the hand side?
    features["lateral_match"] = 1.0 if (hand == "left" and hx < wx) or (hand == "right" and hx > wx) else 0.0

    # ── Body geometry ──
    if shoulder:
        features["shoulder_to_hold"] = _dist(shoulder.x, shoulder.y, hx, hy)
    else:
        features["shoulder_to_hold"] = distance

    if elbow:
        features["elbow_to_hold"] = _dist(elbow.x, elbow.y, hx, hy)
    else:
        features["elbow_to_hold"] = distance

    if hip:
        features["hip_to_hold"] = _dist(hip.x, hip.y, hx, hy)
    else:
        features["hip_to_hold"] = distance + 0.2

    features["elbow_angle"] = _elbow_angle(shoulder, elbow, wrist)

    # ── COM features ──
    com = _compute_com(keypoints)
    if com:
        features["com_x"] = com[0]
        features["com_y"] = com[1]
        features["com_to_hold"] = _dist(com[0], com[1], hx, hy)
        features["com_to_hold_angle"] = _angle_rad(com[0], com[1], hx, hy)
    else:
        features["com_x"] = 0.5
        features["com_y"] = 0.5
        features["com_to_hold"] = distance
        features["com_to_hold_angle"] = 0.0

    # ── Hold context ──
    features["hold_width"] = hw
    features["hold_height"] = hh
    features["hold_area"] = hw * hh
    features["hold_confidence"] = candidate_hold.get("confidence", 0.5)

    # ── Relative hold position ──
    if holds:
        distances = sorted([_dist(wx, wy, h["x"], h["y"]) for h in holds])
        rank_dist = distances.index(distance) if distance in distances else len(distances)
        features["rank_by_distance"] = rank_dist / max(len(holds) - 1, 1)

        vertical_gains = sorted([wy - h["y"] for h in holds], reverse=True)
        this_gain = wy - hy
        rank_vert = vertical_gains.index(this_gain) if this_gain in vertical_gains else len(vertical_gains)
        features["rank_by_vertical_gain"] = rank_vert / max(len(holds) - 1, 1)

        all_x = [h["x"] for h in holds]
        all_y = [h["y"] for h in holds]
        x_range = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
        y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0
        features["hold_percentile_x"] = (hx - min(all_x)) / max(x_range, 1e-6)
        features["hold_percentile_y"] = (hy - min(all_y)) / max(y_range, 1e-6)
    else:
        features["rank_by_distance"] = 0.5
        features["rank_by_vertical_gain"] = 0.5
        features["hold_percentile_x"] = 0.5
        features["hold_percentile_y"] = 0.5

    # ── Symmetry ──
    features["same_side"] = features["lateral_match"]
    if other_wrist:
        features["cross_body_dist"] = _dist(other_wrist.x, other_wrist.y, hx, hy)
    else:
        features["cross_body_dist"] = distance

    # ── Current contact context ──
    if other_wrist and holds:
        features["other_hand_nearest_hold"] = min(
            _dist(other_wrist.x, other_wrist.y, h["x"], h["y"]) for h in holds
        )
    else:
        features["other_hand_nearest_hold"] = 0.1

    feet_ys = []
    for a in [ankle, other_ankle]:
        if a:
            feet_ys.append(a.y)
    features["feet_avg_y"] = sum(feet_ys) / len(feet_ys) if feet_ys else 0.8

    if wrist and other_wrist:
        features["hands_spread"] = _dist(wrist.x, wrist.y, other_wrist.x, other_wrist.y)
    else:
        features["hands_spread"] = 0.2

    # ── Pose summary ──
    l_shoulder = _find_kp(keypoints, "left_shoulder")
    r_shoulder = _find_kp(keypoints, "right_shoulder")
    if l_shoulder and r_shoulder:
        dx_s = r_shoulder.x - l_shoulder.x
        dy_s = r_shoulder.y - l_shoulder.y
        features["body_tilt"] = math.atan2(dy_s, dx_s)
    else:
        features["body_tilt"] = 0.0

    if hip and shoulder:
        features["torso_length"] = _dist(shoulder.x, shoulder.y, hip.x, hip.y)
    else:
        features["torso_length"] = 0.2

    if hip and _find_kp(keypoints, f"{hand}_knee"):
        knee = _find_kp(keypoints, f"{hand}_knee")
        features["hip_angle"] = _elbow_angle(shoulder, hip, knee)
    else:
        features["hip_angle"] = math.pi / 2

    # ── Wall position ──
    features["wrist_x"] = wx
    features["wrist_y"] = wy
    if com:
        if holds:
            all_y = [h["y"] for h in holds]
            y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0
            features["com_progress"] = (max(all_y) - com[1]) / max(y_range, 1e-6)
        else:
            features["com_progress"] = 0.5
    else:
        features["com_progress"] = 0.5

    return features


# Ordered feature names for consistent model input
FEATURE_NAMES: list[str] = [
    "distance", "dx", "dy", "angle",
    "reach_ratio", "is_above", "lateral_match",
    "shoulder_to_hold", "elbow_to_hold", "hip_to_hold", "elbow_angle",
    "com_x", "com_y", "com_to_hold", "com_to_hold_angle",
    "hold_width", "hold_height", "hold_area", "hold_confidence",
    "rank_by_distance", "rank_by_vertical_gain", "hold_percentile_x", "hold_percentile_y",
    "same_side", "cross_body_dist",
    "other_hand_nearest_hold", "feet_avg_y", "hands_spread",
    "body_tilt", "torso_length", "hip_angle",
    "wrist_x", "wrist_y", "com_progress",
]


def features_to_array(features: dict[str, float]) -> list[float]:
    """Convert feature dict to ordered array matching FEATURE_NAMES."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]
