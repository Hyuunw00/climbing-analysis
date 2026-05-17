"""Force distribution estimation based on biomechanics.

Estimates how body weight is distributed across contact points (hands/feet)
using center of mass position and moment-arm principles.
"""

import logging
from typing import Optional

from app.models.schemas import PoseKeypoint

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for COM calculation
_SHOULDER_L = 11
_SHOULDER_R = 12
_HIP_L = 23
_HIP_R = 24
_KNEE_L = 25
_KNEE_R = 26

# Contact point indices
_LEFT_HAND = 15   # left_wrist
_RIGHT_HAND = 16  # right_wrist
_LEFT_FOOT = 27   # left_ankle
_RIGHT_FOOT = 28  # right_ankle

# Body segment weights for COM approximation (fraction of total body mass)
# Simplified: upper torso ~40%, lower torso ~30%, thighs ~30%
_COM_WEIGHTS = {
    "shoulders": 0.40,
    "hips": 0.30,
    "knees": 0.30,
}

_CONFIDENCE_THRESHOLD = 0.3
_MIN_CONTACT_POINTS = 2


def _get_keypoint(
    keypoints: list[PoseKeypoint], index: int
) -> Optional[PoseKeypoint]:
    """Get a keypoint by index if it exists and has sufficient confidence."""
    if index < len(keypoints) and keypoints[index].confidence >= _CONFIDENCE_THRESHOLD:
        return keypoints[index]
    return None


def _compute_com(keypoints: list[PoseKeypoint]) -> Optional[dict[str, float]]:
    """Compute weighted center of mass from major body landmarks.

    Uses shoulder, hip, and knee midpoints with biomechanical weight factors.
    Returns {"x": float, "y": float} or None if insufficient data.
    """
    segments: list[tuple[float, float, float]] = []

    # Shoulder midpoint
    ls = _get_keypoint(keypoints, _SHOULDER_L)
    rs = _get_keypoint(keypoints, _SHOULDER_R)
    if ls and rs:
        segments.append((
            (ls.x + rs.x) / 2,
            (ls.y + rs.y) / 2,
            _COM_WEIGHTS["shoulders"],
        ))

    # Hip midpoint
    lh = _get_keypoint(keypoints, _HIP_L)
    rh = _get_keypoint(keypoints, _HIP_R)
    if lh and rh:
        segments.append((
            (lh.x + rh.x) / 2,
            (lh.y + rh.y) / 2,
            _COM_WEIGHTS["hips"],
        ))

    # Knee midpoint
    lk = _get_keypoint(keypoints, _KNEE_L)
    rk = _get_keypoint(keypoints, _KNEE_R)
    if lk and rk:
        segments.append((
            (lk.x + rk.x) / 2,
            (lk.y + rk.y) / 2,
            _COM_WEIGHTS["knees"],
        ))

    if not segments:
        return None

    total_weight = sum(w for _, _, w in segments)
    com_x = sum(x * w for x, _, w in segments) / total_weight
    com_y = sum(y * w for _, y, w in segments) / total_weight

    return {"x": round(com_x, 4), "y": round(com_y, 4)}


def estimate_force_distribution(
    keypoints: list[PoseKeypoint],
) -> Optional[dict]:
    """Estimate force distribution across contact points.

    Uses inverse-distance weighting from center of mass to each contact point.
    Farther contact points bear more load (moment-arm principle).

    Args:
        keypoints: List of 33 MediaPipe pose keypoints.

    Returns:
        Dict with force ratios per contact point and COM position,
        or None if pose data is insufficient.
    """
    if not keypoints or len(keypoints) < 29:
        return None

    com = _compute_com(keypoints)
    if com is None:
        return None

    # Gather active contact points
    contact_map = {
        "left_hand": _get_keypoint(keypoints, _LEFT_HAND),
        "right_hand": _get_keypoint(keypoints, _RIGHT_HAND),
        "left_foot": _get_keypoint(keypoints, _LEFT_FOOT),
        "right_foot": _get_keypoint(keypoints, _RIGHT_FOOT),
    }

    active_contacts: dict[str, float] = {}
    for name, kp in contact_map.items():
        if kp is not None:
            dist = ((kp.x - com["x"]) ** 2 + (kp.y - com["y"]) ** 2) ** 0.5
            # Avoid zero distance — use small epsilon
            active_contacts[name] = max(dist, 1e-4)

    # Fewer than 2 valid contacts → equal distribution across all 4
    if len(active_contacts) < _MIN_CONTACT_POINTS:
        return {
            "left_hand": 0.25,
            "right_hand": 0.25,
            "left_foot": 0.25,
            "right_foot": 0.25,
            "center_of_mass": com,
        }

    # Moment-arm weighting: force ∝ distance from COM
    total_dist = sum(active_contacts.values())
    force_ratios: dict[str, float] = {}

    for name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
        if name in active_contacts:
            force_ratios[name] = round(active_contacts[name] / total_dist, 4)
        else:
            force_ratios[name] = 0.0

    # Normalize to ensure sum = 1.0 (handle rounding)
    ratio_sum = sum(force_ratios.values())
    if ratio_sum > 0:
        for name in force_ratios:
            force_ratios[name] = round(force_ratios[name] / ratio_sum, 4)

    # Final correction: adjust largest value so sum is exactly 1.0
    diff = 1.0 - sum(force_ratios.values())
    if diff != 0:
        largest = max(force_ratios, key=force_ratios.get)  # type: ignore
        force_ratios[largest] = round(force_ratios[largest] + diff, 4)

    force_ratios["center_of_mass"] = com
    return force_ratios
