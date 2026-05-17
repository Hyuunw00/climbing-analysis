"""Rule-based path prediction for climbing route analysis."""

import math

from app.models.schemas import PoseKeypoint, Prediction


# MediaPipe landmark indices
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Max reachable distance (normalized coords, roughly arm span ratio)
MAX_REACH = 0.35

# Top N candidates to return per hand
TOP_N = 3


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _direction_weight(hand_y: float, hold_y: float) -> float:
    """
    Weight based on vertical direction.
    In normalized coords, y=0 is top, y=1 is bottom.
    Going up means hold_y < hand_y.
    """
    if hold_y < hand_y:
        return 1.5  # upward
    elif abs(hold_y - hand_y) < 0.05:
        return 1.0  # same level
    else:
        return 0.3  # downward


def _score_holds_for_hand(
    hand_x: float,
    hand_y: float,
    holds: list[dict],
    hand_label: str,
) -> list[Prediction]:
    """Score all holds for a single hand and return top candidates."""
    scored: list[tuple[str, float]] = []

    for hold in holds:
        hold_x = hold["x"]
        hold_y = hold["y"]

        distance = _euclidean_distance(hand_x, hand_y, hold_x, hold_y)

        # Reachability gate
        if distance > MAX_REACH or distance < 1e-6:
            continue

        direction_w = _direction_weight(hand_y, hold_y)
        score = (1.0 / distance) * direction_w

        scored.append((hold["id"], score))

    if not scored:
        return []

    # Normalize to probabilities
    total = sum(s for _, s in scored)
    predictions: list[Prediction] = []
    for hold_id, score in sorted(scored, key=lambda x: x[1], reverse=True)[:TOP_N]:
        probability = round((score / total) * 100, 1)
        predictions.append(
            Prediction(hold_id=hold_id, probability=probability, hand=hand_label)
        )

    return predictions


def predict_next_holds(
    pose_keypoints: list[PoseKeypoint],
    holds: list[dict],
) -> list[Prediction]:
    """
    Predict next holds based on current pose and detected holds.

    Args:
        pose_keypoints: Full list of 33 MediaPipe pose keypoints.
        holds: List of detected holds with id, x, y fields.

    Returns:
        List of Prediction objects (up to TOP_N per hand).
    """
    if not pose_keypoints or not holds:
        return []

    # Need at least enough keypoints to access wrists
    if len(pose_keypoints) <= max(LEFT_WRIST, RIGHT_WRIST):
        return []

    predictions: list[Prediction] = []

    left_wrist = pose_keypoints[LEFT_WRIST]
    right_wrist = pose_keypoints[RIGHT_WRIST]

    # Only predict if wrist is visible enough
    if left_wrist.visibility > 0.3:
        predictions.extend(
            _score_holds_for_hand(left_wrist.x, left_wrist.y, holds, "left")
        )

    if right_wrist.visibility > 0.3:
        predictions.extend(
            _score_holds_for_hand(right_wrist.x, right_wrist.y, holds, "right")
        )

    return predictions
