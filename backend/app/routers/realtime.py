"""WebSocket endpoint for real-time camera frame analysis."""

import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import (
    FrameData,
    ForceDistribution,
    Hold,
    Prediction,
)
from app.services.hold_detector import hold_detector
from app.services.pose_estimator import PoseEstimator
from app.services.path_predictor import predict_next_holds
from app.services.force_estimator import estimate_force_distribution

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])


def _decode_frame(data: str) -> np.ndarray | None:
    """Decode a base64-encoded JPEG into a BGR numpy array.

    Returns None if decoding fails.
    """
    try:
        # Strip optional data-URI prefix
        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data)
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning("Base64 frame decode failed: %s", e)
        return None


def _analyze_frame(
    frame_bgr: np.ndarray,
    pose_estimator: PoseEstimator,
    frame_number: int,
) -> dict:
    """Run full analysis pipeline on a single frame.

    Returns a dict matching FrameData schema.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Pose estimation
    keypoints = pose_estimator.estimate(frame_rgb)

    # Hold detection (singleton, thread-safe inference)
    raw_holds = hold_detector.detect(frame_bgr)

    # Normalize hold coordinates and assign IDs
    holds: list[dict] = []
    for idx, rh in enumerate(raw_holds):
        holds.append({
            "id": f"h{idx}",
            "x": round(rh["x"] / w, 4),
            "y": round(rh["y"] / h, 4),
            "width": round(rh["width"] / w, 4),
            "height": round(rh["height"] / h, 4),
            "confidence": round(rh["confidence"], 4),
        })

    # Path prediction
    predictions = predict_next_holds(keypoints, holds)

    # Force distribution
    force_raw = estimate_force_distribution(keypoints)
    force_dist = None
    if force_raw is not None:
        force_dist = {
            "left_hand": force_raw.get("left_hand", 0.0),
            "right_hand": force_raw.get("right_hand", 0.0),
            "left_foot": force_raw.get("left_foot", 0.0),
            "right_foot": force_raw.get("right_foot", 0.0),
            "center_of_mass": force_raw.get("center_of_mass", {}),
        }

    return {
        "frame_number": frame_number,
        "timestamp": round(time.time(), 3),
        "holds": holds,
        "pose": [kp.model_dump() for kp in keypoints],
        "predictions": [p.model_dump() for p in predictions],
        "force_distribution": force_dist,
        "metrics": None,
    }


@router.websocket("/realtime")
async def realtime_analysis(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time frame-by-frame analysis.

    Protocol:
        - Client sends base64-encoded JPEG string per message.
        - Server responds with JSON matching FrameData schema.
        - Errors are sent as JSON with an "error" key; connection stays open.
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    # Per-connection pose estimator instance
    pose_estimator = PoseEstimator()
    frame_count = 0

    try:
        while True:
            data = await ws.receive_text()
            frame_count += 1

            # Decode frame
            frame_bgr = _decode_frame(data)
            if frame_bgr is None:
                await ws.send_json({
                    "error": "invalid_frame",
                    "message": "Failed to decode base64 JPEG",
                    "frame_number": frame_count,
                })
                continue

            # Run analysis pipeline
            try:
                result = _analyze_frame(frame_bgr, pose_estimator, frame_count)
                await ws.send_json(result)
            except Exception as e:
                logger.error("Analysis error on frame %d: %s", frame_count, e)
                await ws.send_json({
                    "frame_number": frame_count,
                    "timestamp": round(time.time(), 3),
                    "holds": [],
                    "pose": [],
                    "predictions": [],
                    "force_distribution": None,
                    "metrics": None,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (frames processed: %d)", frame_count)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        pose_estimator.close()
        logger.info("PoseEstimator resources released")
