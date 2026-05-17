"""Hold detection service using YOLOv8."""

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class HoldDetector:
    """Detects climbing holds in a frame using YOLOv8."""

    def __init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model from configured path. Gracefully handles missing model."""
        model_path = Path(settings.hold_model_path)

        if not model_path.exists():
            logger.warning(
                "Hold detection model not found at '%s'. "
                "Hold detection will return empty results.",
                model_path,
            )
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
            logger.info("Hold detection model loaded from '%s'", model_path)
        except Exception as e:
            logger.error("Failed to load hold detection model: %s", e)
            self._model = None

    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run hold detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected holds with keys:
            x, y (center), width, height, confidence, label.
        """
        if not self.is_available:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=settings.hold_confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.error("Hold detection inference failed: %s", e)
            return []

        holds: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # xyxy -> center x, y, width, height
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, str(cls_id))

                holds.append(
                    {
                        "x": cx,
                        "y": cy,
                        "width": w,
                        "height": h,
                        "confidence": conf,
                        "label": label,
                    }
                )

        return holds


# Singleton instance — initialized once at import time
hold_detector = HoldDetector()
