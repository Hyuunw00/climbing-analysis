"""ML-based path predictor using trained XGBoost model.

Drop-in replacement for the rule-based path predictor.
Falls back gracefully if model file is not available.
"""

import logging
import os

import numpy as np

from app.ml.features import extract_features, features_to_array, FEATURE_NAMES
from app.models.schemas import PoseKeypoint, Prediction

logger = logging.getLogger(__name__)

# Wrist keypoint names
_WRIST_NAMES = {"left": "left_wrist", "right": "right_wrist"}

TOP_N = 3
_MIN_VISIBILITY = 0.3


class MLPathPredictor:
    """XGBoost-based path predictor.

    Loads a trained model and predicts next holds based on pose + hold features.
    If the model file is missing, is_available is False and predict() should not be called.
    """

    def __init__(self, model_path: str):
        self._available = False
        self._model = None

        if not os.path.exists(model_path):
            logger.info("ML path model not found at %s, using rule-based fallback", model_path)
            return

        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(model_path)
            self._available = True
            logger.info("ML path predictor loaded from %s", model_path)
        except ImportError:
            logger.warning("xgboost not installed, using rule-based fallback")
        except Exception as e:
            logger.warning("Failed to load ML path model: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(
        self,
        pose_keypoints: list[PoseKeypoint],
        holds: list[dict],
    ) -> list[Prediction]:
        """Predict next holds using the ML model.

        Args:
            pose_keypoints: Full MediaPipe pose keypoints.
            holds: List of route holds with id, x, y, width, height, confidence.

        Returns:
            List of Prediction objects (up to TOP_N per hand).
        """
        if not self._available or not pose_keypoints or not holds:
            return []

        predictions: list[Prediction] = []

        for hand in ["left", "right"]:
            wrist_name = _WRIST_NAMES[hand]
            wrist = next(
                (kp for kp in pose_keypoints if kp.name == wrist_name and kp.visibility > _MIN_VISIBILITY),
                None,
            )
            if not wrist:
                continue

            # Extract features for all candidate holds
            feature_rows = []
            valid_holds = []
            for hold in holds:
                feats = extract_features(pose_keypoints, holds, hand, hold)
                feature_rows.append(features_to_array(feats))
                valid_holds.append(hold)

            if not feature_rows:
                continue

            X = np.array(feature_rows, dtype=np.float32)
            probabilities = self._model.predict_proba(X)[:, 1]

            # Sort by probability, take top N
            indices = np.argsort(probabilities)[::-1][:TOP_N]

            # Normalize top N probabilities to sum to ~100
            top_probs = probabilities[indices]
            total = top_probs.sum()
            if total < 1e-6:
                continue

            for idx in indices:
                prob = (probabilities[idx] / total) * 100
                if prob < 1.0:
                    continue
                predictions.append(Prediction(
                    hold_id=valid_holds[idx]["id"],
                    probability=round(prob, 1),
                    hand=hand,
                ))

        return predictions
