"""Train XGBoost model for path prediction.

Loads featurized training data from Parquet, trains a binary classifier,
and saves the model to models/path_predictor_xgb.json.

Usage:
    python -m scripts.train_model --data data/processed/training_data.parquet --output models/path_predictor_xgb.json

Compatible with Google Colab free tier (CPU-only, no GPU needed).
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost path prediction model")
    parser.add_argument("--data", type=str, default="data/processed/training_data.parquet", help="Training data Parquet")
    parser.add_argument("--output", type=str, default="models/path_predictor_xgb.json", help="Output model path")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--early-stopping", type=int, default=20, help="Early stopping rounds")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, classification_report

    # Add parent dir to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.ml.features import FEATURE_NAMES

    if not os.path.exists(args.data):
        logger.error("Training data not found: %s", args.data)
        sys.exit(1)

    df = pd.read_parquet(args.data)
    logger.info("Loaded %d samples from %s", len(df), args.data)
    logger.info("Positive: %d, Negative: %d", (df["label"] == 1).sum(), (df["label"] == 0).sum())

    # Features and labels
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    # Group by video for train/val split (prevent leakage)
    groups = df["video"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))
    logger.info("Train videos: %d, Val videos: %d",
                len(set(groups[train_idx])), len(set(groups[val_idx])))

    # Class imbalance handling
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    # Train XGBoost
    model = xgb.XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=args.early_stopping,
        random_state=42,
        tree_method="hist",  # fast CPU training
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Evaluation
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info("Validation AUC: %.4f", auc)

    # Precision@3: for each frame's set of candidates, is the true hold in top 3?
    # Group validation samples by their source (video + frame approximation)
    val_df = df.iloc[val_idx].copy()
    val_df["pred_proba"] = y_pred_proba

    correct_at_3 = 0
    total_groups = 0

    for (video, target_id, hand), group in val_df.groupby(["video", "target_hold_id", "hand"]):
        top3_ids = group.nlargest(3, "pred_proba")["candidate_hold_id"].values
        if target_id in top3_ids:
            correct_at_3 += 1
        total_groups += 1

    if total_groups > 0:
        precision_at_3 = correct_at_3 / total_groups
        logger.info("Precision@3: %.4f (%d/%d)", precision_at_3, correct_at_3, total_groups)

    # Feature importance (top 10)
    importance = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)
    logger.info("Top 10 features:")
    for name, imp in feat_imp[:10]:
        logger.info("  %-25s %.4f", name, imp)

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save_model(args.output)
    logger.info("Model saved to %s", args.output)

    # Save metadata alongside model
    meta_path = args.output.replace(".json", "_meta.json")
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "auc": round(auc, 4),
        "precision_at_3": round(precision_at_3, 4) if total_groups > 0 else None,
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else args.n_estimators,
        "hyperparameters": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "scale_pos_weight": round(scale_pos_weight, 2),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
