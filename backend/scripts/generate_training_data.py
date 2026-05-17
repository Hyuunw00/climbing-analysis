"""Generate training data from downloaded bouldering videos.

Runs hold detection + pose estimation on each video, detects move events,
and produces a Parquet file with featurized training samples.

Usage:
    python -m scripts.generate_training_data --video-dir data/raw/videos/ --output data/processed/training_data.parquet
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate ML training data from climbing videos")
    parser.add_argument("--video-dir", type=str, default="data/raw/videos/", help="Directory with video files")
    parser.add_argument("--output", type=str, default="data/processed/training_data.parquet", help="Output Parquet path")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0=all)")
    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        logger.error("Video directory not found: %s", args.video_dir)
        sys.exit(1)

    # Lazy imports to avoid loading heavy libs until needed
    import pandas as pd
    from app.ml.labeler import label_video
    from app.ml.features import extract_features, features_to_array, FEATURE_NAMES
    from app.models.schemas import PoseKeypoint

    video_files = sorted([
        f for f in os.listdir(args.video_dir)
        if f.lower().endswith((".mp4", ".mov", ".webm", ".avi"))
    ])

    if args.limit > 0:
        video_files = video_files[:args.limit]

    if not video_files:
        logger.error("No video files found in %s", args.video_dir)
        sys.exit(1)

    logger.info("Processing %d videos from %s", len(video_files), args.video_dir)

    all_rows: list[dict] = []
    total_positive = 0
    total_negative = 0

    for i, filename in enumerate(video_files, 1):
        video_path = os.path.join(args.video_dir, filename)
        logger.info("[%d/%d] Processing %s ...", i, len(video_files), filename)

        try:
            samples = label_video(video_path)
        except Exception as e:
            logger.warning("Failed to process %s: %s", filename, e)
            continue

        if not samples:
            logger.info("  No training samples from %s", filename)
            continue

        for sample in samples:
            # Convert pose dicts to PoseKeypoint objects
            kps = [PoseKeypoint(**kp) for kp in sample.pose]

            for hold in sample.holds:
                feats = extract_features(kps, sample.holds, sample.hand, hold)
                row = feats.copy()
                row["label"] = 1.0 if hold.get("id") == sample.target_hold_id else 0.0
                row["hand"] = sample.hand
                row["video"] = filename
                row["target_hold_id"] = sample.target_hold_id
                row["candidate_hold_id"] = hold.get("id", "")
                all_rows.append(row)

                if row["label"] == 1.0:
                    total_positive += 1
                else:
                    total_negative += 1

        logger.info("  %d samples from %s (total so far: %d pos, %d neg)",
                     len(samples), filename, total_positive, total_negative)

    if not all_rows:
        logger.error("No training data generated!")
        sys.exit(1)

    # Save as Parquet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.output, index=False)

    logger.info("Training data saved to %s", args.output)
    logger.info("Total: %d rows (%d positive, %d negative, ratio 1:%.1f)",
                len(all_rows), total_positive, total_negative,
                total_negative / max(total_positive, 1))


if __name__ == "__main__":
    main()
