"""Train YOLOv8s hold detection model using Roboflow datasets.

Run this script in Google Colab (free GPU) or locally.

Usage (Colab):
    1. Upload this script to Colab
    2. Run: !pip install ultralytics roboflow
    3. Run: !python train_hold_detector.py --roboflow-key YOUR_API_KEY

Usage (local):
    python -m scripts.train_hold_detector --roboflow-key YOUR_API_KEY

The trained model will be saved to models/yolov8s-climbing-holds.pt
"""

import argparse
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_datasets(api_key: str, output_dir: str = "datasets") -> list[str]:
    """Download climbing hold datasets from Roboflow.

    Returns list of dataset.yaml paths.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    yaml_paths = []

    datasets_config = [
        # (workspace, project, version, name)
        ("climbing-holds-detection-shjjz", "neural-climb-v2-7rjlj", 1, "neural-climb-v2"),
        ("blackcreed-xpgxh", "climbing-holds-and-volumes", 1, "holds-and-volumes"),
    ]

    for workspace, project, version, name in datasets_config:
        try:
            logger.info("Downloading %s ...", name)
            proj = rf.workspace(workspace).project(project)
            ds = proj.version(version).download("yolov8", location=os.path.join(output_dir, name))
            yaml_path = os.path.join(output_dir, name, "data.yaml")
            if os.path.exists(yaml_path):
                yaml_paths.append(yaml_path)
                logger.info("  Downloaded: %s", yaml_path)
            else:
                logger.warning("  data.yaml not found for %s", name)
        except Exception as e:
            logger.warning("Failed to download %s: %s", name, e)

    return yaml_paths


def merge_datasets(yaml_paths: list[str], output_dir: str = "datasets/merged") -> str:
    """Merge multiple YOLO datasets into one.

    Combines train/val/test images and labels from all datasets.
    Returns path to merged data.yaml.
    """
    import yaml

    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid", "labels"), exist_ok=True)

    all_classes = set()

    for yaml_path in yaml_paths:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        ds_dir = os.path.dirname(yaml_path)
        names = config.get("names", [])
        if isinstance(names, dict):
            names = list(names.values())
        all_classes.update(names)

        # Copy files from train/valid splits
        for split in ["train", "valid", "val", "test"]:
            target_split = "valid" if split in ("valid", "val", "test") else "train"

            for subdir in ["images", "labels"]:
                src_dir = os.path.join(ds_dir, split, subdir)
                if not os.path.isdir(src_dir):
                    continue
                dst_dir = os.path.join(output_dir, target_split, subdir)

                for fname in os.listdir(src_dir):
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

    # All holds map to class 0 ("hold") for simplicity
    merged_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "hold"},
    }

    yaml_out = os.path.join(output_dir, "data.yaml")
    with open(yaml_out, "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False)

    # Count files
    train_count = len(os.listdir(os.path.join(output_dir, "train", "images")))
    val_count = len(os.listdir(os.path.join(output_dir, "valid", "images")))
    logger.info("Merged dataset: %d train, %d val images", train_count, val_count)
    logger.info("Classes: %s → mapped to single 'hold' class", all_classes)

    return yaml_out


def remap_labels(merged_dir: str) -> None:
    """Remap all label class IDs to 0 (single 'hold' class).

    Some datasets have multiple classes (holds, volumes, etc).
    We treat them all as 'hold' for detection.
    """
    for split in ["train", "valid"]:
        labels_dir = os.path.join(merged_dir, split, "labels")
        if not os.path.isdir(labels_dir):
            continue

        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(labels_dir, fname)
            lines = []
            with open(fpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # remap class to 0
                        lines.append(" ".join(parts))
            with open(fpath, "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")

    logger.info("Labels remapped to single class (0: hold)")


def train_model(
    data_yaml: str,
    model_size: str = "s",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    output_dir: str = "runs/hold_detect",
) -> str:
    """Train YOLOv8 model on the dataset.

    Args:
        data_yaml: Path to data.yaml
        model_size: YOLOv8 variant (n/s/m/l)
        epochs: Training epochs
        imgsz: Image size
        batch: Batch size (reduce if GPU OOM)
        output_dir: Training output directory

    Returns:
        Path to best model weights.
    """
    from ultralytics import YOLO

    base_model = f"yolov8{model_size}.pt"
    logger.info("Training YOLOv8%s on %s for %d epochs", model_size, data_yaml, epochs)

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="train",
        patience=20,         # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = os.path.join(output_dir, "train", "weights", "best.pt")
    if os.path.exists(best_path):
        logger.info("Best model saved to %s", best_path)
        return best_path
    else:
        logger.warning("best.pt not found, checking last.pt")
        last_path = os.path.join(output_dir, "train", "weights", "last.pt")
        return last_path


def evaluate_model(model_path: str, data_yaml: str) -> None:
    """Run validation and print metrics."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=True)

    logger.info("=== Evaluation Results ===")
    logger.info("mAP50:     %.4f", metrics.box.map50)
    logger.info("mAP50-95:  %.4f", metrics.box.map)
    logger.info("Precision: %.4f", metrics.box.mp)
    logger.info("Recall:    %.4f", metrics.box.mr)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 hold detection model")
    parser.add_argument("--roboflow-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l"],
                        help="YOLOv8 model size (default: s)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce for less GPU memory)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default="models/yolov8s-climbing-holds.pt",
                        help="Where to copy final model")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download (use existing)")
    args = parser.parse_args()

    # Step 1: Download datasets
    if not args.skip_download:
        yaml_paths = download_datasets(args.roboflow_key)
        if not yaml_paths:
            logger.error("No datasets downloaded. Check your API key and network.")
            return
    else:
        yaml_paths = []
        for name in ["neural-climb-v2", "holds-and-volumes"]:
            p = os.path.join("datasets", name, "data.yaml")
            if os.path.exists(p):
                yaml_paths.append(p)
        if not yaml_paths:
            logger.error("No existing datasets found in datasets/")
            return

    # Step 2: Merge datasets
    merged_yaml = merge_datasets(yaml_paths)

    # Step 3: Remap labels to single class
    remap_labels("datasets/merged")

    # Step 4: Train
    best_model = train_model(
        data_yaml=merged_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    # Step 5: Evaluate
    if best_model and os.path.exists(best_model):
        evaluate_model(best_model, merged_yaml)

        # Copy to target location
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.copy2(best_model, args.output)
        logger.info("Final model copied to %s", args.output)
        logger.info("")
        logger.info("To use this model, set in .env:")
        logger.info("  HOLD_MODEL_PATH=%s", args.output)
    else:
        logger.error("Training failed — no model produced")


if __name__ == "__main__":
    main()
