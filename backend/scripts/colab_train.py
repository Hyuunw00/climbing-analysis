"""One-click hold detection model training for Google Colab.

Upload this file to Colab, then run:
    !pip install -q ultralytics
    !python colab_train.py

It handles everything: Drive mount, dataset find, merge, train, save.
All errors are caught and reported — no manual debugging needed.
"""

import os
import sys
import shutil
import yaml


def log(msg):
    print(f"[*] {msg}")


def error(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def mount_drive():
    """Mount Google Drive (skip if already mounted)."""
    if os.path.isdir("/content/drive/MyDrive"):
        log("Google Drive already mounted")
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        log("Google Drive mounted")
    except Exception as e:
        error(f"Drive mount failed: {e}\n  → Run this in a separate cell first:\n    from google.colab import drive\n    drive.mount('/content/drive')")


def find_datasets():
    """Find climbing dataset folders in Drive root (fast, no recursive find)."""
    drive_root = "/content/drive/MyDrive"
    if not os.path.isdir(drive_root):
        error("Drive not mounted or MyDrive not found")

    keywords = ["climb", "hold", "neural", "volume", "boulder"]
    yaml_paths = []

    log("Scanning Drive root for datasets...")
    try:
        items = os.listdir(drive_root)
    except Exception as e:
        error(f"Cannot list Drive: {e}")

    for item in items:
        if not any(kw in item.lower() for kw in keywords):
            continue
        full = os.path.join(drive_root, item)
        if not os.path.isdir(full):
            continue

        # Search max 3 levels deep for data.yaml
        for depth, (root, dirs, files) in enumerate(os.walk(full)):
            if depth > 3:
                break
            if "data.yaml" in files:
                yaml_paths.append(os.path.join(root, "data.yaml"))
                log(f"  Found: {os.path.join(root, 'data.yaml')}")
                break

    if not yaml_paths:
        log("Auto-detect failed. Drive root folders:")
        for item in sorted(items)[:30]:
            print(f"    {item}")
        error("No datasets found. Rename folders to include 'climb' or 'hold'.")

    log(f"Found {len(yaml_paths)} dataset(s)")
    return yaml_paths


def copy_to_local(yaml_paths):
    """Copy datasets from Drive to local (Drive I/O is slow for training)."""
    local_dirs = []
    for i, yp in enumerate(yaml_paths):
        ds_dir = os.path.dirname(yp)
        local_dir = f"/content/datasets/dataset_{i}"
        log(f"Copying {ds_dir} → {local_dir} ...")
        shutil.copytree(ds_dir, local_dir, dirs_exist_ok=True)
        local_dirs.append(local_dir)
    return local_dirs


def merge_datasets(local_dirs):
    """Merge all datasets into one with single 'hold' class."""
    merged = "/content/datasets/merged"
    for split in ["train", "valid"]:
        os.makedirs(f"{merged}/{split}/images", exist_ok=True)
        os.makedirs(f"{merged}/{split}/labels", exist_ok=True)

    # Find all data.yaml in local dirs
    yaml_paths = []
    for d in local_dirs:
        for root, dirs, files in os.walk(d):
            if "data.yaml" in files:
                yaml_paths.append(os.path.join(root, "data.yaml"))
                break

    for yp in yaml_paths:
        ds_dir = os.path.dirname(yp)
        with open(yp) as f:
            config = yaml.safe_load(f)
        log(f"  Merging {yp}: classes={config.get('names', {})}")

        for split in ["train", "valid", "val", "test"]:
            target = "valid" if split in ("valid", "val", "test") else "train"
            for sub in ["images", "labels"]:
                src = os.path.join(ds_dir, split, sub)
                if not os.path.isdir(src):
                    continue
                dst = f"{merged}/{target}/{sub}"
                for f in os.listdir(src):
                    s, d = os.path.join(src, f), os.path.join(dst, f)
                    if not os.path.exists(d):
                        shutil.copy2(s, d)

    # Remap all labels to class 0
    remapped = 0
    for split in ["train", "valid"]:
        labels_dir = f"{merged}/{split}/labels"
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(labels_dir, fname)
            lines = []
            with open(fpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"
                        lines.append(" ".join(parts))
            with open(fpath, "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            remapped += 1

    # Create merged data.yaml
    yaml_out = f"{merged}/data.yaml"
    config = {
        "path": os.path.abspath(merged),
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "hold"},
    }
    with open(yaml_out, "w") as f:
        yaml.dump(config, f)

    train_n = len([x for x in os.listdir(f"{merged}/train/images") if not x.startswith(".")])
    val_n = len([x for x in os.listdir(f"{merged}/valid/images") if not x.startswith(".")])
    log(f"Merged: {train_n} train, {val_n} val images ({remapped} labels remapped)")

    if train_n == 0:
        error("No training images found after merge!")

    return yaml_out


def train(data_yaml):
    """Train YOLOv8s model."""
    from ultralytics import YOLO

    log("Starting YOLOv8s training (this takes 1-2 hours)...")
    model = YOLO("yolov8s.pt")

    try:
        model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            patience=20,
            project="/content/runs/hold_detect",
            name="yolov8s",
            save=True,
            plots=True,
            verbose=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log("OOM with batch=16, retrying with batch=8...")
            model = YOLO("yolov8s.pt")
            model.train(
                data=data_yaml,
                epochs=100,
                imgsz=640,
                batch=8,
                patience=20,
                project="/content/runs/hold_detect",
                name="yolov8s",
                save=True,
                plots=True,
                verbose=True,
            )
        else:
            raise

    best = "/content/runs/hold_detect/yolov8s/weights/best.pt"
    if not os.path.exists(best):
        best = "/content/runs/hold_detect/yolov8s/weights/last.pt"
    if not os.path.exists(best):
        error("Training completed but no model file found!")

    log(f"Training done! Best model: {best}")
    return best


def evaluate(model_path, data_yaml):
    """Run validation and print metrics."""
    from ultralytics import YOLO

    log("Evaluating model...")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)

    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 50 + "\n")


def save_to_drive(model_path):
    """Save trained model to Google Drive."""
    save_path = "/content/drive/MyDrive/climbing-models/yolov8s-climbing-holds.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copy2(model_path, save_path)
    log(f"Model saved to Drive: {save_path}")
    log("Download from Drive and put in backend/models/ to use")


def main():
    print("\n" + "=" * 50)
    print("  CLIMBING HOLD DETECTOR TRAINING")
    print("=" * 50 + "\n")

    # Step 1: Mount Drive
    mount_drive()

    # Step 2: Find datasets
    yaml_paths = find_datasets()

    # Step 3: Copy to local
    local_dirs = copy_to_local(yaml_paths)

    # Step 4: Merge + remap labels
    data_yaml = merge_datasets(local_dirs)

    # Step 5: Train
    model_path = train(data_yaml)

    # Step 6: Evaluate
    evaluate(model_path, data_yaml)

    # Step 7: Save to Drive
    save_to_drive(model_path)

    print("\nDone! Next steps:")
    print("1. Download yolov8s-climbing-holds.pt from Drive")
    print("2. Put in backend/models/")
    print("3. Add to .env: HOLD_MODEL_PATH=models/yolov8s-climbing-holds.pt")
    print("4. Restart server")


if __name__ == "__main__":
    main()
