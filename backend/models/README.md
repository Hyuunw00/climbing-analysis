# Models Directory

Place trained model weights here.

## Hold Detection Model

Expected path: `hold_detector.pt` (YOLOv8 format)

### Quick start with pretrained YOLOv8

For testing, you can use the base YOLOv8n model:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=pt
cp yolov8n.pt backend/models/hold_detector.pt
```

### Training a custom hold detector

1. Prepare a dataset in YOLO format (images + labels with bounding boxes around holds).
2. Train:

```bash
yolo detect train data=holds.yaml model=yolov8n.pt epochs=100 imgsz=640
```

3. Copy the best weights:

```bash
cp runs/detect/train/weights/best.pt backend/models/hold_detector.pt
```

### Configuration

Set these environment variables (or in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOLD_MODEL_PATH` | `models/hold_detector.pt` | Path to the model file |
| `HOLD_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for detections |
