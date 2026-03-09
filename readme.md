# aivision6s

Top-down human detection using YOLOv8, optimized for overhead camera views (canteens, factory floors, etc.).

## overview

This project trains a YOLOv8 model to detect people from a **top-down perspective**. The dataset and augmentations are tailored for overhead footage where orientation and scale matter differently than in side-view surveillance.

## dataset

- **Source**: custom top-down person dataset
- **Classes**: 1 (person)
- **Images**: ~2,400 (train/valid/test split)
- **Format**: YOLOv8, 640×640

Dataset config: `dataset/data.yaml`

## project structure

| File | Purpose |
|------|---------|
| `extract.py` | Extract frames from video at fixed intervals (e.g. 1 per second) |
| `extractx.py` | Extract frames around a specific timestamp (MM:SS) with context |
| `train.py` | Train YOLOv8 on the dataset with top-down augmentations |
| `record.py` | Run detection on a full video and save an annotated output |
| `video.py` | Flask web player with live detection, pause, and speed controls |
| `export.py` | Export ONNX to RKNN for Rockchip RK3588 NPU deployment |

## setup

```bash
pip install ultralytics opencv-python flask torch tqdm
# For RKNN export: pip install rknn-toolkit2
```

## usage

### 1. extract frames from video

```bash
# Edit extract.py: VIDEO_PATH, OUTPUT_FOLDER, INTERVAL_SECONDS
python extract.py

# Or around a specific timestamp (extractx.py)
# Edit: VIDEO_PATH, OUTPUT_FOLDER, TARGET_TIME (MM:SS)
python extractx.py
```

### 2. train model

```bash
python train.py
```

Outputs to `runs/detect/<name>/weights/best.pt`. Uses GPU/MPS if available, otherwise CPU. Top-down augmentations include `flipud`, `fliplr`, and `degrees=180`.

### 3. process video (save annotated output)

```bash
# Edit record.py: video_path, output_path, model_path
python record.py
```

### 4. web player (live detection)

```bash
# Edit video.py: video_path, model_path
python video.py
# Open http://localhost:8890
```

Pause, slow down, and speed up playback in the browser.

### 5. export for edge (RK3588)

```bash
# Edit export.py: ONNX_MODEL path
python export.py
```

Produces an `.rknn` file for Rockchip RK3588 NPU inference.

## configuration

Detection parameters (in `record.py` and `video.py`):

- `conf`: confidence threshold (default 0.45–0.50)
- `iou`: NMS IoU threshold (default 0.45)
- `imgsz`: input size (default 640)

## model paths

- Default trained model: `runs/detect/aiv6s3/weights/best.pt`
- ONNX export from `train.py`: `runs/detect/<name>/weights/best.onnx`
