# aivision6s

Top-down person detection with YOLOv8, tuned for overhead camera views such as canteens and factory floors, with export to the Rockchip RK3588 NPU.

## Overview

Most person detectors are trained on side-view footage. From directly above, people look like heads and shoulders, can face any direction, and stay roughly the same size. This project trains YOLOv8 for that view:

- 180° rotation and vertical/horizontal flips, because orientation carries no meaning from above
- Mosaic, mixup, shear and perspective turned off, so the model sees whole, undistorted frames
- Little scale jitter, so people stay close to their real on-camera size
- Colour jitter, so the model learns shape rather than clothing colour

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`export.py` also needs [`rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2), and the labelling tools need `label-studio` and `label-studio-ml`.

## Dataset

The dataset, videos and trained weights are **not included**. Bring your own top-down footage and labels in YOLOv8 format:

```
dataset/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

Example `dataset/data.yaml`:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 1
names: ['person']
```

## Workflow

Every script has `--help`. Defaults point at `videos/`, `images/` and `runs/`.

### 1. Extract frames from video

```bash
python extract.py videos/site.mp4 -o images/site -i 1        # one frame per second
python extractx.py 28:42 -v videos/site.mp4 -o images/hard   # frames around a timestamp
```

### 2. Label

`label/` runs Label Studio with a backend that pre-annotates images with your current model. See [label/label.md](label/label.md).

```bash
label/start.sh site     # serves images/site
```

### 3. Train

```bash
python train.py --name myrun                    # yolov8n, 300 epochs, ONNX export
python train.py --model yolov8s.pt --batch 16   # larger model, less memory
```

The device (CUDA, Apple MPS or CPU) is selected automatically. Output goes to `runs/detect/<name>/weights/best.pt` and `best.onnx`.

### 4. Run on video

```bash
# Save an annotated copy
python record.py videos/site.mp4 -m runs/detect/myrun/weights/best.pt

# Watch in the browser with pause and speed controls at http://localhost:8890
python video.py videos/site.mp4 -m runs/detect/myrun/weights/best.pt
```

`video.py` listens on `127.0.0.1` only. Pass `--host 0.0.0.0` to allow other machines on your network.

Detection options: `--conf` (confidence, default 0.45–0.50), `--iou` (NMS IoU, default 0.45), `--imgsz` (default 640).

### 5. Export for RK3588

```bash
python export.py runs/detect/myrun/weights/best.onnx    # writes best.rknn (FP16)
```

## Project structure

| File | Purpose |
|------|---------|
| `extract.py` | Extract frames from a video at a fixed interval |
| `extractx.py` | Extract frames around a timestamp |
| `train.py` | Train YOLOv8 with top-down augmentations |
| `record.py` | Save an annotated detection video |
| `video.py` | Browser player with live detection |
| `export.py` | Convert ONNX to RKNN for the RK3588 NPU |
| `label/` | Label Studio pre-annotation setup |

## License

[AGPL-3.0](LICENSE). This project builds on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is also AGPL-3.0.
