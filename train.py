"""Train a YOLOv8 person detector tuned for top-down (overhead) camera views."""

import argparse

import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
        return 0
    if torch.backends.mps.is_available():
        print("🚀 Training on Apple Silicon (MPS)")
        return "mps"
    print("⚠️ Training on CPU (This will be slow)")
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset/data.yaml", help="dataset YAML (default: %(default)s)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="base weights; downloaded automatically if missing (default: %(default)s)")
    p.add_argument("--name", default="aiv6s5", help="run name under runs/detect/ (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50, help="early-stopping patience in epochs")
    p.add_argument("--batch", type=int, default=64, help="lower to 8 or 16 on out-of-memory errors")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0, cpu or mps (default: auto-detect)")
    p.add_argument("--no-export", action="store_true", help="skip ONNX export after training")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device is not None else pick_device()

    # yolov8n is small enough for edge NPUs; use yolov8s/m for more accuracy.
    model = YOLO(args.model)

    model.train(
        data=args.data,

        # --- TRAINING DYNAMICS ---
        epochs=args.epochs,       # Long training because aggressive augmentation is off
        patience=args.patience,   # Stop if no improvement for this many epochs
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        workers=args.workers,
        device=device,
        cache=True,               # Load images into RAM for speed
        exist_ok=True,            # Reuse the run folder if it exists
        amp=False,                # Full precision is more stable on small custom datasets

        # --- "NO FRAGMENTATION" AUGMENTATIONS ---
        mosaic=0.0,       # OFF. No chopping; the model only sees whole real frames.
        mixup=0.0,        # OFF. No transparent ghost overlays.

        # --- "NO ZOOM" AUGMENTATIONS ---
        scale=0.1,        # +/- 10% keeps people close to their real on-camera size.

        # --- TOP-DOWN ESSENTIALS ---
        degrees=180.0,    # A person walking south looks like one walking north.
        flipud=0.5,
        fliplr=0.5,

        # --- COLOUR & LIGHTING ---
        # Teach the model that shape matters more than clothing colour.
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,

        # --- GEOMETRY ---
        shear=0.0,        # Top-down views are flat; no perspective warping needed.
        perspective=0.0,
        translate=0.1,
    )

    if not args.no_export:
        # After training, `model` holds the best checkpoint.
        model.export(format="onnx", opset=12)
    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
