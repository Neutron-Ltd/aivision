"""Run detection on a whole video and save an annotated copy."""

import argparse
import os
import sys

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", default="videos/canteen2.mp4", help="input video (default: %(default)s)")
    p.add_argument("-o", "--output", help="output video (default: <input>_detected.mp4)")
    p.add_argument("-m", "--model", default="runs/detect/aiv6s3/weights/best.pt", help="trained weights")
    p.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def process_video(args):
    if not os.path.exists(args.video):
        sys.exit(f"❌ Input video not found: {args.video}")
    output = args.output or f"{os.path.splitext(args.video)[0]}_detected.mp4"

    print(f"🚀 Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"❌ Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📂 Processing: {args.video} ({width}x{height} @ {fps:.2f} FPS, "
          f"{total_frames or 'unknown'} frames)")

    # 'mp4v' is widely available; try 'avc1' if the output does not play.
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        sys.exit(f"❌ Could not create output video: {output}")

    # Read until the stream ends rather than trusting the reported frame
    # count, which is 0 or inaccurate for some containers.
    with tqdm(total=total_frames or None, unit="frames") as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            out.write(results[0].plot())
            bar.update(1)

    cap.release()
    out.release()
    print(f"✅ Saved to: {output}")


if __name__ == "__main__":
    process_video(parse_args())
