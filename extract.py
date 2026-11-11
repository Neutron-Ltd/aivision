"""Extract frames from a video at a fixed time interval (e.g. one per second)."""

import argparse
import os
import sys

import cv2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", default="videos/test.mp4", help="input video (default: %(default)s)")
    p.add_argument("-o", "--output", default="images/test", help="output folder (default: %(default)s)")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between extracted frames (default: %(default)s)")
    return p.parse_args()


def extract_frames(args):
    if not os.path.exists(args.video):
        sys.exit(f"❌ Video file not found: {args.video}")
    if args.interval <= 0:
        sys.exit("❌ --interval must be greater than 0")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        sys.exit("❌ Could not read FPS from video.")
    os.makedirs(args.output, exist_ok=True)

    # At least one frame apart, so very short intervals keep every frame.
    frames_to_skip = max(1, round(fps * args.interval))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"ℹ️  {fps:.2f} FPS, {total_frames} frames; saving every {frames_to_skip} frames")

    count = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frames_to_skip == 0:
            filename = os.path.join(args.output, f"frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"✅ Extracted {saved} images to '{args.output}'")


if __name__ == "__main__":
    extract_frames(parse_args())
