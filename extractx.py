"""Extract frames around a specific timestamp, e.g. to collect hard examples."""

import argparse
import os
import sys

import cv2


def time_to_seconds(time_str):
    """Convert 'MM:SS' or 'HH:MM:SS' to seconds."""
    try:
        parts = [int(p) for p in time_str.split(":")]
    except ValueError:
        parts = []
    if len(parts) not in (2, 3) or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"invalid time '{time_str}', use MM:SS or HH:MM:SS")
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("time", nargs="?", default="28:42", help="target timestamp MM:SS (default: %(default)s)")
    p.add_argument("-v", "--video", default="videos/canteen2.mp4", help="input video (default: %(default)s)")
    p.add_argument("-o", "--output", default="images/canteen3", help="output folder (default: %(default)s)")
    p.add_argument("-c", "--context", type=float, default=2.0,
                   help="seconds to capture before and after the timestamp (default: %(default)s)")
    p.add_argument("-r", "--rate", type=float, default=5.0,
                   help="frames per second to capture (default: %(default)s)")
    return p.parse_args()


def extract_from_timestamp(args):
    try:
        target_seconds = time_to_seconds(args.time)
    except argparse.ArgumentTypeError as e:
        sys.exit(f"❌ {e}")
    if args.rate <= 0 or args.context < 0:
        sys.exit("❌ --rate must be > 0 and --context must be >= 0")
    if not os.path.exists(args.video):
        sys.exit(f"❌ Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps:
        sys.exit("❌ Could not detect video FPS.")

    center_frame = int(target_seconds * fps)
    step_size = max(1, int(fps / args.rate))
    frames_per_side = int(args.rate * args.context)
    indices = [center_frame + i * step_size for i in range(-frames_per_side, frames_per_side + 1)]

    os.makedirs(args.output, exist_ok=True)
    print(f"📸 Capturing up to {len(indices)} frames around {args.time} (frame {center_frame})")

    clean_time = args.time.replace(":", "-")  # "10:56" -> "10-56"
    saved = 0
    for i, frame_idx in enumerate(indices):
        if not 0 <= frame_idx < total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(args.output, f"{clean_time}_{i + 1:02d}.jpg"), frame)
            saved += 1

    cap.release()
    print(f"🎉 {saved} images saved to '{args.output}'")


if __name__ == "__main__":
    extract_from_timestamp(parse_args())
