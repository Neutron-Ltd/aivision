"""Convert an ONNX model to RKNN for the Rockchip RK3588 NPU (requires rknn-toolkit2)."""

import argparse
import os
import sys

from rknn.api import RKNN


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx", nargs="?", default="runs/detect/aiv6s5/weights/best.onnx",
                   help="input ONNX model (default: %(default)s)")
    p.add_argument("-o", "--output", help="output .rknn file (default: next to the ONNX file)")
    p.add_argument("--platform", default="rk3588", help="target platform (default: %(default)s)")
    return p.parse_args()


def export_rknn(args):
    output = args.output or os.path.splitext(args.onnx)[0] + ".rknn"
    rknn = RKNN(verbose=True)
    try:
        # Ultralytics expects pixel values scaled to 0-1.
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform)

        print(f"--> Loading {args.onnx}")
        if rknn.load_onnx(model=args.onnx) != 0:
            sys.exit("❌ Load model failed!")

        print("--> Building model (this takes 1-2 minutes)...")
        # No quantization keeps FP16: higher accuracy and usually fast enough.
        if rknn.build(do_quantization=False) != 0:
            sys.exit("❌ Build model failed!")

        print(f"--> Exporting to {output}")
        if rknn.export_rknn(output) != 0:
            sys.exit("❌ Export rknn model failed!")
    finally:
        rknn.release()

    print(f"✅ Done! Model written to {output}")


if __name__ == "__main__":
    export_rknn(parse_args())
