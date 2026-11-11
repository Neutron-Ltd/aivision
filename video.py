"""Browser-based video player with live YOLOv8 detection and pause/speed controls."""

import argparse
import logging
import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

HTML_TEMPLATE = """
<html>
<head>
    <title>YOLOv8 Web Player</title>
    <style>
        body { background-color: #1e1e1e; color: white; font-family: sans-serif; text-align: center; }
        img { border: 2px solid #333; max-width: 90%; height: auto; margin-top: 20px; }
        .controls { margin-top: 20px; }
        button { padding: 10px 20px; font-size: 16px; margin: 0 10px; cursor: pointer; background: #444; color: white; border: none; border-radius: 5px; }
        button:hover { background: #666; }
    </style>
    <script>
        function sendCmd(cmd) { fetch('/' + cmd, { method: 'POST' }); }
    </script>
</head>
<body>
    <h1>YOLOv8 Detection Player</h1>
    <img src="/video_feed">
    <div class="controls">
        <button onclick="sendCmd('slower')">⏪ Slower</button>
        <button onclick="sendCmd('pause')">⏯ Pause/Resume</button>
        <button onclick="sendCmd('faster')">⏩ Faster</button>
    </div>
</body>
</html>
"""


class PlayerState:
    def __init__(self):
        self.paused = False
        self.speed_multiplier = 1.0  # 1.0 = normal, 0.5 = slow, 2.0 = fast
        self.lock = threading.Lock()


def create_app(args):
    model = YOLO(args.model)  # loaded once and shared by all viewers
    state = PlayerState()
    app = Flask(__name__)

    def generate_frames():
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Could not open video: {args.video}")
            return
        base_delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)
        last_frame = None
        try:
            while True:
                if state.paused:
                    # Keep resending the last frame so the image does not break.
                    if last_frame is not None:
                        yield last_frame
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    continue

                with state.lock:
                    results = model.predict(frame, conf=args.conf, iou=args.iou,
                                            imgsz=args.imgsz, verbose=False)
                annotated = results[0].plot()
                cv2.putText(annotated, f"Speed: {state.speed_multiplier}x", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    last_frame = (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                  + buffer.tobytes() + b"\r\n")
                    yield last_frame

                time.sleep(base_delay / state.speed_multiplier)
        finally:
            cap.release()  # runs when the browser disconnects

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/pause", methods=["POST"])
    def pause():
        state.paused = not state.paused
        return "OK"

    @app.route("/faster", methods=["POST"])
    def faster():
        state.speed_multiplier = min(state.speed_multiplier + 0.5, 5.0)
        return "OK"

    @app.route("/slower", methods=["POST"])
    def slower():
        state.speed_multiplier = max(state.speed_multiplier - 0.5, 0.1)
        return "OK"

    return app


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", default="videos/factory1.mp4", help="input video (default: %(default)s)")
    p.add_argument("-m", "--model", default="runs/detect/aiv6s3/weights/best.pt", help="trained weights")
    p.add_argument("--conf", type=float, default=0.50, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; use 0.0.0.0 to allow other machines (default: %(default)s)")
    p.add_argument("--port", type=int, default=8890)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.video):
        sys.exit(f"❌ Video not found: {args.video}")
    app = create_app(args)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"✅ Player ready: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
