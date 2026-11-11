"""Label Studio ML backend that pre-annotates images with a trained YOLOv8 model.

Configuration (environment variables):
    MODEL_PATH    trained weights (default: runs/detect/aiv6s5/weights/best.pt)
    CONF          confidence threshold (default: 0.45)
    IOU           NMS IoU threshold (default: 0.45)
    BACKEND_HOST  bind address (default: 0.0.0.0)
    BACKEND_PORT  port (default: 9090)
"""

import os
import urllib.request

import cv2
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "runs/detect/aiv6s5/weights/best.pt"))
CONF = float(os.environ.get("CONF", 0.45))
IOU = float(os.environ.get("IOU", 0.45))
PERSON_CLASS = 0


def load_image(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        arr = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class YOLOBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Weights not found at: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

    def predict_task(self, task):
        url = task.get("data", {}).get("image", "")
        if not url:
            return []
        try:
            img = load_image(url)
        except OSError as e:
            print(f"⚠️ Could not fetch {url}: {e}")
            return []
        if img is None:
            print(f"⚠️ Could not decode {url}")
            return []

        results = self.model.predict(img, conf=CONF, iou=IOU, verbose=False)[0]
        h, w = results.orig_shape
        regions = []
        if results.boxes is None:
            return regions
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) != PERSON_CLASS:
                continue
            x1, y1, x2, y2 = box.tolist()
            regions.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": w,
                "original_height": h,
                "value": {
                    "x": x1 / w * 100,
                    "y": y1 / h * 100,
                    "width": (x2 - x1) / w * 100,
                    "height": (y2 - y1) / h * 100,
                    "rectanglelabels": ["person"],
                },
            })
        return regions

    def predict(self, tasks, **kwargs):
        # Label Studio matches predictions to tasks by position, so return
        # exactly one entry per task, even when an image cannot be loaded.
        return [{"result": self.predict_task(task)} for task in tasks]


if __name__ == "__main__":
    from label_studio_ml.api import init_app

    app = init_app(model_class=YOLOBackend)
    app.run(host=os.environ.get("BACKEND_HOST", "0.0.0.0"), port=int(os.environ.get("BACKEND_PORT", 9090)))
