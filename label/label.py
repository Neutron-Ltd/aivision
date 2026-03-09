import os
import urllib.request
import cv2
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

MODEL_PATH = os.path.abspath("runs/detect/aiv6s5/weights/best.pt")

class AIV6S5Backend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Weights not found at: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            url = task.get("data", {}).get("image", "")
            if not url:
                continue
            
            # Fetch image from your local HTTP server
            req = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            results = self.model.predict(img, conf=0.45, iou=0.45, verbose=False)[0]
            h, w = results.orig_shape
            
            result = []
            if results.boxes is not None:
                for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                    if int(cls) != 0:  # 0 is 'person' in standard YOLO
                        continue
                    
                    x1, y1, x2, y2 = box
                    result.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "original_width": w,
                        "original_height": h,
                        "value": {
                            "x": (x1 / w) * 100,
                            "y": (y1 / h) * 100,
                            "width": ((x2 - x1) / w) * 100,
                            "height": ((y2 - y1) / h) * 100,
                            "rectanglelabels": ["person"]
                        }
                    })
            predictions.append({"result": result})
        return predictions

if __name__ == "__main__":
    from label_studio_ml.api import init_app
    app = init_app(model_class=AIV6S5Backend)
    app.run(host="0.0.0.0", port=9090)