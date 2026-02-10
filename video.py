import time
import cv2
import logging
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
from threading import Thread

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    "video_path": "videos/factory1.mp4",
    "model_path": "runs/detect/aiv6s3/weights/best.pt",
    
    # Detection Parameters
    "conf": 0.50,
    "iou": 0.45,
    "imgsz": 640,
}

# ==============================================================================
# 🎮 GLOBAL PLAYER STATE
# ==============================================================================
class PlayerState:
    def __init__(self):
        self.paused = False
        self.speed_multiplier = 1.0  # 1.0 = Normal, 0.5 = Slow, 2.0 = Fast
        self.frame = None

state = PlayerState()

# ==============================================================================
# 🖥️ FLASK APP & CONTROLS
# ==============================================================================
app = Flask(__name__)

# Minimal HTML Interface for Controls
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
        .active { background: #007bff; }
    </style>
    <script>
        function sendCmd(cmd) { fetch('/' + cmd); }
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

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- CONTROL ENDPOINTS ---
@app.route('/pause')
def pause():
    state.paused = not state.paused
    return "OK"

@app.route('/faster')
def faster():
    state.speed_multiplier = min(state.speed_multiplier + 0.5, 5.0)
    return "OK"

@app.route('/slower')
def slower():
    state.speed_multiplier = max(state.speed_multiplier - 0.5, 0.1)
    return "OK"

# ==============================================================================
# 🧠 VIDEO PROCESSING LOOP
# ==============================================================================
def generate_frames():
    # 1. Load Model
    print(f"🚀 Loading Model: {CONFIG['model_path']}")
    model = YOLO(CONFIG['model_path'])

    # 2. Open Video
    print(f"📂 Opening Video: {CONFIG['video_path']}")
    cap = cv2.VideoCapture(CONFIG['video_path'])
    
    # Calculate Base Delay (Physics)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    base_delay = 1.0 / fps

    while True:
        # A. PAUSE LOGIC
        if state.paused:
            # If paused, keep sending the LAST frame so the image doesn't break
            if state.frame is not None:
                yield state.frame
            time.sleep(0.1)
            continue

        # B. READ FRAME
        ret, frame = cap.read()
        if not ret:
            print("✅ Video Finished. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # C. INFERENCE
        results = model.predict(
            frame, 
            conf=CONFIG['conf'], 
            iou=CONFIG['iou'], 
            imgsz=CONFIG['imgsz'],
            verbose=False
        )

        # D. DRAWING
        annotated_frame = results[0].plot()

        # Add HUD (Speed Status)
        cv2.putText(annotated_frame, f"Speed: {state.speed_multiplier}x", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # E. ENCODE FOR WEB
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            state.frame = (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            yield state.frame

        # F. SPEED CONTROL (Sleep logic)
        # Higher multiplier = Less sleep = Faster
        # Lower multiplier = More sleep = Slower
        time.sleep(base_delay / state.speed_multiplier)

if __name__ == "__main__":
    # Disable Flask logs to keep terminal clean
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("\n✅ PLAYER READY!")
    print("   👉 Open this URL in your browser: http://localhost:8890")
    print("   (Or use your server IP: http://<server_ip>:8890)\n")
    
    app.run(host='0.0.0.0', port=8890, threaded=True)