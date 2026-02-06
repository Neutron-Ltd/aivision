import time
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
from threading import Thread

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    # INPUT: Your Canteen Video (Looping)
    "video_path": "videos/canteen.mp4",
    
    # MODEL: Your Custom Top-Down Model
    "model_path": "runs/detect/aiv6s1/weights/best.pt",
    
    "conf_threshold": 0.50,
    "classes_to_track": [0], # 0 is standard for 'person' in YOLO
    
    # ZONES (Generic Setup for Testing)
    # You can adjust these coordinates (0.0 - 1.0) later
    "outer_zone": np.array([
        [0.1, 0.1], [0.9, 0.1], 
        [0.9, 0.9], [0.1, 0.9]
    ], np.float32),

    "inner_zone": np.array([
        [0.3, 0.3], [0.7, 0.3], 
        [0.7, 0.7], [0.3, 0.7]
    ], np.float32)
}

# ==============================================================================
# 🧠 TRACKING LOGIC (Simple Zone Check)
# ==============================================================================
class SafetySystem:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.is_inside_map = {} 

    def update_batch(self, ids, centers, inner_poly, outer_poly):
        if len(ids) == 0: return []

        # Check if points are inside polygons
        in_inner = [cv2.pointPolygonTest(inner_poly, tuple(map(int, c)), False) >= 0 for c in centers]
        in_outer = [cv2.pointPolygonTest(outer_poly, tuple(map(int, c)), False) >= 0 for c in centers]

        status_list = []
        for i, obj_id in enumerate(ids):
            prev = self.is_inside_map.get(obj_id, False)
            curr = prev

            # Logic: Enter if in inner, Exit if left outer
            if not prev:
                if in_inner[i]: 
                    curr = True
                    self.enter_count += 1
            else:
                if not in_outer[i]: 
                    curr = False
                    self.exit_count += 1
            
            self.is_inside_map[obj_id] = curr
            status_list.append(curr)
        
        return status_list

    def get_stats(self):
        return self.enter_count, self.exit_count

# ==============================================================================
# 🖥️ MAIN WEB SERVER
# ==============================================================================
to_web_frame = None

def main():
    global to_web_frame

    print(f"🚀 Loading Model: {CONFIG['model_path']}...")
    try:
        model = YOLO(CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- FLASK SETUP ---
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def stream():
        return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_stream():
        while True:
            if to_web_frame is not None:
                # Encode frame to JPEG for browser
                ret, buffer = cv2.imencode('.jpg', to_web_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03) # Cap at ~30 FPS stream

    # Start Flask in background thread
    print("🌐 Starting Web Server...")
    Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8891, "debug": False}, daemon=True).start()

    # --- VIDEO LOOP ---
    print(f"📂 Opening Video: {CONFIG['video_path']}")
    cap = cv2.VideoCapture(CONFIG['video_path'])
    
    # Calculate Resolution
    width = 1280
    orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_w == 0: orig_w = 1280 # Fallback
    aspect = orig_h / orig_w
    height = int(width * aspect)
    
    # Prepare Zones
    outer_pts = (CONFIG['outer_zone'] * [width, height]).astype(np.int32)
    inner_pts = (CONFIG['inner_zone'] * [width, height]).astype(np.int32)

    safety_sys = SafetySystem()

    print("\n✅ STREAM RUNNING: http://localhost:8891 (or your server IP)")
    print("ℹ️  Use 'Ctrl+C' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (width, height))

        # --- GRID (Visual Helper) ---
        # Draws faint lines at 10%, 20%, etc.
        for i in range(1, 10):
            x, y = int(width * i/10), int(height * i/10)
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)

        # --- INFERENCE ---
        results = model.track(frame, persist=True, conf=CONFIG['conf_threshold'], 
                              classes=CONFIG['classes_to_track'], verbose=False)
        
        # Draw Zones
        cv2.polylines(frame, [outer_pts], True, (0, 255, 0), 2)
        cv2.polylines(frame, [inner_pts], True, (0, 255, 255), 2)

        current_active = 0

        # --- LOGIC ---
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            centers = np.column_stack(((boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2)).astype(int)

            active_states = safety_sys.update_batch(track_ids, centers, inner_pts, outer_pts)

            for box, obj_id, is_active, center in zip(boxes, track_ids, active_states, centers):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if is_active else (0, 255, 0)
                if is_active: current_active += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- HUD ---
        enter_c, exit_c = safety_sys.get_stats()
        cv2.putText(frame, f"ENTER: {enter_c} | EXIT: {exit_c} | ACTIVE: {current_active}", 
                    (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update global frame for Flask
        to_web_frame = frame

    cap.release()

if __name__ == "__main__":
    main()