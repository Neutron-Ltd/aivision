import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    # PATHS
    "video_path": "videos/canteen.mp4",
    "model_path": "runs/detect/aiv6s1/weights/best.pt",
    
    # WINDOW
    "window_name": "YOLOv8 Player",
    "window_width": 1280,  # Force window size
    
    # DETECTION PARAMETERS
    "conf": 0.50,          # Confidence Threshold (0.0 - 1.0)
    "iou": 0.45,           # NMS IoU Threshold (0.0 - 1.0)
    "imgsz": 640,          # Inference Size
    "classes": None,       # Set to [0] to detect only people, or None for all
}

# ==============================================================================
# 🖥️ MAIN PLAYER
# ==============================================================================
def main():
    print(f"🚀 Loading Model: {CONFIG['model_path']}...")
    try:
        model = YOLO(CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"📂 Opening Video: {CONFIG['video_path']}...")
    cap = cv2.VideoCapture(CONFIG['video_path'])
    if not cap.isOpened():
        print(f"❌ Error: Could not open {CONFIG['video_path']}")
        return

    # 1. Setup Playback Physics
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps): orig_fps = 30
    
    realtime_delay = int(1000 / orig_fps) # Standard delay for 1.0x speed
    current_delay = realtime_delay        # Current delay (adjustable)

    # 2. Setup Window (Linux Friendly)
    # We calculate height to keep aspect ratio
    aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    window_h = int(CONFIG['window_width'] * aspect_ratio)

    cv2.namedWindow(CONFIG['window_name'], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG['window_name'], CONFIG['window_width'], window_h)

    print("-" * 40)
    print("🎮 CONTROLS:")
    print("   [SPACE] : Pause / Resume")
    print("   [W]     : Speed UP")
    print("   [S]     : Speed DOWN")
    print("   [R]     : Reset Speed")
    print("   [Q]     : Quit")
    print("-" * 40)

    paused = False

    while True:
        # A. Check if Window was closed manually (The 'X' button)
        try:
            if cv2.getWindowProperty(CONFIG['window_name'], cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

        # B. Handle Pause State
        if paused:
            # Wait 100ms repeatedly for a key press
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '): 
                paused = not paused
                print("▶️ Resumed")
            elif key == ord('q'):
                break
            continue

        # C. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("✅ Video Finished.")
            break

        # D. Run Detection (Simple Inference)
        results = model.predict(
            frame, 
            conf=CONFIG['conf'], 
            iou=CONFIG['iou'], 
            imgsz=CONFIG['imgsz'],
            classes=CONFIG['classes'],
            verbose=False
        )

        # E. Draw Results
        # .plot() draws the boxes on the frame for us
        annotated_frame = results[0].plot()

        # Optional: Draw Speed Info
        speed_x = round(realtime_delay / max(1, current_delay), 1)
        cv2.putText(annotated_frame, f"Speed: {speed_x}x", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # F. Show Frame
        cv2.imshow(CONFIG['window_name'], annotated_frame)
        
        # G. Handle Keyboard Input
        key = cv2.waitKey(current_delay) & 0xFF

        if key == ord('q'): 
            break
        elif key == ord(' '): 
            paused = not paused
            print("II Paused")
        elif key == ord('w'): # Speed UP (Lower delay)
            current_delay = max(1, current_delay - 5)
        elif key == ord('s'): # Speed DOWN (Higher delay)
            current_delay += 5
        elif key == ord('r'): # Reset
            current_delay = realtime_delay

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()