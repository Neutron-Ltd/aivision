import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    "video_path": "videos/cantee.mp4",
    "model_path": "runs/detect/aiv6s1/weights/best.pt",
    "window_name": "YOLOv8 Player",
    "conf_threshold": 0.50,
    "iou_threshold": 0.45,
    "imgsz": 640,
}

# ==============================================================================
# 🖥️ MAIN PLAYER
# ==============================================================================
def main():
    print(f"🚀 Loading Model: {CONFIG['model_path']}...")
    try:
        model = YOLO(CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"📂 Opening Video: {CONFIG['video_path']}...")
    cap = cv2.VideoCapture(CONFIG['video_path'])
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    # --- 1. SYNC PHYSICS (Copied from working code) ---
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps): orig_fps = 30
    realtime_delay = int(1000 / orig_fps)
    current_delay = realtime_delay

    # --- 2. RESOLUTION SETUP ---
    # We maintain aspect ratio but define a fixed display width
    target_width = 1280
    video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = video_h / video_w
    
    display_width = target_width
    display_height = int(display_width * aspect_ratio)

    # --- 3. EXPLICIT WINDOW INITIALIZATION (Critical Fix) ---
    # This specific sequence matches your working code to ensure the GUI backend loads
    cv2.namedWindow(CONFIG['window_name'], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG['window_name'], display_width, display_height)

    print("-" * 40)
    print("🎮 CONTROLS:")
    print("   [SPACE] : Pause / Resume")
    print("   [W]     : Speed UP (Lower delay)")
    print("   [S]     : Speed DOWN (Higher delay)")
    print("   [R]     : Reset Speed")
    print("   [Q]     : Quit")
    print("-" * 40)

    paused = False

    while True:
        # Check if window is closed
        try:
            if cv2.getWindowProperty(CONFIG['window_name'], cv2.WND_PROP_VISIBLE) < 1: 
                break
        except: 
            break

        # --- PAUSE LOGIC ---
        if paused:
            # Wait for Space to resume or Q to quit
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '): 
                paused = not paused
                print("▶️ Resumed")
            elif key == ord('q'):
                break
            continue

        # --- READ FRAME ---
        ret, frame = cap.read()
        if not ret:
            print("✅ Video Finished.")
            break

        # --- INFERENCE ---
        # Using .predict() as requested for pure detection
        results = model.predict(
            frame,
            conf=CONFIG['conf_threshold'],
            iou=CONFIG['iou_threshold'],
            imgsz=CONFIG['imgsz'],
            verbose=False
        )

        # --- DRAWING ---
        # Plotting directly ensures we get the built-in labels/colors
        annotated_frame = results[0].plot()

        # Resize for display (Optional, but keeps window manageable)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # Add Speed Overlay
        speed_x = round(realtime_delay / max(1, current_delay), 1)
        cv2.putText(display_frame, f"Speed: {speed_x}x", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- SHOW FRAME ---
        cv2.imshow(CONFIG['window_name'], display_frame)
        
        # --- KEY HANDLING ---
        key = cv2.waitKey(current_delay) & 0xFF

        if key == ord('q'): 
            break
        elif key == ord(' '): 
            paused = not paused
            print("II Paused")
        elif key == ord('w'): # Speed UP
            current_delay = max(1, current_delay - 5)
        elif key == ord('s'): # Speed DOWN
            current_delay += 5
        elif key == ord('r'): # Reset
            current_delay = realtime_delay

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()