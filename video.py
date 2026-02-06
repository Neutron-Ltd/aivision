import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    "video_path": "videos/canteen.mp4", 
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

    # Sync Physics (From your working code)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps): orig_fps = 30
    realtime_delay = int(1000 / orig_fps)
    current_delay = realtime_delay

    # Resolution Setup
    target_width = 1280
    aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = target_width
    height = int(width * aspect_ratio)

    # Window Initialization (Exact sequence from your working code)
    cv2.namedWindow(CONFIG['window_name'], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG['window_name'], width, height)
    
    paused = False

    print("-" * 40)
    print("🎮 CONTROLS:")
    print("   [SPACE] : Pause / Resume")
    print("   [W]     : Speed UP (Lower delay)")
    print("   [S]     : Speed DOWN (Higher delay)")
    print("   [R]     : Reset Speed")
    print("   [Q]     : Quit")
    print("-" * 40)

    while True:
        try:
            if cv2.getWindowProperty(CONFIG['window_name'], cv2.WND_PROP_VISIBLE) < 1: break
        except: break

        if paused:
            # Wait 100ms for input while paused
            if cv2.waitKey(100) & 0xFF == ord(' '): paused = not paused
            continue

        ret, frame = cap.read()
        if not ret:
            # Loop video or break? (Currently stops, un-comment to loop)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            print("✅ Video Finished.")
            break

        # Resize frame to match window target
        frame = cv2.resize(frame, (width, height))

        # 🧠 INFERENCE (Simple Detection)
        # We use .predict() instead of .track() since we don't need IDs
        results = model.predict(
            frame, 
            conf=CONFIG['conf_threshold'], 
            iou=CONFIG['iou_threshold'],
            imgsz=CONFIG['imgsz'], 
            verbose=False
        )

        # Draw Results
        annotated_frame = results[0].plot()

        # Draw Speed Info Overlay
        speed_text = f"Speed: {round(realtime_delay / max(1, current_delay), 1)}x"
        cv2.putText(annotated_frame, speed_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show Frame using the exact config key
        cv2.imshow(CONFIG['window_name'], annotated_frame)
        
        # Key Handling
        key = cv2.waitKey(current_delay) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('w'): current_delay = max(1, current_delay - 5)
        elif key == ord('s'): current_delay += 5
        elif key == ord('r'): current_delay = realtime_delay

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()