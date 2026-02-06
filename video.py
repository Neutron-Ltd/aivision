import cv2
from ultralytics import YOLO

# --- ⚙️ USER CONFIGURATION ---
VIDEO_PATH = "videos/canteen.mp4"
MODEL_PATH = "runs/detect/aiv6s1/weights/best.pt"

# Initial Detection Settings
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# Initial Playback Settings
PLAYBACK_SPEED = 1.0     # 1.0 = Normal, 0.5 = Slow, 2.0 = Fast
WINDOW_NAME = "Linux AI Player"
# -----------------------------

def main():
    # 1. Load Model
    print(f"🚀 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {VIDEO_PATH}")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    base_delay = int(1000 / fps)

    print("-" * 50)
    print("🎮 PLAYER CONTROLS:")
    print("  [SPACE] : Pause / Resume")
    print("  [Q]     : Quit")
    print("  [W]     : Speed UP (+0.5x)")
    print("  [S]     : Speed DOWN (-0.5x)")
    print("-" * 50)

    paused = False
    current_speed = PLAYBACK_SPEED

    while True:
        # A. Read Frame (Only if not paused)
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video Finished.")
                break

            # --- DETECTION ---
            results = model.predict(
                frame, 
                conf=CONF_THRESHOLD, 
                iou=IOU_THRESHOLD, 
                imgsz=640, 
                verbose=False
            )
            
            # Draw boxes
            annotated_frame = results[0].plot()

            # Add Speed Overlay text
            cv2.putText(annotated_frame, f"Speed: {current_speed}x", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show Frame
            cv2.imshow(WINDOW_NAME, annotated_frame)

        # B. Smart Delay Logic
        # If paused, wait forever (0) for a key. 
        # If playing, calculate delay based on speed.
        if paused:
            delay = 0
        else:
            delay = int(base_delay / current_speed)
            if delay < 1: delay = 1

        # C. Key Handler
        key = cv2.waitKey(delay) & 0xFF

        # Quit
        if key == ord('q'):
            break
        
        # Pause
        elif key == ord(' '):
            paused = not paused
            state = "PAUSED" if paused else "PLAYING"
            print(f"⏯  {state}")

        # Speed Controls
        elif key == ord('w'): # Faster
            current_speed += 0.5
            print(f"⏩ Speed Up: {current_speed}x")
        
        elif key == ord('s'): # Slower
            if current_speed > 0.5:
                current_speed -= 0.5
                print(f"⏪ Speed Down: {current_speed}x")

        # Check if window closed via 'X' button
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            # Handle standard Linux cv2 error on window check
            pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()