import cv2
from ultralytics import YOLO

# --- ⚙️ CONFIGURATION PARAMETERS ---
VIDEO_PATH = "videos/cantee.mp4"
MODEL_PATH = "runs/detect/aiv6s1/weights/best.pt"

# Detection Settings
CONF_THRESHOLD = 0.50   # Confidence required to see an object (0.0 - 1.0)
IOU_THRESHOLD = 0.45    # Overlap required to filter duplicates (0.0 - 1.0)
IMAGE_SIZE = 640        # Inference resolution (640 is standard)

# Playback Settings
PLAYBACK_SPEED = 1.0    # 1.0 = Normal, 0.5 = Slow Motion, 2.0 = Fast Forward
WINDOW_NAME = "YOLOv8 Detection Player"
# -----------------------------------

def main():
    # 1. Load the Model
    print(f"🚀 Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties for accurate timing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback if FPS is missing
    
    # Calculate delay per frame (in milliseconds)
    # If speed is 1.0, delay is 1000/fps. If speed is 0.5, delay is doubled.
    frame_delay = int((1000 / fps) / PLAYBACK_SPEED)
    if frame_delay < 1: frame_delay = 1

    print(f"▶️  Video Playing. Controls:")
    print(f"    [SPACE] or [P] : Pause / Resume")
    print(f"    [Q] or [ESC]   : Quit")

    paused = False

    while True:
        # A. If NOT paused, read and process the next frame
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("✅ Video finished.")
                break

            # --- RUN DETECTION ---
            results = model.predict(
                frame, 
                conf=CONF_THRESHOLD, 
                iou=IOU_THRESHOLD, 
                imgsz=IMAGE_SIZE, 
                verbose=False # Keep terminal clean
            )

            # --- DRAW RESULTS ---
            # .plot() returns a numpy array (image) with boxes drawn
            annotated_frame = results[0].plot()

            # Show the frame
            cv2.imshow(WINDOW_NAME, annotated_frame)

        # B. Handle Key Presses & Window Events
        # If paused, wait indefinitely (0) for a key.
        # If playing, wait for the calculated frame delay.
        wait_time = 0 if paused else frame_delay
        
        key = cv2.waitKey(wait_time) & 0xFF

        # [Q] or [ESC] to Quit
        if key == ord('q') or key == 27:
            break
        
        # [SPACE] or [P] to Pause/Resume
        if key == ord(' ') or key == ord('p'):
            paused = not paused
            status = "PAUSED II" if paused else "RESUMED ▶"
            print(f"Video {status}")

        # Check if the window was closed manually using the 'X' button
        # (This prevents the script from crashing if window is closed)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()