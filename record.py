import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm  # Library for progress bar

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
CONFIG = {
    "video_path": "videos/canteen2.mp4",             # Your input video (converted from .dav)
    "output_path": "videos/canteen2_detected.mp4",   # The result file name
    "model_path": "runs/detect/aiv6s2/weights/best.pt", # Your trained model
    
    # Detection Parameters
    "conf": 0.45,
    "iou": 0.45,
    "imgsz": 640,
}

def process_video():
    # 1. Load Model
    print(f"🚀 Loading Model: {CONFIG['model_path']}")
    try:
        model = YOLO(CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Open Source Video
    if not os.path.exists(CONFIG['video_path']):
        print(f"❌ Error: Input video not found at {CONFIG['video_path']}")
        return

    cap = cv2.VideoCapture(CONFIG['video_path'])
    
    # Get Video Properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("⚠️ Warning: Could not determine total frames. Progress bar might be inaccurate.")

    print(f"📂 Processing: {CONFIG['video_path']}")
    print(f"   ℹ️ Resolution: {width}x{height}")
    print(f"   ℹ️ FPS: {fps}")
    print(f"   ℹ️ Total Frames: {total_frames}")

    # 3. Setup Video Writer
    # 'mp4v' is a standard codec. If it fails, try 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(CONFIG['output_path'], fourcc, fps, (width, height))

    # 4. Processing Loop
    print("\n🎥 Starting Inference & Saving...")
    
    # Wrap loop in tqdm for a progress bar
    for _ in tqdm(range(total_frames), unit="frames"):
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # A. Run Inference
        results = model.predict(
            frame, 
            conf=CONFIG['conf'], 
            iou=CONFIG['iou'], 
            imgsz=CONFIG['imgsz'],
            verbose=False
        )

        # B. Draw Bounding Boxes
        # plot() creates a new image with the boxes drawn
        annotated_frame = results[0].plot()

        # C. Write to Output Video
        out.write(annotated_frame)

    # 5. Cleanup
    cap.release()
    out.release()
    print(f"\n✅ Done! Saved to: {CONFIG['output_path']}")
    print("   (You can now download this file to your computer to view it)")

if __name__ == "__main__":
    # Ensure tqdm is installed: pip install tqdm
    process_video()