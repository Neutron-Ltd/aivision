import cv2
import os
import math

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
VIDEO_PATH = "videos/canteen2.mp4"
OUTPUT_FOLDER = "images/canteen2"
INTERVAL_SECONDS = 1  # How many seconds between each extracted image

def extract_frames():
    # 1. Verify Video Exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        return

    # 2. Create Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📂 Created folder: {OUTPUT_FOLDER}")

    # 3. Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)       # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("❌ Error: Could not read FPS from video.")
        return

    # Calculate frames to skip (e.g., if FPS is 30, skip 30 frames for 1 sec)
    frames_to_skip = int(fps * INTERVAL_SECONDS)
    
    print(f"ℹ️  Video FPS: {fps}")
    print(f"ℹ️  Total Frames: {total_frames}")
    print(f"ℹ️  Extracting every {frames_to_skip} frames ({INTERVAL_SECONDS}s interval)")

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Only save if the current frame count is a multiple of our interval
        if count % frames_to_skip == 0:
            # Create filename (e.g., frame_001.jpg)
            filename = f"{OUTPUT_FOLDER}/frame_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   Saved: {filename}")
            saved_count += 1

        count += 1

    cap.release()
    print(f"\n✅ Done! Extracted {saved_count} images to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    extract_frames()