import cv2
import os

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
VIDEO_PATH = "videos/canteen2.mp4"       # Path to your video
OUTPUT_FOLDER = "images/canteen3"        # Output folder
TARGET_TIME = "28:42"                    # Format -> "MM:SS" (Minutes:Seconds)

# Extraction Settings
CONTEXT_SECONDS = 2      # Capture 1 second before and 1 second after
EXTRACT_FPS = 5          # Capture 5 frames per second (Interval)

def time_to_seconds(time_str):
    """Converts 'MM:SS' string to total seconds."""
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return (minutes * 60) + seconds
    except ValueError:
        print(f"❌ Error: Invalid time format '{time_str}'. Use 'MM:SS'")
        return None

def extract_from_timestamp():
    # 1. Setup Video
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print("❌ Error: Could not detect Video FPS.")
        return

    # 2. Calculate Target Frame
    target_seconds = time_to_seconds(TARGET_TIME)
    if target_seconds is None: return

    center_frame = int(target_seconds * fps)
    
    print(f"⏱️  Target Time: {TARGET_TIME} ({target_seconds}s)")
    print(f"🎯 Center Frame: {center_frame}")
    print(f"ℹ️  Video FPS: {fps}")

    # 3. Calculate Steps
    # To get 5 images in 1 second, we skip (FPS / 5) frames
    step_size = int(fps / EXTRACT_FPS)
    if step_size < 1: step_size = 1

    # How many frames total to grab? (5 frames * number of seconds)
    frames_per_side = int(EXTRACT_FPS * CONTEXT_SECONDS)

    # 4. Prepare Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 5. Generate List of Frames to Capture
    # Range: -5 to +5 steps
    indices_to_grab = []
    for i in range(-frames_per_side, frames_per_side + 1):
        offset = i * step_size
        indices_to_grab.append(center_frame + offset)

    print(f"📸 Capturing {len(indices_to_grab)} frames around {TARGET_TIME}...")

    # 6. Extract Loop
    saved_count = 0
    clean_time_name = TARGET_TIME.replace(":", "-") # "10:56" -> "10-56"

    for i, frame_idx in enumerate(indices_to_grab):
        if frame_idx < 0 or frame_idx >= total_frames:
            continue # Skip out of bounds

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Name: 10-56_01.jpg, 10-56_02.jpg ...
            filename = f"{OUTPUT_FOLDER}/{clean_time_name}_{i+1:02d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   ✅ Saved: {filename}")
            saved_count += 1

    cap.release()
    print(f"\n🎉 Done! {saved_count} images saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    extract_from_timestamp()