import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

video_path = str(PROJECT_ROOT / "fifa_clip.mp4")
output_dir = str(PROJECT_ROOT / "frames")
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

duration_minutes = 11
interval_seconds = 1

saved_id = 0

for sec in range(0, duration_minutes * 60, interval_seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

    ret, frame = cap.read()
    if not ret:
        print(f"Failed at {sec} seconds")
        continue

    frame = cv2.resize(frame, (1280, 720))

    cv2.imwrite(
        f"{output_dir}/frame_{saved_id:04d}.jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    saved_id += 1
    print(f"Saved {saved_id} at {sec}s")

cap.release()

print(f"Saved {saved_id} images.")
