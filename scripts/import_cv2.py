import cv2
import os
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

frames_dir = str(PROJECT_ROOT / "frames")
dataset_dir = str(PROJECT_ROOT / "dataset")
csv_path = str(PROJECT_ROOT / "labels.csv")

labels = {
    ord("1"): "open_play",
    ord("2"): "corner",
    ord("3"): "free_kick",
    ord("4"): "throw_in",
    ord("5"): "goal_kick",
    ord("6"): "kickoff",
    ord("7"): "penalty",
    ord("8"): "cutscene",
    ord("9"): "menu",
    ord("0"): "waiting",
}

for label in labels.values():
    os.makedirs(os.path.join(dataset_dir, label), exist_ok=True)

images = [
    f for f in os.listdir(frames_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

images.sort()

with open(csv_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)

    for img_name in images:
        img_path = os.path.join(frames_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        display = img.copy()

        instructions = "1 open | 2 corner | 3 free | 4 throw | 5 goal | 6 kickoff | 7 pen | 8 cut | 9 menu | 0 wait | q quit"
        cv2.putText(display, instructions, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("FIFA Labeler", display)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        if key in labels:
            label = labels[key]
            dest_path = os.path.join(dataset_dir, label, img_name)

            shutil.copy(img_path, dest_path)
            writer.writerow([img_name, label])

            print(f"{img_name} -> {label}")

cv2.destroyAllWindows()
