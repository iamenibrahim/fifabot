import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "ball_dataset"
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABEL_DIR = DATASET_DIR / "labels" / "train"

VAL_IMG_DIR = DATASET_DIR / "images" / "val"
VAL_LABEL_DIR = DATASET_DIR / "labels" / "val"

VAL_RATIO = 0.2

VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

images = [
    p for p in TRAIN_IMG_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_images = images[:val_count]

print(f"Total images: {len(images)}")
print(f"Moving to val: {len(val_images)}")

for img_path in val_images:
    label_path = TRAIN_LABEL_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"Missing label for {img_path.name}, skipping")
        continue

    shutil.move(str(img_path), str(VAL_IMG_DIR / img_path.name))
    shutil.move(str(label_path), str(VAL_LABEL_DIR / label_path.name))

print("Done.")
