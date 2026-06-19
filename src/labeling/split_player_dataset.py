import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = PROJECT_ROOT / "player_dataset"

IMG_TRAIN = BASE / "images" / "train"
LBL_TRAIN = BASE / "labels" / "train"

IMG_VAL = BASE / "images" / "val"
LBL_VAL = BASE / "labels" / "val"

VAL_RATIO = 0.2

IMG_VAL.mkdir(parents=True, exist_ok=True)
LBL_VAL.mkdir(parents=True, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png"]

images = [
    p for p in IMG_TRAIN.iterdir()
    if p.suffix.lower() in image_extensions
]

random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_images = images[:val_count]

print("Total images:", len(images))
print("Moving to val:", len(val_images))

for img_path in val_images:
    label_path = LBL_TRAIN / f"{img_path.stem}.txt"

    new_img_path = IMG_VAL / img_path.name
    new_label_path = LBL_VAL / label_path.name

    shutil.move(str(img_path), str(new_img_path))

    if label_path.exists():
        shutil.move(str(label_path), str(new_label_path))
    else:
        print("Missing label for:", img_path.name)

print("Done.")
