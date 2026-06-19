import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = PROJECT_ROOT / "indicator_dataset"

IMG_TRAIN = BASE / "images" / "train"
LBL_TRAIN = BASE / "labels" / "train"

IMG_VAL = BASE / "images" / "val"
LBL_VAL = BASE / "labels" / "val"

VAL_RATIO = 0.2
random.seed(42)

IMG_VAL.mkdir(parents=True, exist_ok=True)
LBL_VAL.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png"}

images = [
    p for p in IMG_TRAIN.iterdir()
    if p.suffix.lower() in image_exts
]

random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_images = images[:val_count]

for img_path in val_images:
    label_path = LBL_TRAIN / f"{img_path.stem}.txt"

    shutil.move(str(img_path), str(IMG_VAL / img_path.name))

    if label_path.exists():
        shutil.move(str(label_path), str(LBL_VAL / label_path.name))
    else:
        print("Missing label:", img_path.name)

print("Done.")
print("Train images:", len([p for p in IMG_TRAIN.iterdir() if p.suffix.lower() in image_exts]))
print("Val images:", len([p for p in IMG_VAL.iterdir() if p.suffix.lower() in image_exts]))
print("Train labels:", len(list(LBL_TRAIN.glob("*.txt"))))
print("Val labels:", len(list(LBL_VAL.glob("*.txt"))))
