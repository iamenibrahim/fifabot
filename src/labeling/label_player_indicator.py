import cv2
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_FOLDER = PROJECT_ROOT / "frames"

OUTPUT_IMAGE_FOLDER = PROJECT_ROOT / "indicator_dataset" / "images" / "train"
OUTPUT_LABEL_FOLDER = PROJECT_ROOT / "indicator_dataset" / "labels" / "train"

CLASS_ID = 0  # controlled_indicator

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

image_paths = sorted([
    p for p in IMAGE_FOLDER.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

drawing = False
ix, iy = -1, -1
bbox = None

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x, y)

def save_yolo_label(label_path, bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    x_min = max(0, min(x1, x2))
    y_min = max(0, min(y1, y2))
    x_max = min(img_w, max(x1, x2))
    y_max = min(img_h, max(y1, y2))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 0 or box_h <= 0:
        return False

    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    box_w = box_w / img_w
    box_h = box_h / img_h

    with open(label_path, "w") as f:
        f.write(f"{CLASS_ID} {x_center} {y_center} {box_w} {box_h}\n")

    return True

cv2.namedWindow("Label Controlled Indicator")
cv2.setMouseCallback("Label Controlled Indicator", mouse_callback)

for image_path in image_paths:
    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]
    bbox = None

    while True:
        display = img.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(display, image_path.name, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display,
                    "Draw box around controlled-player indicator | S=save | N=skip | Q=quit",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Label Controlled Indicator", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            if bbox is None:
                print("No box selected.")
                continue

            out_img_path = Path(OUTPUT_IMAGE_FOLDER) / image_path.name
            out_label_path = Path(OUTPUT_LABEL_FOLDER) / f"{image_path.stem}.txt"

            if save_yolo_label(out_label_path, bbox, w, h):
                shutil.copy(str(image_path), str(out_img_path))
                print("Saved:", image_path.name)
                break

        elif key == ord("n"):
            print("Skipped:", image_path.name)
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Done labeling.")
