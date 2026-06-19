import cv2
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_FOLDER = PROJECT_ROOT / "frames"

OUTPUT_IMAGE_FOLDER = PROJECT_ROOT / "player_dataset" / "images" / "train"
OUTPUT_LABEL_FOLDER = PROJECT_ROOT / "player_dataset" / "labels" / "train"

CLASSES = {
    "1": ("player", 0),
    "2": ("goalkeeper", 1),
    "3": ("referee", 2),
}

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

image_paths = sorted([
    p for p in IMAGE_FOLDER.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

current_class_key = "1"
boxes = []
drawing = False
ix, iy = -1, -1
temp_box = None


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, temp_box, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        temp_box = (ix, iy, x, y)

        class_name, class_id = CLASSES[current_class_key]
        boxes.append((class_id, temp_box))
        temp_box = None


def bbox_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    x_min = max(0, min(x1, x2))
    y_min = max(0, min(y1, y2))
    x_max = min(img_w, max(x1, x2))
    y_max = min(img_h, max(y1, y2))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 0 or box_h <= 0:
        return None

    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    box_w = box_w / img_w
    box_h = box_h / img_h

    return x_center, y_center, box_w, box_h


def draw_boxes(img):
    display = img.copy()

    for class_id, bbox in boxes:
        x1, y1, x2, y2 = bbox
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)

        name = [k for k, v in CLASSES.items() if v[1] == class_id][0]
        class_name = CLASSES[name][0]

        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(display, class_name, (x_min, max(20, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if temp_box is not None:
        x1, y1, x2, y2 = temp_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return display


def save_labels(image_path, img_w, img_h):
    out_img_path = Path(OUTPUT_IMAGE_FOLDER) / image_path.name
    out_label_path = Path(OUTPUT_LABEL_FOLDER) / f"{image_path.stem}.txt"

    with open(out_label_path, "w") as f:
        for class_id, bbox in boxes:
            yolo_box = bbox_to_yolo(bbox, img_w, img_h)
            if yolo_box is None:
                continue

            x_center, y_center, box_w, box_h = yolo_box
            f.write(f"{class_id} {x_center} {y_center} {box_w} {box_h}\n")

    shutil.copy(str(image_path), str(out_img_path))
    print(f"Saved {len(boxes)} boxes:", image_path.name)


cv2.namedWindow("Player Labeler")
cv2.setMouseCallback("Player Labeler", mouse_callback)

for image_path in image_paths:
    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]
    boxes = []
    temp_box = None

    while True:
        display = draw_boxes(img)

        class_name, class_id = CLASSES[current_class_key]

        cv2.putText(display, f"Image: {image_path.name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, f"Current class: {class_name}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display,
                    "1=player | 2=goalkeeper | 3=referee | U=undo | S=save next | N=skip | Q=quit",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Player Labeler", display)
        key = cv2.waitKey(20) & 0xFF

        if chr(key) in CLASSES:
            current_class_key = chr(key)

        elif key == ord("u"):
            if boxes:
                removed = boxes.pop()
                print("Removed last box:", removed)

        elif key == ord("s"):
            save_labels(image_path, w, h)
            break

        elif key == ord("n"):
            print("Skipped:", image_path.name)
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Done labeling.")
