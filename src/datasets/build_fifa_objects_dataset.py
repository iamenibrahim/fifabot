import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DATASET = PROJECT_ROOT / "fifa_objects_dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42

SOURCES = [
    {
        "name": "player",
        "root": PROJECT_ROOT / "player_dataset",
        "class_map": {0: 0, 1: 1, 2: 2},
    },
    {
        "name": "ball",
        "root": PROJECT_ROOT / "ball_dataset",
        "class_map": {0: 3},
    },
    {
        "name": "indicator",
        "root": PROJECT_ROOT / "indicator_dataset",
        "class_map": {0: 4},
    },
]


def image_files(path: Path):
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def has_existing_split(dataset_root: Path) -> bool:
    return (dataset_root / "images" / "train").exists() or (
        dataset_root / "images" / "val"
    ).exists()


def split_unsplit_images(dataset_root: Path):
    images_dir = dataset_root / "images"
    images = image_files(images_dir)
    random.Random(RANDOM_SEED).shuffle(images)
    val_count = max(1, int(len(images) * 0.2)) if images else 0
    val_names = {p.name for p in images[:val_count]}
    return {
        "train": [p for p in images if p.name not in val_names],
        "val": [p for p in images if p.name in val_names],
    }


def source_images_by_split(source):
    root = source["root"]
    if has_existing_split(root):
        return {
            "train": image_files(root / "images" / "train"),
            "val": image_files(root / "images" / "val"),
        }
    return split_unsplit_images(root)


def label_path_for(source_root: Path, split: str, image_path: Path) -> Path:
    split_label = source_root / "labels" / split / f"{image_path.stem}.txt"
    if split_label.exists():
        return split_label
    return source_root / "labels" / f"{image_path.stem}.txt"


def remap_label_lines(label_path: Path, class_map: dict, source_name: str):
    remapped = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue

        try:
            source_class = int(float(parts[0]))
        except ValueError:
            print(f"WARNING: {label_path}:{line_number} has invalid class id")
            continue

        if source_class not in class_map:
            print(
                f"WARNING: {label_path}:{line_number} class {source_class} "
                f"is not mapped for {source_name}; skipping"
            )
            continue

        remapped.append(" ".join([str(class_map[source_class]), *parts[1:]]))

    return remapped


def build_dataset():
    for split in ("train", "val"):
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    pending_labels = defaultdict(list)
    copied_images = set()
    class_counts = Counter()
    skipped_missing_labels = 0
    copied_count = 0

    for source in SOURCES:
        split_images = source_images_by_split(source)

        for split, images in split_images.items():
            for image_path in images:
                label_path = label_path_for(source["root"], split, image_path)
                if not label_path.exists():
                    skipped_missing_labels += 1
                    print(f"WARNING: missing label for {image_path}; skipping")
                    continue

                label_lines = remap_label_lines(
                    label_path,
                    source["class_map"],
                    source["name"],
                )
                if not label_lines:
                    print(f"WARNING: no usable labels in {label_path}; skipping")
                    continue

                output_name = image_path.name
                output_image = OUTPUT_DATASET / "images" / split / output_name
                output_label = OUTPUT_DATASET / "labels" / split / f"{Path(output_name).stem}.txt"

                if output_image not in copied_images:
                    shutil.copy2(image_path, output_image)
                    copied_images.add(output_image)
                    copied_count += 1

                pending_labels[output_label].extend(label_lines)
                for label_line in label_lines:
                    class_counts[int(label_line.split()[0])] += 1

    for label_path, lines in pending_labels.items():
        label_path.write_text("\n".join(lines) + "\n")

    print("\nCombined FIFA objects dataset built.")
    print(f"Images copied: {copied_count}")
    print(f"Label files written: {len(pending_labels)}")
    print(f"Missing labels skipped: {skipped_missing_labels}")
    print("Class counts:")
    for class_id in range(5):
        print(f"  {class_id}: {class_counts[class_id]}")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_dataset()


if __name__ == "__main__":
    main()
