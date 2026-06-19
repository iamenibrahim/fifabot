import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = PROJECT_ROOT / "yolov8s.pt"
DATA = PROJECT_ROOT / "configs" / "fifa_objects.yaml"
DATASET = PROJECT_ROOT / "fifa_objects_dataset"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
WORKERS = 0
RUN_NAME = "fifa_objects_v1"


def write_runtime_data_yaml() -> Path:
    runtime_data = PROJECT_ROOT / "configs" / "fifa_objects.runtime.yaml"
    runtime_data.write_text(
        "\n".join(
            [
                f"path: {DATASET.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: player",
                "  1: goalkeeper",
                "  2: referee",
                "  3: ball",
                "  4: controlled_player_indicator",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()

    if not (DATASET / "images" / "train").exists() or not (
        DATASET / "images" / "val"
    ).exists():
        raise FileNotFoundError(
            f"Combined dataset not found at {DATASET}. "
            "Run src/datasets/build_fifa_objects_dataset.py first."
        )

    data_yaml = write_runtime_data_yaml()
    model = YOLO(str(MODEL))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
