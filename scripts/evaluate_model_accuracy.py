import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_MODEL_GROUPS = {
    "ball": [
        "ball_v2-4",
        "ball_v3",
        "ball_v6_yolov8s_960",
        "train-2",
        "train-3",
        "train-4",
    ],
    "player": [
        "player_detector_v1-5",
        "player_detector_v1-6",
    ],
    "indicator": [
        "indicator_v1",
    ],
}


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _find_results_csv(run_name: str):
    run_dir = PROJECT_ROOT / "runs" / "detect" / run_name
    candidates = [
        run_dir / "results.csv",
        run_dir / "weights" / "results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_yolo_metrics():
    rows = []
    for group, run_names in YOLO_MODEL_GROUPS.items():
        for run_name in run_names:
            weights_path = PROJECT_ROOT / "runs" / "detect" / run_name / "weights" / "best.pt"
            results_path = _find_results_csv(run_name)
            row = {
                "group": group,
                "run": run_name,
                "weights": str(weights_path),
                "has_weights": weights_path.exists(),
                "results_csv": str(results_path) if results_path else "",
                "precision": math.nan,
                "recall": math.nan,
                "map50": math.nan,
                "map50_95": math.nan,
            }

            if results_path:
                with results_path.open("r", newline="", encoding="utf-8") as f:
                    records = list(csv.DictReader(f))
                if records:
                    last = records[-1]
                    row.update(
                        {
                            "precision": _float_or_nan(last.get("metrics/precision(B)")),
                            "recall": _float_or_nan(last.get("metrics/recall(B)")),
                            "map50": _float_or_nan(last.get("metrics/mAP50(B)")),
                            "map50_95": _float_or_nan(last.get("metrics/mAP50-95(B)")),
                        }
                    )

            rows.append(row)
    return rows


def choose_best_yolo(rows):
    best = {}
    for row in rows:
        if not row["has_weights"] or math.isnan(row["map50_95"]):
            continue
        current = best.get(row["group"])
        if current is None or row["map50_95"] > current["map50_95"]:
            best[row["group"]] = row
    return best


def evaluate_random_forest_state_classifier():
    try:
        import joblib
        import numpy as np
        from PIL import Image
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        return {
            "model": "fifa_state_classifier.pkl",
            "status": f"skipped: missing dependency {exc.name}",
        }

    dataset_dir = PROJECT_ROOT / "dataset"
    model = joblib.load(PROJECT_ROOT / "fifa_state_classifier.pkl")
    labels = joblib.load(PROJECT_ROOT / "fifa_state_labels.pkl")

    image_paths = []
    targets = []
    for label_id, label in enumerate(labels):
        class_dir = dataset_dir / label
        for path in class_dir.glob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                targets.append(label_id)

    _, test_paths, _, test_targets = train_test_split(
        image_paths,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )

    features = []
    for path in test_paths:
        image = Image.open(path).convert("L").resize((160, 90))
        features.append(np.asarray(image).flatten())

    predictions = model.predict(features)
    return {
        "model": "fifa_state_classifier.pkl",
        "status": "ok",
        "accuracy": float(accuracy_score(test_targets, predictions)),
        "samples": len(test_targets),
    }


def print_yolo_report(rows):
    best = choose_best_yolo(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    for group, group_rows in grouped.items():
        print(f"\n[{group}]")
        for row in sorted(
            group_rows,
            key=lambda r: -1 if math.isnan(r["map50_95"]) else r["map50_95"],
            reverse=True,
        ):
            score = "unscored" if math.isnan(row["map50_95"]) else f"{row['map50_95']:.5f}"
            print(f"{row['run']}: mAP50-95={score} weights={row['has_weights']}")

        winner = best.get(group)
        if winner:
            print(f"selected: {winner['run']} ({winner['map50_95']:.5f})")
        else:
            print("selected: none - no scored weights found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        action="store_true",
        help="also evaluate the existing sklearn game-state classifier",
    )
    args = parser.parse_args()

    rows = read_yolo_metrics()
    print_yolo_report(rows)

    if args.state:
        print("\n[game_state]")
        print(evaluate_random_forest_state_classifier())


if __name__ == "__main__":
    main()
