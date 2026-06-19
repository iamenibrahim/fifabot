from pathlib import Path

import cv2
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "fifa_state_classifier.pkl"
LABELS_PATH = PROJECT_ROOT / "fifa_state_labels.pkl"


class GameStateClassifier:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.model = joblib.load(self.model_path)
        self.labels = joblib.load(self.labels_path)

    def predict(self, frame) -> dict:
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        features = gray.flatten()

        prediction = int(self.model.predict([features])[0])
        state = self.labels[prediction]
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([features])[0]
            confidence = float(probabilities[prediction])

        return {
            "state": state,
            "class_id": prediction,
            "confidence": confidence,
            "model": str(self.model_path.name),
        }
