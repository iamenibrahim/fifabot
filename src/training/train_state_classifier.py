import cv2
import os
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

labels = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print("Classes found:", labels)

X = []
y = []

for label_id, label in enumerate(labels):
    folder = os.path.join(DATASET_DIR, label)
    files = os.listdir(folder)

    print(label, ":", len(files), "files")

    for filename in files:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (160, 90))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X.append(gray.flatten())
        y.append(label_id)

X = np.array(X)
y = np.array(y)

print("Total usable images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(labels))),
    target_names=labels,
    zero_division=0
))

joblib.dump(model, PROJECT_ROOT / "fifa_state_classifier.pkl")
joblib.dump(labels, PROJECT_ROOT / "fifa_state_labels.pkl")

print("Saved fifa_state_classifier.pkl")
print("Saved fifa_state_labels.pkl")
