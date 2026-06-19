from ultralytics import YOLO
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "fifa_resnet_state_classifier.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
class_names = full_dataset.classes

print("Classes:", class_names)
print("Total images:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weights
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = Counter(train_labels)

weights = []
for i in range(len(class_names)):
    weights.append(1.0 / class_counts[i])

class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# Pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    zero_division=0
))

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, MODEL_PATH)

print("Saved:", MODEL_PATH)
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "player_detector_v1-6" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-27 18-28-58.mp4"

model = YOLO(str(MODEL_PATH))

model.track(
    source=str(VIDEO_PATH),
    tracker=str(PROJECT_ROOT / "configs" / "fifa_bytetrack.yaml"),
    conf=0.10,
    imgsz=320,
    save=True,
    show=True,
    persist=True
)
model2 = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

model2.train(
    data=str(PROJECT_ROOT / "configs" / "ball.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,
    name="ball_v2"
)
