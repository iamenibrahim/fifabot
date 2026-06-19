# Google Colab Training

Use Colab when local CPU training is too slow. The code is already on GitHub,
but the large datasets and model weights are intentionally ignored by Git, so
you need to upload or mount them separately.

## 1. Start Colab GPU Runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
```

Then check the GPU:

```bash
!nvidia-smi
```

## 2. Clone The Repo

```bash
!git clone https://github.com/iamenibrahim/fifabot.git
%cd fifabot
```

Install dependencies:

```bash
!pip install ultralytics opencv-python numpy
```

## 3. Add The Dataset And Base Weights

The combined dataset is ignored by Git, so Colab will not get it from GitHub.
Recommended options:

### Option A: Upload A Zip

On your PC, zip this folder:

```text
I:\FIFA files\fifabot\fifa_objects_dataset
```

Also upload:

```text
yolov8s.pt
```

In Colab:

```python
from google.colab import files
files.upload()
```

Then unzip:

```bash
!unzip fifa_objects_dataset.zip -d .
```

If the zip contains the folder itself, you should end with:

```text
fifabot/fifa_objects_dataset/images/train
fifabot/fifa_objects_dataset/images/val
```

### Option B: Google Drive

Upload `fifa_objects_dataset/` and `yolov8s.pt` to Google Drive, then mount:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Copy into the repo:

```bash
!cp -r "/content/drive/MyDrive/fifa_objects_dataset" .
!cp "/content/drive/MyDrive/yolov8s.pt" .
```

## 4. Fix The Dataset YAML In Colab

The local Windows YAML path will not work in Colab. Create a Colab-specific
config:

```bash
cat > configs/fifa_objects_colab.yaml <<'EOF'
path: /content/fifabot/fifa_objects_dataset
train: images/train
val: images/val

names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
  4: controlled_player_indicator
EOF
```

## 5. Train

Run training directly with Ultralytics:

```bash
!yolo detect train model=yolov8s.pt data=configs/fifa_objects_colab.yaml epochs=100 imgsz=640 batch=16 name=fifa_objects_v1 device=0
```

If Colab runs out of GPU memory, lower the batch:

```bash
!yolo detect train model=yolov8s.pt data=configs/fifa_objects_colab.yaml epochs=100 imgsz=640 batch=8 name=fifa_objects_v1 device=0
```

For a quick smoke test:

```bash
!yolo detect train model=yolov8s.pt data=configs/fifa_objects_colab.yaml epochs=1 imgsz=640 batch=8 name=fifa_objects_smoke device=0
```

## 6. Download The Trained Model

After training:

```python
from google.colab import files
files.download('/content/fifabot/runs/detect/fifa_objects_v1/weights/best.pt')
```

Copy `best.pt` back to your PC, for example:

```text
I:\FIFA files\fifabot\runs\detect\fifa_objects_v1\weights\best.pt
```

Then test locally:

```powershell
py src/inference/test_fifa_objects_video.py
```

## Notes

- Do not train the game-state classifier here for this task.
- Keep datasets and weights out of Git.
- Colab storage resets when the runtime disconnects, so download `best.pt`
  before ending the session.
