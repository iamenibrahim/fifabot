import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

cap = cv2.VideoCapture(str(PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-21 17-03-01.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cv2.imshow("frame 192", frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
