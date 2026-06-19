import cv2
import numpy as np
import mss

sct = mss.mss()
monitor = sct.monitors[1]

def detect_ball_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White ball mask
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask  = cv2.inRange(hsv, white_lower, white_upper)

    # Brown/orange ball mask
    brown_lower = np.array([10, 50, 50])
    brown_upper = np.array([25, 255, 200])
    brown_mask  = cv2.inRange(hsv, brown_lower, brown_upper)

    # Combine masks
    mask = cv2.bitwise_or(white_mask, brown_mask)

    # Remove noise
    mask = cv2.erode(mask,  np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

    # Find circles
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=25,
        minRadius=3,
        maxRadius=25
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        best = None
        best_score = 0

        for (cx, cy, r) in circles:
            # Check if circle overlaps with our color mask
            roi = mask[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
            if roi.size == 0:
                continue
            score = np.sum(roi) / (roi.size * 255)

            if score > best_score:
                best_score = score
                best = (cx, cy, r)

        if best is not None and best_score > 0.2:
            return best

    return None

while True:
    screenshot = np.array(sct.grab(monitor))
    frame      = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    result = detect_ball_hsv(frame)

    if result is not None:
        cx, cy, r = result
        cv2.circle(frame, (cx, cy), r,     (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 2,     (0, 255, 0), -1)
        cv2.putText(frame, "ball", (cx+r+5, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("HSV Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()