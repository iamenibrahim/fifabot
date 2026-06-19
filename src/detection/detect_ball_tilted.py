def detect_ball_tiled(frame, model, conf=0.25):
    h, w = frame.shape[:2]
    
    # Split screen into 4 overlapping tiles
    tiles = [
        frame[0:h//2+50,    0:w//2+50],     # top left
        frame[0:h//2+50,    w//2-50:w],     # top right
        frame[h//2-50:h,    0:w//2+50],     # bottom left
        frame[h//2-50:h,    w//2-50:w],     # bottom right
    ]
    
    offsets = [
        (0, 0),
        (w//2-50, 0),
        (0, h//2-50),
        (w//2-50, h//2-50),
    ]
    
    best_box = None
    best_conf = 0

    for tile, (ox, oy) in zip(tiles, offsets):
        results = model(tile, conf=conf, verbose=False)
        for box in results[0].boxes:
            bc = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Translate back to full frame coordinates
            x1 += ox; x2 += ox
            y1 += oy; y2 += oy
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if bc > best_conf:
                best_conf = bc
                best_box = (x1, y1, x2, y2, cx, cy)

    return best_box, best_conf