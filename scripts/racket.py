"""Detect padel rackets in a video using YOLO11 pre-trained on COCO (class 38 = tennis racket).

Outputs:
  - <name>_racket.csv   one row per detection: frame, det_idx, x1, y1, x2, y2,
                        cx, cy, conf
"""
import argparse
import csv
import os
import sys

import cv2
from ultralytics import YOLO

YOLO_WEIGHTS = r"C:\claude\padelvision\padel-vision\yolo11x.pt"
RACKET_CLASS = 38  # COCO 'tennis racket'


def process(video_in, csv_out, conf_thresh=0.10, imgsz=1920):
    if not os.path.exists(video_in):
        sys.exit(f"missing {video_in}")
    model = YOLO(YOLO_WEIGHTS)

    csv_fh = open(csv_out, "w", newline="")
    cw = csv.writer(csv_fh)
    cw.writerow(["frame", "det_idx", "x1", "y1", "x2", "y2", "cx", "cy", "conf"])

    results = model.predict(source=video_in, classes=[RACKET_CLASS], conf=conf_thresh,
                            imgsz=imgsz, device=0, stream=True, verbose=False)
    n_frames = 0
    n_dets = 0
    frames_with_det = 0
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        if n:
            frames_with_det += 1
        for k in range(n):
            x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy().tolist()
            c = float(boxes.conf[k].cpu())
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cw.writerow([frame_idx, k,
                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                         f"{cx:.1f}", f"{cy:.1f}", f"{c:.3f}"])
            n_dets += 1
        n_frames = frame_idx + 1
    csv_fh.close()
    print(f"frames: {n_frames}, frames with racket: {frames_with_det} "
          f"({100*frames_with_det/max(n_frames,1):.1f}%), total detections: {n_dets}")
    print(f"csv -> {csv_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    p.add_argument("--conf", type=float, default=0.10)
    args = p.parse_args()

    base = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    process(args.video, os.path.join(args.out_dir, f"{base}_racket.csv"), args.conf)
