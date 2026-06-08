"""Quick test: does YOLO11 pre-trained on COCO detect padel rackets as 'tennis racket' (class 38)?
Runs on a video, prints per-frame detection count + average confidence,
and writes an annotated video with bounding boxes for visual review."""
import argparse
import os
import sys

import cv2
from ultralytics import YOLO

YOLO_WEIGHTS = r"C:\claude\padelvision\padel-vision\yolo11x.pt"
RACKET_CLASS = 38   # COCO 'tennis racket'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    p.add_argument("--conf", type=float, default=0.10,
                   help="confidence threshold (low because padel rackets are out-of-distribution)")
    args = p.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"missing {args.video}")
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_mp4 = os.path.join(args.out_dir, f"{base}_racket.mp4")

    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    frames_with_racket = 0
    total_dets = 0
    confidences = []
    per_frame = []

    # Use model.predict() per frame (we want raw detections, not tracking).
    results = model.predict(source=args.video, classes=[RACKET_CLASS], conf=args.conf,
                            imgsz=1920, device=0, stream=True, verbose=False)

    for r in results:
        frame = r.orig_img.copy()
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        per_frame.append(n)
        if n > 0:
            frames_with_racket += 1
            total_dets += n
            for i in range(n):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                c = float(boxes.conf[i].cpu())
                confidences.append(c)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"racket {c:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"f{frame_idx}  n={n}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"frames                : {frame_idx}")
    print(f"frames with >=1 racket: {frames_with_racket} ({100 * frames_with_racket / max(frame_idx,1):.1f}%)")
    print(f"total detections      : {total_dets}")
    if confidences:
        print(f"avg confidence        : {sum(confidences)/len(confidences):.3f}")
        print(f"min/max confidence    : {min(confidences):.3f} / {max(confidences):.3f}")
    print(f"per-frame counts (first 30): {per_frame[:30]}")
    print(f"video -> {out_mp4}")


if __name__ == "__main__":
    main()
