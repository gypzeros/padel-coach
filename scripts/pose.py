"""Extract per-frame body pose from a video using MediaPipe's PoseLandmarker (tasks API).

Detects up to --num-poses people per frame and assigns stable person_ids via
hip-centroid tracking across frames.

Outputs:
  - <name>_pose.mp4    annotated video with skeleton overlay (color per person_id)
  - <name>_pose.csv    long-format CSV: one row per (frame, person_id, landmark)
                       cols: frame, t, person_id, lm_idx, lm_name,
                             x_px, y_px, x_world, y_world, z_world,
                             visibility, presence
"""
import argparse
import csv
import os
import sys
from math import hypot

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LM_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]

ID_COLORS = [
    (60, 220, 60), (220, 80, 80), (80, 80, 220), (220, 220, 60),
    (220, 60, 220), (60, 220, 220), (200, 130, 50), (130, 50, 200),
]


def draw_skeleton(img, lms_px, color, edges=POSE_EDGES, conf_thresh=0.4):
    h, w = img.shape[:2]
    pts = []
    for lm in lms_px:
        x = int(lm.x * w)
        y = int(lm.y * h)
        ok = getattr(lm, "visibility", 1.0) >= conf_thresh
        pts.append((x, y, ok))
    for a, b in edges:
        if a < len(pts) and b < len(pts) and pts[a][2] and pts[b][2]:
            cv2.line(img, pts[a][:2], pts[b][:2], color, 2)
    for x, y, ok in pts:
        if ok:
            cv2.circle(img, (x, y), 3, color, -1)


def dedupe_poses(poses_px, poses_w, W, H, min_dist_px=80):
    """MediaPipe occasionally fires two detections on the same person (overlapping
    skeletons). Drop the lower-confidence duplicate whenever two hip-centroids
    fall within `min_dist_px` of each other."""
    if len(poses_px) <= 1:
        return poses_px, poses_w
    centroids, confs = [], []
    for pose in poses_px:
        lh, rh = pose[23], pose[24]
        centroids.append(((lh.x + rh.x) / 2 * W, (lh.y + rh.y) / 2 * H))
        confs.append(sum(getattr(lm, "visibility", 1.0) for lm in pose) / len(pose))
    keep = [True] * len(poses_px)
    for i in range(len(poses_px)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(poses_px)):
            if not keep[j]:
                continue
            d = hypot(centroids[i][0] - centroids[j][0],
                      centroids[i][1] - centroids[j][1])
            if d < min_dist_px:
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    new_px = [p for p, k in zip(poses_px, keep) if k]
    new_w = [p for p, k in zip(poses_w, keep) if k] if poses_w else []
    return new_px, new_w


class PersonTracker:
    """Assigns stable person_ids by matching each frame's hip-centroids to recent ones.
    Keeps IDs in memory for `memory_frames` frames after the last sighting, so a person
    briefly missed by MediaPipe (e.g., motion blur) keeps the same id."""
    def __init__(self, max_dist_px=350, memory_frames=8):
        self.max_dist = max_dist_px
        self.memory_frames = memory_frames
        self.prev = {}      # pid -> (cx, cy, age_frames_since_seen)
        self.next_id = 0

    def assign(self, centroids):
        """centroids: list of (cx, cy). Returns parallel list of person_ids."""
        n = len(centroids)
        out = [None] * n
        # Greedy matching, prefers spatially close AND recently seen IDs.
        candidates = []
        for i, c in enumerate(centroids):
            for pid, (pcx, pcy, age) in self.prev.items():
                d = hypot(c[0] - pcx, c[1] - pcy)
                if d <= self.max_dist:
                    # Stale tracks pay a small penalty so a fresh closer match wins.
                    candidates.append((d + age * 8, i, pid))
        candidates.sort()
        used_i, used_pid = set(), set()
        for _, i, pid in candidates:
            if i in used_i or pid in used_pid:
                continue
            out[i] = pid
            used_i.add(i)
            used_pid.add(pid)
        for i in range(n):
            if out[i] is None:
                out[i] = self.next_id
                self.next_id += 1
        # Refresh prev: matched IDs at age 0, unmatched but still-young carry over.
        new_prev = {pid: (centroids[i][0], centroids[i][1], 0)
                    for i, pid in enumerate(out)}
        for pid, (cx, cy, age) in self.prev.items():
            if pid not in new_prev and age + 1 < self.memory_frames:
                new_prev[pid] = (cx, cy, age + 1)
        self.prev = new_prev
        return out


def process(video_in, video_out, csv_out, model_path, num_poses):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        sys.exit(f"could not open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    csv_fh = open(csv_out, "w", newline="")
    cw = csv.writer(csv_fh)
    cw.writerow([
        "frame", "t", "person_id",
        "lm_idx", "lm_name",
        "x_px", "y_px",
        "x_world", "y_world", "z_world",
        "visibility", "presence",
    ])

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    tracker = PersonTracker()
    frame_idx = 0
    detected_frames = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, ts_ms)

            annotated = bgr.copy()
            poses_px = result.pose_landmarks or []
            poses_w = result.pose_world_landmarks or []
            poses_px, poses_w = dedupe_poses(poses_px, poses_w, W, H)
            if poses_px:
                detected_frames += 1
                # Compute hip-centroids in pixel coords for tracking.
                centroids = []
                for pose in poses_px:
                    lh, rh = pose[23], pose[24]
                    cx = (lh.x + rh.x) / 2 * W
                    cy = (lh.y + rh.y) / 2 * H
                    centroids.append((cx, cy))
                person_ids = tracker.assign(centroids)

                t = frame_idx / fps
                for pid, pose_px, pose_w in zip(person_ids, poses_px,
                                                poses_w + [None] * (len(poses_px) - len(poses_w))):
                    color = ID_COLORS[pid % len(ID_COLORS)]
                    draw_skeleton(annotated, pose_px, color)
                    # label near pose centroid
                    lh, rh = pose_px[23], pose_px[24]
                    cx = int((lh.x + rh.x) / 2 * W)
                    cy = int((lh.y + rh.y) / 2 * H)
                    cv2.putText(annotated, f"id{pid}", (cx - 18, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    for i in range(33):
                        p = pose_px[i]
                        w = pose_w[i] if pose_w else None
                        cw.writerow([
                            frame_idx, f"{t:.4f}", pid,
                            i, LM_NAMES[i],
                            f"{p.x * W:.2f}", f"{p.y * H:.2f}",
                            f"{w.x:.4f}" if w else "", f"{w.y:.4f}" if w else "",
                            f"{w.z:.4f}" if w else "",
                            f"{getattr(p, 'visibility', 1.0):.3f}",
                            f"{getattr(p, 'presence', 1.0):.3f}",
                        ])

            color = (0, 200, 0) if poses_px else (0, 0, 220)
            cv2.putText(annotated, f"f{frame_idx} n={len(poses_px)}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            writer.write(annotated)
            frame_idx += 1

    cap.release()
    writer.release()
    csv_fh.close()
    print(f"frames: {frame_idx}, frames with pose: {detected_frames} "
          f"({100*detected_frames/max(frame_idx,1):.1f}%)")
    print(f"video -> {video_out}")
    print(f"csv   -> {csv_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="input video path")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    p.add_argument("--model",
                   default=r"C:\claude\padel-coach\models\pose_landmarker_heavy.task")
    p.add_argument("--num-poses", type=int, default=5)
    args = p.parse_args()

    base = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    out_video = os.path.join(args.out_dir, f"{base}_pose.mp4")
    out_csv = os.path.join(args.out_dir, f"{base}_pose.csv")

    process(args.video, out_video, out_csv, args.model, args.num_poses)
