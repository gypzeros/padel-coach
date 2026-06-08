"""Locate the smash phases in a pose CSV.

Heuristic (works for camera angles where height maps to y_px):
  1. Pick dominant wrist = the one that reaches the highest point (min y_px).
  2. Apex / impact = frame with the smallest smoothed y_px for that wrist.
     In a smash the racket meets the ball at the top of the arm extension;
     after that frame the wrist starts coming down (follow-through).
  3. Start = walking backwards from apex while the wrist is still rising.
  4. End   = walking forwards from apex while the wrist is descending,
              then a few frames of follow-through.

Outputs (next to the CSV):
  - <name>_smash.json   {dominant, start, apex, impact, end, fps, ...}
  - <name>_smash.png    plot of wrist y_px over time with markers
  - <name>_smash_marked.mp4   the original video overlaid with phase markers
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d


SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]


def conditional_median(cube, window=5, threshold_px=100):
    """Replace cube[i, lm, axis] with the median of its neighbours ONLY when it
    deviates from that median by more than `threshold_px`. Kills single-frame
    glitches while preserving genuine fast motion like the impact-frame wrist peak."""
    n, n_lm, n_axes = cube.shape
    half = window // 2
    out = cube.copy()
    for lm in range(n_lm):
        for axis in range(n_axes):
            series = cube[:, lm, axis]
            for i in range(n):
                lo, hi = max(0, i - half), min(n, i + half + 1)
                neighbours = np.concatenate([series[lo:i], series[i + 1:hi]])
                neighbours = neighbours[~np.isnan(neighbours)]
                if neighbours.size == 0 or np.isnan(series[i]):
                    continue
                med = float(np.median(neighbours))
                if abs(series[i] - med) > threshold_px:
                    out[i, lm, axis] = med
    return out


def load_smasher_pose(df, person_id):
    """Return {frame: ndarray(33, 2)} pixel coords for the smasher, with single-frame
    landmark glitches (>100px deviation) replaced by the local median."""
    if "person_id" in df.columns and person_id is not None:
        df = df[df.person_id == person_id]
    frames_sorted = sorted(df.frame.unique())
    if not frames_sorted:
        return {}
    n = len(frames_sorted)
    frame_to_row = {f: i for i, f in enumerate(frames_sorted)}
    cube = np.full((n, 33, 2), np.nan)
    for f, sub in df.groupby("frame"):
        sub = sub.sort_values("lm_idx")
        if len(sub) == 33:
            cube[frame_to_row[int(f)]] = sub[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    cube = conditional_median(cube)
    out = {}
    for i, f in enumerate(frames_sorted):
        if not np.isnan(cube[i]).any():
            out[int(f)] = cube[i]
    return out


def anchor_bbox_at_wrist(x1, y1, x2, y2, wrist_xy):
    """Translate an axis-aligned bbox so its NEAREST EDGE in the direction from the
    wrist to the bbox centre sits exactly at the wrist. Preserves bbox size."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
    vx, vy = cx - wrist_xy[0], cy - wrist_xy[1]
    vmag = (vx * vx + vy * vy) ** 0.5
    if vmag < 1e-6:
        return x1, y1, x2, y2, cx, cy
    ux, uy = vx / vmag, vy / vmag
    # Distance from new centre to the wrist, along (ux, uy), so the edge meets the wrist.
    t_candidates = []
    if abs(ux) > 1e-9:
        t_candidates.append(hw / abs(ux))
    if abs(uy) > 1e-9:
        t_candidates.append(hh / abs(uy))
    t = min(t_candidates) if t_candidates else 0
    new_cx = wrist_xy[0] + t * ux
    new_cy = wrist_xy[1] + t * uy
    dx, dy = new_cx - cx, new_cy - cy
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy, new_cx, new_cy


def draw_skeleton(canvas, pts, color, thickness=2, joint_radius=3):
    pts_int = pts.astype(np.int32)
    h, w = canvas.shape[:2]
    in_bounds = lambda p: 0 <= p[0] < w and 0 <= p[1] < h
    for a, b in SKELETON_EDGES:
        if a < len(pts_int) and b < len(pts_int):
            pa, pb = tuple(pts_int[a]), tuple(pts_int[b])
            if in_bounds(pa) and in_bounds(pb):
                cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
    for j in range(len(pts_int)):
        if in_bounds(tuple(pts_int[j])):
            cv2.circle(canvas, tuple(pts_int[j]), joint_radius, color, -1, cv2.LINE_AA)


def find_smasher(df):
    """Returns (person_id, wrist_name). Picks the (person, wrist) combo whose y_px
    has the LARGEST RANGE during the clip — the most dynamic vertical movement,
    which is the signature of a smash. Filters out poses with too few frames
    or low visibility."""
    candidates = {}
    if "person_id" not in df.columns:  # backwards compat with old single-person CSV
        df = df.assign(person_id=0)
    for pid in df.person_id.dropna().unique():
        person = df[df.person_id == pid]
        for wrist in ("left_wrist", "right_wrist"):
            sub = person[(person.lm_name == wrist) & (person.visibility >= 0.5)]
            if len(sub) < 5:
                continue
            y_range = float(sub.y_px.max() - sub.y_px.min())
            candidates[(int(pid), wrist)] = y_range
    if not candidates:
        sys.exit("no usable wrist landmarks")
    (pid, wrist), score = max(candidates.items(), key=lambda kv: kv[1])
    print(f"  smasher picked: person_id={pid}, wrist={wrist} "
          f"(y_px range during clip = {score:.0f}px)")
    print("  candidate scores:")
    for (p, w), s in sorted(candidates.items(), key=lambda kv: -kv[1]):
        print(f"    person {p} {w}: {s:.0f}")
    return pid, wrist


def find_phases(frames, y, fps,
                smooth_window=3,
                start_descent_thresh=1.0,
                end_descent_thresh=1.0,
                follow_through_frames=8,
                pre_pad_frames=8):
    y_smooth = uniform_filter1d(y, size=smooth_window, mode="nearest")
    vy = np.gradient(y_smooth)  # positive = wrist moving downward in image

    # Apex = highest wrist position = moment of ball impact for a smash.
    apex_i = int(np.argmin(y_smooth))
    impact_i = apex_i

    # Start: walk back from apex while wrist was still rising (vy < 0), then pad
    # earlier to capture the loading/preparation phase before the wrist visibly rises.
    start_i = apex_i
    while start_i > 0 and vy[start_i - 1] < -start_descent_thresh:
        start_i -= 1
    start_i = max(0, start_i - pre_pad_frames)

    # End: walk forward from apex while still descending, then add follow-through.
    end_i = apex_i
    while end_i < len(vy) - 1 and vy[end_i + 1] > end_descent_thresh:
        end_i += 1
    end_i = min(len(vy) - 1, end_i + follow_through_frames)

    return {
        "start": int(frames[start_i]),
        "apex": int(frames[apex_i]),
        "impact": int(frames[impact_i]),
        "end": int(frames[end_i]),
        "fps": float(fps),
        "duration_s": float((end_i - start_i) / fps),
        "y_at_apex_px": float(y_smooth[apex_i]),
        "vy_post_impact_max_pxpf": float(vy[apex_i:].max()) if apex_i < len(vy) else 0.0,
    }


def render_plot(frames, y, smooth_y, info, out_png, dominant):
    plt.figure(figsize=(10, 4))
    plt.plot(frames, y, color="lightgray", label=f"{dominant} y_px crudo")
    plt.plot(frames, smooth_y, color="steelblue", label="suavizado")
    plt.axvline(info["start"], color="gray", ls="--", lw=1, label="inicio")
    plt.axvline(info["apex"], color="green", ls="--", lw=1, label="apex")
    plt.axvline(info["impact"], color="red", ls="-", lw=2, label="impacto")
    plt.axvline(info["end"], color="gray", ls="--", lw=1, label="fin")
    plt.gca().invert_yaxis()  # so "up" on the plot = up in the video
    plt.xlabel("frame")
    plt.ylabel("muñeca y_px (invertido, arriba = alto)")
    plt.title(f"Fases del smash — dominante: {dominant}")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def load_rackets(csv_path):
    """Return {frame: list of dicts with x1,y1,x2,y2,cx,cy,conf}."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for f, sub in df.groupby("frame"):
        out[int(f)] = [
            {"x1": float(r.x1), "y1": float(r.y1),
             "x2": float(r.x2), "y2": float(r.y2),
             "cx": float(r.cx), "cy": float(r.cy),
             "conf": float(r.conf)}
            for _, r in sub.iterrows()
        ]
    return out


def pick_smasher_racket(rackets_per_frame, wrist_per_frame, max_dist_px=300):
    """For each frame, pick the racket whose centre is closest to the smasher's
    dominant wrist (within max_dist_px). Returns {frame: racket_dict}."""
    out = {}
    for f, (wx, wy) in wrist_per_frame.items():
        rs = rackets_per_frame.get(f, [])
        if not rs:
            continue
        best = min(rs, key=lambda r: (r["cx"] - wx) ** 2 + (r["cy"] - wy) ** 2)
        d = ((best["cx"] - wx) ** 2 + (best["cy"] - wy) ** 2) ** 0.5
        if d <= max_dist_px:
            out[f] = best
    return out


def render_marked_video(video_in, video_out, info, dominant_xy_per_frame,
                        smasher_poses=None, smasher_rackets=None):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        sys.exit(f"could not open {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    phase_color = {"pre": (180, 180, 180), "swing": (0, 200, 200),
                   "impact": (0, 0, 255), "follow": (0, 200, 0)}
    skeleton_color = (60, 220, 60)  # green like pose.py

    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i < info["start"]:
            phase = "pre"
        elif i < info["apex"]:
            phase = "swing"
        elif i == info["impact"]:
            phase = "impact"
        elif i <= info["end"]:
            phase = "follow"
        else:
            phase = "pre"
        col = phase_color[phase]

        # Smasher skeleton (median-filtered so single-frame glitches don't show).
        if smasher_poses is not None and i in smasher_poses:
            draw_skeleton(bgr, smasher_poses[i], skeleton_color)

        # Smasher's racket: bbox + centre dot, repositioned so its nearest edge
        # touches the wrist (so the bbox visibly hangs off the hand).
        if (smasher_rackets is not None and i in smasher_rackets
                and i in dominant_xy_per_frame):
            r = smasher_rackets[i]
            x1, y1, x2, y2, cx, cy = anchor_bbox_at_wrist(
                r["x1"], r["y1"], r["x2"], r["y2"], dominant_xy_per_frame[i])
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 200, 255), 2)
            cv2.circle(bgr, (int(cx), int(cy)), 6, (0, 200, 255), -1)

        # Phase header.
        cv2.rectangle(bgr, (0, 0), (W, 40), col, -1)
        cv2.putText(bgr, f"f{i}  {phase.upper()}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        # Highlight dominant wrist with a phase-coloured ring on top of the skeleton.
        if i in dominant_xy_per_frame:
            x, y = dominant_xy_per_frame[i]
            cv2.circle(bgr, (int(x), int(y)), 14, col, 3)

        writer.write(bgr)
        i += 1
    cap.release()
    writer.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="pose CSV produced by pose.py")
    p.add_argument("--video", help="(optional) original video to render the marked output from")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    p.add_argument("--pre-pad", type=int, default=8,
                   help="extra frames added before detected swing start (capture loading phase)")
    args = p.parse_args()

    base = os.path.splitext(os.path.basename(args.csv))[0].replace("_pose", "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"{base}_smash.json")
    out_png = os.path.join(args.out_dir, f"{base}_smash.png")
    out_video = os.path.join(args.out_dir, f"{base}_smash_marked.mp4")

    df = pd.read_csv(args.csv)
    fps_series = (df.frame / df.t).replace([np.inf, -np.inf], np.nan).dropna()
    fps = float(fps_series.median()) if len(fps_series) else 30.0

    person_id, dominant = find_smasher(df)
    if "person_id" in df.columns:
        df = df[df.person_id == person_id]
    print(f"dominant wrist: {dominant} (person_id={person_id})")

    sub = df[df.lm_name == dominant].sort_values("frame").reset_index(drop=True)
    frames = sub.frame.to_numpy()
    y = sub.y_px.to_numpy()
    smooth_y = uniform_filter1d(y, size=3, mode="nearest")

    info = find_phases(frames, y, fps, pre_pad_frames=args.pre_pad)
    info["dominant"] = dominant
    info["person_id"] = int(person_id)
    info["video_csv"] = os.path.basename(args.csv)
    print(json.dumps(info, indent=2))

    with open(out_json, "w") as f:
        json.dump(info, f, indent=2)
    render_plot(frames, y, smooth_y, info, out_png, dominant)
    print(f"plot -> {out_png}")
    print(f"json -> {out_json}")

    if args.video:
        xy = dict(zip(sub.frame.astype(int), zip(sub.x_px, sub.y_px)))
        full_df = pd.read_csv(args.csv)
        smasher_poses = load_smasher_pose(full_df, person_id)
        # Optional racket integration: if a racket CSV exists next to the pose CSV, use it.
        racket_csv = args.csv.replace("_pose.csv", "_racket.csv")
        rackets = load_rackets(racket_csv)
        smasher_rackets = pick_smasher_racket(rackets, xy) if rackets else None
        if smasher_rackets is not None:
            print(f"racket paired on {len(smasher_rackets)}/{len(xy)} frames")
        render_marked_video(args.video, out_video, info, xy, smasher_poses, smasher_rackets)
        print(f"marked video -> {out_video}")


if __name__ == "__main__":
    main()
