"""Time-align two smashes via DTW and surface the mechanical differences.

Inputs (typically):
  - pro_metrics_csv, pro_smash_json, pro_video
  - user_metrics_csv, user_smash_json, user_video

Pipeline:
  1. Load each clip's metric time-series, restricted to its swing window
     (frames start..end from its smash JSON).
  2. Run DTW on the dominant_wrist_height series — the most distinctive
     signal of a smash — to align user time to pro time.
  3. Apply the warp path to every metric; compute deltas.
  4. Render:
       - comparison.png    paired plots (pro vs user warped onto a common axis)
       - comparison.json   summary of biggest deltas at impact + over swing
       - comparison_sxs.mp4   side-by-side video, frames paired by DTW
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.ndimage import median_filter, uniform_filter1d


METRIC_COLS = [
    "dominant_elbow_angle_deg",
    "dominant_wrist_speed_ms",
    "dominant_wrist_height_m",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "trunk_lean_deg",
    "hip_shoulder_separation_m",
    "feet_separation_m",
]

# Skeleton edges — same set drawn by pose.py so overlay matches the annotated video.
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),       # arms + shoulders
    (11, 23), (12, 24), (23, 24),                           # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),       # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),       # right leg
    (15, 17), (15, 19), (15, 21), (17, 19),                 # left hand
    (16, 18), (16, 20), (16, 22), (18, 20),                 # right hand
]
SKELETON_JOINTS = list(range(33))  # all landmarks


def load_window(metrics_csv, smash_json):
    with open(smash_json) as f:
        sm = json.load(f)
    df = pd.read_csv(metrics_csv)
    win = df[(df.frame >= sm["start"]) & (df.frame <= sm["end"])].reset_index(drop=True)
    return win, sm


def trim_to_common(pro_win, pro_sm, user_win, user_sm):
    """Crop both swings so they share the same number of pre-impact and post-impact
    frames (the min of each side). Returns (pro_win, pro_sm, user_win, user_sm) with
    updated start/end (impact stays at the same absolute frame)."""
    pro_pre = pro_sm["impact"] - pro_sm["start"]
    pro_post = pro_sm["end"] - pro_sm["impact"]
    user_pre = user_sm["impact"] - user_sm["start"]
    user_post = user_sm["end"] - user_sm["impact"]
    pre = min(pro_pre, user_pre)
    post = min(pro_post, user_post)

    def crop(win, sm):
        sm2 = dict(sm)
        sm2["start"] = sm["impact"] - pre
        sm2["end"] = sm["impact"] + post
        new_win = win[(win.frame >= sm2["start"]) & (win.frame <= sm2["end"])].reset_index(drop=True)
        return new_win, sm2

    pro_win2, pro_sm2 = crop(pro_win, pro_sm)
    user_win2, user_sm2 = crop(user_win, user_sm)
    print(f"trimmed to common pre={pre}, post={post} "
          f"(pro had pre={pro_pre} post={pro_post}, user had pre={user_pre} post={user_post})")
    return pro_win2, pro_sm2, user_win2, user_sm2


def dtw_align(a, b):
    """Plain DTW on 1D arrays. Returns (path, distance)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    path = dtw.warping_path(a, b)
    d = dtw.distance(a, b)
    return path, float(d)


def dtw_align_anchored(a, b, impact_a, impact_b):
    """DTW that is GUARANTEED to map index `impact_a` of a to index `impact_b` of b.
    Achieved by running DTW separately on pre-impact and post-impact slices and joining them.
    Indices in the returned path refer to the original (un-sliced) arrays a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def safe_path(s1, s2):
        if len(s1) < 2 or len(s2) < 2:
            # Trivial alignment when one side is too short.
            return [(i, min(i, len(s2) - 1)) for i in range(len(s1))]
        return dtw.warping_path(s1, s2)

    pre_a = a[:impact_a + 1]
    pre_b = b[:impact_b + 1]
    pre_path = safe_path(pre_a, pre_b)

    post_a = a[impact_a:]
    post_b = b[impact_b:]
    post_path = safe_path(post_a, post_b)
    post_path_offset = [(impact_a + i, impact_b + j) for i, j in post_path]

    # The two halves both include the impact step; drop the duplicate at the join.
    joined = list(pre_path)
    if joined and post_path_offset and joined[-1] == post_path_offset[0]:
        joined.extend(post_path_offset[1:])
    else:
        joined.extend(post_path_offset)

    # Distance: sum of the two halves' DTW distances (only meaningful as a relative number).
    d_pre = dtw.distance(pre_a, pre_b) if len(pre_a) >= 2 and len(pre_b) >= 2 else 0.0
    d_post = dtw.distance(post_a, post_b) if len(post_a) >= 2 and len(post_b) >= 2 else 0.0
    return joined, float(d_pre + d_post)


METRIC_LABELS_ES = {
    "dominant_elbow_angle_deg":  "Codo (°)",
    "dominant_wrist_speed_ms":   "Vel. muñeca (m/s)",
    "dominant_wrist_height_m":   "Altura muñeca (m)",
    "left_knee_angle_deg":       "Rodilla izq. (°)",
    "right_knee_angle_deg":      "Rodilla der. (°)",
    "trunk_lean_deg":            "Inclinación tronco (°)",
    "hip_shoulder_separation_m": "Cadera-hombro Δy (m)",
    "feet_separation_m":         "Separación pies (m)",
}


def render_paired_plot(pro_win, user_win, path, pro_sm, user_sm, out_png):
    fig, axes = plt.subplots(len(METRIC_COLS), 1, figsize=(11, 2 * len(METRIC_COLS)),
                             sharex=True)
    step = np.arange(len(path))
    pro_idx = np.array([p[0] for p in path])
    user_idx = np.array([p[1] for p in path])
    pro_impact_local = pro_sm["impact"] - pro_sm["start"]
    impact_steps = np.where(pro_idx == pro_impact_local)[0]
    impact_step = int(impact_steps[0]) if len(impact_steps) else None

    for ax, col in zip(axes, METRIC_COLS):
        pro_vals = pro_win[col].to_numpy()[pro_idx]
        user_vals = user_win[col].to_numpy()[user_idx]
        ax.plot(step, uniform_filter1d(np.nan_to_num(pro_vals, nan=0.0), 3),
                label="PRO", color="steelblue", lw=2)
        ax.plot(step, uniform_filter1d(np.nan_to_num(user_vals, nan=0.0), 3),
                label="USER", color="crimson", lw=2)
        ax.set_ylabel(METRIC_LABELS_ES.get(col, col), fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        if impact_step is not None:
            ax.axvline(impact_step, color="red", lw=1, ls="--", alpha=0.6)

    axes[0].set_title("Curvas alineadas DTW — PRO vs USER", fontsize=11)
    axes[-1].set_xlabel("Step alineado DTW (línea roja vertical = impacto)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


# (label, csv column, units, "good" tolerance — within ±tol → green, ±2·tol → yellow, else red)
OVERLAY_METRICS = [
    ("Codo",      "dominant_elbow_angle_deg",  " deg", 10.0),
    ("Muneca v",  "dominant_wrist_speed_ms",   " m/s", 1.0),
    ("Muneca h",  "dominant_wrist_height_m",   " m",   0.10),
    ("Rodilla L", "left_knee_angle_deg",       " deg", 10.0),
    ("Rodilla R", "right_knee_angle_deg",      " deg", 10.0),
    ("Tronco",    "trunk_lean_deg",            " deg", 5.0),
    ("Pies",      "feet_separation_m",         " m",   0.08),
]


def _delta_color(delta, tol):
    mag = abs(delta) / max(tol, 1e-9)
    if mag < 1.0:
        return (60, 220, 60)        # green
    if mag < 2.0:
        return (40, 220, 220)       # yellow
    return (60, 60, 255)            # red


def draw_metric_panel(canvas, x0, y0, row, side, paired_row=None):
    """Stack metric rows starting at (x0, y0). When side == 'user', show (Δ vs pro)
    with colour-coded magnitude."""
    for k, (label, col, unit, tol) in enumerate(OVERLAY_METRICS):
        y = y0 + k * 26
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if side == "pro":
            text = f"{label}: {v:.1f}{unit}"
            color = (235, 235, 235)
        else:
            pv = paired_row.get(col) if paired_row is not None else None
            if pv is None or (isinstance(pv, float) and np.isnan(pv)):
                text = f"{label}: {v:.1f}{unit}"
                color = (235, 235, 235)
            else:
                d = float(v) - float(pv)
                sign = "+" if d >= 0 else ""
                text = f"{label}: {v:.1f}{unit}  ({sign}{d:.1f})"
                color = _delta_color(d, tol)
        # Black outline for legibility on bright frames.
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)


def render_sxs_video(pro_video, user_video, pro_sm, user_sm, path, out_mp4,
                     pro_win=None, user_win=None):
    pro_cap = cv2.VideoCapture(pro_video)
    user_cap = cv2.VideoCapture(user_video)
    if not pro_cap.isOpened() or not user_cap.isOpened():
        raise RuntimeError("could not open one of the videos")

    fps = pro_cap.get(cv2.CAP_PROP_FPS) or 25.0

    def grab_window(cap, start, end):
        frames = []
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if start <= i <= end:
                frames.append(f)
            i += 1
            if i > end:
                break
        return frames

    pro_frames = grab_window(pro_cap, pro_sm["start"], pro_sm["end"])
    user_frames = grab_window(user_cap, user_sm["start"], user_sm["end"])
    pro_cap.release(); user_cap.release()

    if not pro_frames or not user_frames:
        raise RuntimeError("empty window for one of the videos")

    h_pro, w_pro = pro_frames[0].shape[:2]
    h_user, w_user = user_frames[0].shape[:2]
    target_h = min(h_pro, h_user, 720)
    def fit(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))
    pro_frames = [fit(f) for f in pro_frames]
    user_frames = [fit(f) for f in user_frames]
    w_pro = pro_frames[0].shape[1]
    w_user = user_frames[0].shape[1]
    out_w = w_pro + w_user + 10
    out_h = target_h + 40

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_w, out_h))
    pro_impact_local = pro_sm["impact"] - pro_sm["start"]
    user_impact_local = user_sm["impact"] - user_sm["start"]

    for step, (i, j) in enumerate(path):
        i = min(i, len(pro_frames) - 1)
        j = min(j, len(user_frames) - 1)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[40:40 + target_h, 0:w_pro] = pro_frames[i]
        canvas[40:40 + target_h, w_pro + 10:w_pro + 10 + w_user] = user_frames[j]

        cv2.rectangle(canvas, (0, 0), (out_w, 40), (40, 40, 40), -1)
        is_impact = (i == pro_impact_local) and (j == user_impact_local)
        col = (0, 0, 255) if is_impact else (220, 220, 220)
        cv2.putText(canvas, f"PRO  f{i + pro_sm['start']}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(canvas, f"USER f{j + user_sm['start']}", (w_pro + 20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if is_impact:
            cv2.putText(canvas, "IMPACT", (out_w // 2 - 50, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Per-frame metric panels.
        if pro_win is not None and i < len(pro_win):
            draw_metric_panel(canvas, 10, 70, pro_win.iloc[i], side="pro")
        if user_win is not None and j < len(user_win):
            paired = pro_win.iloc[i] if pro_win is not None and i < len(pro_win) else None
            draw_metric_panel(canvas, w_pro + 20, 70, user_win.iloc[j],
                              side="user", paired_row=paired)

        writer.write(canvas)
    writer.release()


def racket_angle_deg(wrist_xy, racket_center_xy):
    """2D handle-direction angle in degrees.
    0° = pointing straight up from the wrist (above), 90° = right, -90° = left,
    ±180° = pointing down. Image y is flipped to make "up" the zero reference."""
    dx = racket_center_xy[0] - wrist_xy[0]
    dy = racket_center_xy[1] - wrist_xy[1]
    return float(np.degrees(np.arctan2(dx, -dy)))


def anchor_bbox_at_wrist(x1, y1, x2, y2, wrist_xy):
    """Translate an axis-aligned bbox so its nearest edge (in the wrist→centre direction)
    sits exactly at the wrist. Preserves bbox size."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
    vx, vy = cx - wrist_xy[0], cy - wrist_xy[1]
    vmag = (vx * vx + vy * vy) ** 0.5
    if vmag < 1e-6:
        return x1, y1, x2, y2, cx, cy
    ux, uy = vx / vmag, vy / vmag
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


def draw_racket(canvas, wrist_xy, racket_center_xy, color,
                head_long=46, head_short=38, handle_thickness=4):
    """Stylised padel racket: thick handle from wrist to bbox-centre,
    plus an oval head oriented along the handle direction."""
    wx, wy = wrist_xy
    cx, cy = racket_center_xy
    dx, dy = cx - wx, cy - wy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    # Handle
    cv2.line(canvas, (int(wx), int(wy)), (int(cx), int(cy)), color,
             handle_thickness, cv2.LINE_AA)
    # Outer frame
    cv2.ellipse(canvas, (int(cx), int(cy)), (head_long, head_short),
                angle_deg, 0, 360, color, 2, cv2.LINE_AA)
    # Inner contour to suggest strings/face
    cv2.ellipse(canvas, (int(cx), int(cy)), (head_long - 10, head_short - 10),
                angle_deg, 0, 360, color, 1, cv2.LINE_AA)


def _conditional_median(cube, window=5, threshold_px=100):
    """Per-landmark, per-axis: replace value with local median ONLY when it deviates
    from that median by more than `threshold_px`. Preserves real fast motion (impact
    wrist peak) while removing single-frame outliers (limb teleports)."""
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


def load_smasher_rackets(racket_csv, smasher_wrist_per_frame, max_dist_px=300):
    """For each frame, return {frame: dict with x1,y1,x2,y2,cx,cy} of the racket
    nearest the smasher's wrist."""
    if not racket_csv or not os.path.exists(racket_csv):
        return {}
    df = pd.read_csv(racket_csv)
    out = {}
    for f, sub in df.groupby("frame"):
        f = int(f)
        if f not in smasher_wrist_per_frame:
            continue
        wx, wy = smasher_wrist_per_frame[f]
        best = None
        best_d = float("inf")
        for _, r in sub.iterrows():
            d = (float(r.cx) - wx) ** 2 + (float(r.cy) - wy) ** 2
            if d < best_d:
                best_d = d
                best = {"x1": float(r.x1), "y1": float(r.y1),
                        "x2": float(r.x2), "y2": float(r.y2),
                        "cx": float(r.cx), "cy": float(r.cy)}
        if best is not None and best_d ** 0.5 <= max_dist_px:
            out[f] = best
    return out


def smasher_wrist_per_frame(pose_csv, person_id, dominant, smash_sm):
    """Return {frame: (x_px, y_px)} for the smasher's dominant wrist,
    using the SAME conditional-median-filtered pose that the skeleton draws,
    so racket pairing aligns with the rendered wrist."""
    poses = load_pose_pixels(pose_csv, person_id, smash_sm)
    wrist_idx = 15 if dominant.startswith("left") else 16
    return {f: (float(p[wrist_idx, 0]), float(p[wrist_idx, 1]))
            for f, p in poses.items()}


def load_pose_pixels(csv_path, person_id, smash_sm):
    """Return {frame: ndarray(33, 2)} of 2D pixel coords for the smasher, restricted
    to the swing window. Single-frame landmark glitches (>100px from neighbours) are
    replaced by the local median; real fast motion is preserved."""
    df = pd.read_csv(csv_path)
    if "person_id" in df.columns and person_id is not None:
        df = df[df.person_id == person_id]
    df = df[(df.frame >= smash_sm["start"]) & (df.frame <= smash_sm["end"])]
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
    cube = _conditional_median(cube)
    out = {}
    for i, f in enumerate(frames_sorted):
        if not np.isnan(cube[i]).any():
            out[int(f)] = cube[i]
    return out


def normalize_pose(pts_px, target_cx, target_cy, target_torso_px):
    """Translate hip-midpoint to (target_cx, target_cy) and scale so torso length is target_torso_px."""
    lh, rh = pts_px[23], pts_px[24]
    ls, rs = pts_px[11], pts_px[12]
    hip_mid = (lh + rh) / 2.0
    sh_mid = (ls + rs) / 2.0
    torso_len = float(np.linalg.norm(sh_mid - hip_mid))
    if torso_len < 1e-6:
        return None
    scale = target_torso_px / torso_len
    return (pts_px - hip_mid) * scale + np.array([target_cx, target_cy])


def draw_skeleton(canvas, pts, color, thickness=3, joint_radius=5):
    pts_int = pts.astype(np.int32)
    h, w = canvas.shape[:2]
    def in_bounds(p):
        return 0 <= p[0] < w and 0 <= p[1] < h
    for a, b in SKELETON_EDGES:
        if a < len(pts_int) and b < len(pts_int):
            pa, pb = tuple(pts_int[a]), tuple(pts_int[b])
            if in_bounds(pa) and in_bounds(pb):
                cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
    for j in SKELETON_JOINTS:
        if j < len(pts_int) and in_bounds(tuple(pts_int[j])):
            cv2.circle(canvas, tuple(pts_int[j]), joint_radius, color, -1, cv2.LINE_AA)


def smooth_path(path, factor):
    """Insert `factor`-1 linearly interpolated sub-steps between consecutive (i,j) pairs.
    Returns a list of float tuples."""
    if factor <= 1 or len(path) < 2:
        return [(float(i), float(j)) for i, j in path]
    out = []
    for k in range(len(path) - 1):
        i0, j0 = path[k]
        i1, j1 = path[k + 1]
        for s in range(factor):
            t = s / factor
            out.append((i0 + t * (i1 - i0), j0 + t * (j1 - j0)))
    out.append((float(path[-1][0]), float(path[-1][1])))
    return out


def interpolate_pose(poses, frame_float, start_frame):
    """Linear-interp pose at a fractional local frame index."""
    f = start_frame + frame_float
    f0 = int(np.floor(f))
    f1 = f0 + 1
    t = f - f0
    if f0 not in poses and f1 not in poses:
        return None
    if f0 not in poses:
        return poses[f1]
    if f1 not in poses:
        return poses[f0]
    return poses[f0] * (1.0 - t) + poses[f1] * t


def interpolate_xy(xy_dict, frame_float, start_frame):
    """Linear-interp a (x,y) tuple at a fractional local frame index."""
    f = start_frame + frame_float
    f0 = int(np.floor(f))
    f1 = f0 + 1
    t = f - f0
    if f0 not in xy_dict and f1 not in xy_dict:
        return None
    if f0 not in xy_dict:
        return xy_dict[f1]
    if f1 not in xy_dict:
        return xy_dict[f0]
    return (xy_dict[f0][0] * (1 - t) + xy_dict[f1][0] * t,
            xy_dict[f0][1] * (1 - t) + xy_dict[f1][1] * t)


def interpolate_racket(rackets, frame_float, start_frame):
    """Linear-interp full racket bbox dict at a fractional local frame index."""
    f = start_frame + frame_float
    f0 = int(np.floor(f))
    f1 = f0 + 1
    t = f - f0
    if f0 not in rackets and f1 not in rackets:
        return None
    if f0 not in rackets:
        return dict(rackets[f1])
    if f1 not in rackets:
        return dict(rackets[f0])
    r0, r1 = rackets[f0], rackets[f1]
    return {k: r0[k] * (1 - t) + r1[k] * t for k in ("x1", "y1", "x2", "y2", "cx", "cy")}


def _metric_at(win, frame_float, start_frame, col):
    """Linear-interpolate `col` from a metrics dataframe at a fractional local frame."""
    f = start_frame + frame_float
    f0 = int(np.floor(f))
    f1 = f0 + 1
    t = f - f0
    row0 = win[win.frame == f0]
    row1 = win[win.frame == f1]
    if row0.empty and row1.empty:
        return None
    if row0.empty:
        v = float(row1[col].iloc[0])
    elif row1.empty:
        v = float(row0[col].iloc[0])
    else:
        v = float(row0[col].iloc[0]) * (1 - t) + float(row1[col].iloc[0]) * t
    return v if not np.isnan(v) else None


def load_video_frames(video_path, start_frame, end_frame):
    """Read [start_frame, end_frame] inclusive from a video into {frame_idx: BGR ndarray}."""
    cap = cv2.VideoCapture(video_path)
    out = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i > end_frame:
            break
        if start_frame <= i <= end_frame:
            out[i] = fr
        i += 1
    cap.release()
    return out


def paste_crop(canvas, src_frame, bbox, dest_box):
    """Crop the bbox from src_frame, resize to dest_box, and paste onto canvas.
    bbox: (x1, y1, x2, y2) in src_frame coords.
    dest_box: (ax1, ay1, ax2, ay2) in canvas coords."""
    sh, sw = src_frame.shape[:2]
    sx1, sy1, sx2, sy2 = [int(v) for v in bbox]
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(sw, sx2), min(sh, sy2)
    if sx2 - sx1 < 2 or sy2 - sy1 < 2:
        return
    crop = src_frame[sy1:sy2, sx1:sx2]

    ch, cw = canvas.shape[:2]
    dx1, dy1, dx2, dy2 = [int(v) for v in dest_box]
    dw, dh = dx2 - dx1, dy2 - dy1
    if dw < 2 or dh < 2:
        return
    crop_resized = cv2.resize(crop, (dw, dh))

    # Clip to canvas.
    cx1, cy1 = max(0, dx1), max(0, dy1)
    cx2, cy2 = min(cw, dx2), min(ch, dy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    src_x1 = cx1 - dx1
    src_y1 = cy1 - dy1
    src_x2 = src_x1 + (cx2 - cx1)
    src_y2 = src_y1 + (cy2 - cy1)
    canvas[cy1:cy2, cx1:cx2] = crop_resized[src_y1:src_y2, src_x1:src_x2]


def render_overlay_video(pro_csv, user_csv, pro_sm, user_sm, path, out_mp4,
                         canvas_size=900, torso_px=150, smooth_factor=8,
                         pro_win=None, user_win=None,
                         pro_racket_csv=None, user_racket_csv=None,
                         pro_video=None, user_video=None,
                         freeze_impact_frames=25):
    pro_poses = load_pose_pixels(pro_csv, pro_sm.get("person_id"), pro_sm)
    user_poses = load_pose_pixels(user_csv, user_sm.get("person_id"), user_sm)
    # Pair the smasher's racket per frame, using the same filtered wrist as the skeleton.
    pro_wrists = smasher_wrist_per_frame(pro_csv, pro_sm.get("person_id"),
                                         pro_sm["dominant"], pro_sm)
    user_wrists = smasher_wrist_per_frame(user_csv, user_sm.get("person_id"),
                                          user_sm["dominant"], user_sm)
    pro_rackets = load_smasher_rackets(pro_racket_csv, pro_wrists) if pro_racket_csv else {}
    user_rackets = load_smasher_rackets(user_racket_csv, user_wrists) if user_racket_csv else {}
    # Pre-load source frames so we can crop the actual racket pixels per step.
    pro_frames = (load_video_frames(pro_video, pro_sm["start"], pro_sm["end"])
                  if pro_video else {})
    user_frames = (load_video_frames(user_video, user_sm["start"], user_sm["end"])
                   if user_video else {})
    cx, cy = canvas_size // 2, canvas_size // 2 + 80

    PRO_COLOR = (255, 165, 0)    # blue-ish in BGR
    USER_COLOR = (60, 60, 255)   # red
    fps = pro_sm["fps"]
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (canvas_size, canvas_size))

    pro_impact_local = pro_sm["impact"] - pro_sm["start"]
    user_impact_local = user_sm["impact"] - user_sm["start"]
    fine_path = smooth_path(path, smooth_factor)
    impact_step = min(
        range(len(fine_path)),
        key=lambda k: abs(fine_path[k][0] - pro_impact_local) +
                      abs(fine_path[k][1] - user_impact_local),
    )
    # Which landmark to colour for the dominant elbow depends on dominant side.
    user_elbow_lm = 13 if user_sm["dominant"].startswith("left") else 14
    pro_wrist_lm = 15 if pro_sm["dominant"].startswith("left") else 16
    user_wrist_lm = 15 if user_sm["dominant"].startswith("left") else 16
    JOINT_BADGES = [
        (user_elbow_lm, "dominant_elbow_angle_deg", "codo",  10.0),
        (25,            "left_knee_angle_deg",      "rod L", 10.0),
        (26,            "right_knee_angle_deg",     "rod R", 10.0),
    ]

    for step, (i_f, j_f) in enumerate(fine_path):
        canvas = np.full((canvas_size, canvas_size, 3), 28, dtype=np.uint8)
        for k in range(0, canvas_size, 60):
            cv2.line(canvas, (k, 0), (k, canvas_size), (45, 45, 45), 1)
            cv2.line(canvas, (0, k), (canvas_size, k), (45, 45, 45), 1)
        cv2.circle(canvas, (cx, cy), 4, (100, 100, 100), -1)

        pro_pts = interpolate_pose(pro_poses, i_f, pro_sm["start"])
        user_pts = interpolate_pose(user_poses, j_f, user_sm["start"])
        pro_norm = user_norm = None
        if pro_pts is not None:
            pro_norm = normalize_pose(pro_pts, cx, cy, torso_px)
            if pro_norm is not None:
                draw_skeleton(canvas, pro_norm, PRO_COLOR)
        if user_pts is not None:
            user_norm = normalize_pose(user_pts, cx, cy, torso_px)
            if user_norm is not None:
                draw_skeleton(canvas, user_norm, USER_COLOR)

        # Racket only on the impact frame: crop the actual racket pixels from the
        # source video and paste them at the wrist-anchored normalized location.
        if step == impact_step:
            for pts, norm, wrist_lm, rackets, source_frames, sm, color, frame_float in [
                (pro_pts, pro_norm, pro_wrist_lm, pro_rackets, pro_frames, pro_sm, PRO_COLOR, i_f),
                (user_pts, user_norm, user_wrist_lm, user_rackets, user_frames, user_sm, USER_COLOR, j_f),
            ]:
                if pts is None or norm is None:
                    continue
                r = interpolate_racket(rackets, frame_float, sm["start"])
                if r is None:
                    continue
                lh, rh = pts[23], pts[24]
                ls, rs = pts[11], pts[12]
                hip_mid = (lh + rh) / 2.0
                torso_len = float(np.linalg.norm((ls + rs) / 2.0 - hip_mid))
                if torso_len < 1e-6:
                    continue
                scale = torso_px / torso_len

                def proj_xy(x, y):
                    return ((x - hip_mid[0]) * scale + cx,
                            (y - hip_mid[1]) * scale + cy)

                nx1, ny1 = proj_xy(r["x1"], r["y1"])
                nx2, ny2 = proj_xy(r["x2"], r["y2"])
                wx_n, wy_n = norm[wrist_lm][0], norm[wrist_lm][1]
                ax1, ay1, ax2, ay2, acx, acy = anchor_bbox_at_wrist(
                    nx1, ny1, nx2, ny2, (wx_n, wy_n))

                src_idx = int(round(sm["start"] + frame_float))
                src_frame = source_frames.get(src_idx)
                if src_frame is not None:
                    raw_r = rackets.get(src_idx, r)
                    paste_crop(canvas, src_frame,
                               (raw_r["x1"], raw_r["y1"], raw_r["x2"], raw_r["y2"]),
                               (ax1, ay1, ax2, ay2))
                cv2.rectangle(canvas, (int(ax1), int(ay1)), (int(ax2), int(ay2)),
                              color, 2, cv2.LINE_AA)

            # Racket angle (wrist → bbox-center direction) for both players, with delta.
            pro_int = int(round(pro_sm["start"] + i_f))
            user_int = int(round(user_sm["start"] + j_f))
            pro_w = pro_wrists.get(pro_int)
            user_w = user_wrists.get(user_int)
            pro_r = pro_rackets.get(pro_int)
            user_r = user_rackets.get(user_int)
            if pro_w and pro_r and user_w and user_r:
                pa = racket_angle_deg(pro_w, (pro_r["cx"], pro_r["cy"]))
                ua = racket_angle_deg(user_w, (user_r["cx"], user_r["cy"]))
                delta = ua - pa
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360
                d_color = _delta_color(delta, 15.0)
                lines = [
                    ("Angulo pala (desde vertical)", (200, 200, 200)),
                    (f"PRO  {pa:+.0f} deg", PRO_COLOR),
                    (f"USER {ua:+.0f} deg", USER_COLOR),
                    (f"Delta {delta:+.1f} deg", d_color),
                ]
                for k, (text, col) in enumerate(lines):
                    y = 60 + k * 26
                    cv2.putText(canvas, text, (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, text, (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)

        # Impact frame: vertical plumb lines from each smasher's RACKET CENTRE
        # (falls back to wrist if no racket detected at that frame).
        if step == impact_step:
            for norm, pts, wrist_lm, rackets, sm, color in [
                (pro_norm, pro_pts, pro_wrist_lm, pro_rackets, pro_sm, PRO_COLOR),
                (user_norm, user_pts, user_wrist_lm, user_rackets, user_sm, USER_COLOR),
            ]:
                if norm is None or pts is None:
                    continue
                # Compute the same hip-centred / torso-scaled transform used by normalize_pose.
                lh, rh = pts[23], pts[24]
                ls, rs = pts[11], pts[12]
                hip_mid = (lh + rh) / 2.0
                torso_len = float(np.linalg.norm((ls + rs) / 2.0 - hip_mid))
                if torso_len < 1e-6:
                    continue
                scale = torso_px / torso_len
                # Pick racket from the integer frame closest to the current fractional one.
                frame_int = int(round(sm["start"] + (i_f if color == PRO_COLOR else j_f)))
                r = rackets.get(frame_int)
                if r is not None:
                    px = (r["cx"] - hip_mid[0]) * scale + cx
                    py = (r["cy"] - hip_mid[1]) * scale + cy
                else:
                    px, py = norm[wrist_lm][0], norm[wrist_lm][1]
                px, py = int(px), int(py)
                cv2.line(canvas, (px, py), (px, canvas_size - 1), color, 1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 7, color, 2, cv2.LINE_AA)

        # Coloured badges only on the single impact frame.
        if step == impact_step and user_norm is not None and pro_win is not None and user_win is not None:
            for lm_idx, col, label, tol in JOINT_BADGES:
                pv = _metric_at(pro_win, i_f, pro_sm["start"], col)
                uv = _metric_at(user_win, j_f, user_sm["start"], col)
                if pv is None or uv is None:
                    continue
                d = uv - pv
                color = _delta_color(d, tol)
                jx, jy = int(user_norm[lm_idx][0]), int(user_norm[lm_idx][1])
                cv2.circle(canvas, (jx, jy), 14, color, 3, cv2.LINE_AA)
                sign = "+" if d >= 0 else ""
                txt = f"{sign}{d:.0f} deg"
                cv2.putText(canvas, txt, (jx + 18, jy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, txt, (jx + 18, jy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (canvas_size, 38), (20, 20, 20), -1)
        cv2.putText(canvas, "PRO", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PRO_COLOR, 2)
        cv2.putText(canvas, "YOU", (90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, USER_COLOR, 2)
        cv2.putText(canvas,
                    f"step {step + 1}/{len(fine_path)}  pro f{pro_sm['start'] + i_f:.1f}  user f{user_sm['start'] + j_f:.1f}",
                    (170, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Impact marker shown on a single frame.
        if step == impact_step:
            cv2.putText(canvas, "IMPACT", (canvas_size - 150, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(canvas)
        # Freeze on the exact impact frame so the badges/lines/label are readable.
        if step == impact_step:
            for _ in range(freeze_impact_frames):
                writer.write(canvas)
    writer.release()


def summarize(pro_win, user_win, path, pro_sm, user_sm):
    pro_impact_local = pro_sm["impact"] - pro_sm["start"]
    user_impact_local = user_sm["impact"] - user_sm["start"]

    per_metric = {}
    pro_vals = {c: pro_win[c].to_numpy() for c in METRIC_COLS}
    user_vals = {c: user_win[c].to_numpy() for c in METRIC_COLS}
    pro_idx = np.array([p[0] for p in path])
    user_idx = np.array([p[1] for p in path])
    for c in METRIC_COLS:
        diff = user_vals[c][user_idx] - pro_vals[c][pro_idx]
        mask = np.isfinite(diff)
        if not mask.any():
            continue
        per_metric[c] = {
            "delta_at_impact": float(user_vals[c][user_impact_local] -
                                     pro_vals[c][pro_impact_local])
                if (0 <= pro_impact_local < len(pro_vals[c])
                    and 0 <= user_impact_local < len(user_vals[c])
                    and np.isfinite(user_vals[c][user_impact_local])
                    and np.isfinite(pro_vals[c][pro_impact_local]))
                else None,
            "mean_abs_delta": float(np.mean(np.abs(diff[mask]))),
            "max_abs_delta": float(np.max(np.abs(diff[mask]))),
        }
    ranked = sorted(
        per_metric.items(),
        key=lambda kv: abs(kv[1].get("delta_at_impact") or 0) + kv[1]["mean_abs_delta"],
        reverse=True,
    )
    return {
        "pro_window": {"start": pro_sm["start"], "impact": pro_sm["impact"], "end": pro_sm["end"]},
        "user_window": {"start": user_sm["start"], "impact": user_sm["impact"], "end": user_sm["end"]},
        "warp_path_length": len(path),
        "per_metric": per_metric,
        "biggest_diffs": [k for k, _ in ranked[:5]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pro-metrics", required=True)
    p.add_argument("--pro-smash", required=True)
    p.add_argument("--pro-video", required=True)
    p.add_argument("--user-metrics", required=True)
    p.add_argument("--user-smash", required=True)
    p.add_argument("--user-video", required=True)
    p.add_argument("--pro-pose", help="pro pose CSV (defaults to <pro-metrics> with _metrics→_pose)")
    p.add_argument("--user-pose", help="user pose CSV (defaults likewise)")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    p.add_argument("--name", default="comparison")
    p.add_argument("--smooth", type=int, default=8,
                   help="overlay slow-mo factor (frames inserted between integer DTW steps)")
    p.add_argument("--torso-px", type=int, default=150,
                   help="rendered torso length in pixels (lower = smaller skeleton, more margin)")
    args = p.parse_args()
    args.pro_pose = args.pro_pose or args.pro_metrics.replace("_metrics.csv", "_pose.csv")
    args.user_pose = args.user_pose or args.user_metrics.replace("_metrics.csv", "_pose.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    pro_win, pro_sm = load_window(args.pro_metrics, args.pro_smash)
    user_win, user_sm = load_window(args.user_metrics, args.user_smash)
    print(f"pro window: {len(pro_win)} frames | user window: {len(user_win)} frames")
    pro_win, pro_sm, user_win, user_sm = trim_to_common(pro_win, pro_sm, user_win, user_sm)
    print(f"after trim   pro: {len(pro_win)} frames  user: {len(user_win)} frames")

    # DTW on the dominant-wrist-height signal, anchored at impact so the impact
    # frames of pro and user are guaranteed to coincide in the warped sequence.
    pro_h = uniform_filter1d(pro_win["dominant_wrist_height_m"].to_numpy(), 3)
    user_h = uniform_filter1d(user_win["dominant_wrist_height_m"].to_numpy(), 3)
    pro_impact_local = pro_sm["impact"] - pro_sm["start"]
    user_impact_local = user_sm["impact"] - user_sm["start"]
    path, d = dtw_align_anchored(pro_h, user_h, pro_impact_local, user_impact_local)
    impact_step = next((s for s, (i, j) in enumerate(path)
                        if i == pro_impact_local and j == user_impact_local), None)
    print(f"DTW distance (wrist height, anchored): {d:.3f}  path length: {len(path)}  "
          f"impact at step {impact_step}")

    out_png = os.path.join(args.out_dir, f"{args.name}.png")
    out_json = os.path.join(args.out_dir, f"{args.name}.json")
    out_sxs = os.path.join(args.out_dir, f"{args.name}_sxs.mp4")
    out_overlay = os.path.join(args.out_dir, f"{args.name}_overlay.mp4")

    render_paired_plot(pro_win, user_win, path, pro_sm, user_sm, out_png)
    summary = summarize(pro_win, user_win, path, pro_sm, user_sm)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    render_sxs_video(args.pro_video, args.user_video, pro_sm, user_sm, path, out_sxs,
                     pro_win=pro_win, user_win=user_win)
    pro_racket_csv = args.pro_pose.replace("_pose.csv", "_racket.csv")
    user_racket_csv = args.user_pose.replace("_pose.csv", "_racket.csv")
    render_overlay_video(args.pro_pose, args.user_pose, pro_sm, user_sm, path, out_overlay,
                         smooth_factor=args.smooth, torso_px=args.torso_px,
                         pro_win=pro_win, user_win=user_win,
                         pro_racket_csv=pro_racket_csv, user_racket_csv=user_racket_csv,
                         pro_video=args.pro_video, user_video=args.user_video)

    print(json.dumps(summary, indent=2))
    print(f"plot    -> {out_png}")
    print(f"json    -> {out_json}")
    print(f"sxs     -> {out_sxs}")
    print(f"overlay -> {out_overlay}")


if __name__ == "__main__":
    main()
