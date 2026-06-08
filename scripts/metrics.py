"""Extract biomechanical metrics from a pose CSV + smash JSON.

Per-frame metrics (computed in MediaPipe's hip-centered 3D world frame, meters):
  - dominant_elbow_angle_deg   (180° = fully extended)
  - dominant_wrist_speed_ms    (3D speed in m/s)
  - dominant_wrist_height_m    (negative = above hip; MediaPipe y-axis is down)
  - left_knee_angle_deg
  - right_knee_angle_deg
  - trunk_lean_deg             (angle of shoulder-midpoint→hip-midpoint from vertical)
  - hip_shoulder_separation_m  (vertical distance, indicates body extension)
  - feet_separation_m          (horizontal distance between ankles, ignoring height)

Summary metrics (at key smash phases):
  - elbow_angle_at_impact_deg
  - peak_wrist_speed_ms        (and the frame it happened on)
  - wrist_speed_at_impact_ms
  - knee_angles_at_start/impact
  - trunk_lean_at_impact_deg
  - timings: prep_to_impact_s, follow_through_s

Outputs (next to the input CSV):
  - <name>_metrics.csv     long-format per-frame
  - <name>_metrics.json    summary
  - <name>_metrics.png     time-series plot with phase markers
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# Landmark indices (MediaPipe Pose, 33 keypoints).
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}


def to_world_arrays(df, n_frames):
    """Returns dict {lm_name: ndarray shape (n_frames, 3)} in MediaPipe world coords (m)."""
    out = {}
    for name in LM:
        sub = df[df.lm_name == name].sort_values("frame")
        a = np.full((n_frames, 3), np.nan)
        for _, row in sub.iterrows():
            f = int(row.frame)
            if 0 <= f < n_frames and pd.notna(row.x_world):
                a[f] = [row.x_world, row.y_world, row.z_world]
        out[name] = a
    return out


def joint_angle_deg(a, b, c):
    """Vectorized angle at vertex b (degrees). Inputs shape (N, 3)."""
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=1)
    norm = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))


def speed_ms(coords_3d, fps):
    """3D speed in m/s using central differences. NaN-safe (returns NaN where any neighbour is NaN)."""
    n = len(coords_3d)
    out = np.full(n, np.nan)
    for i in range(1, n - 1):
        a, b = coords_3d[i - 1], coords_3d[i + 1]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            out[i] = np.linalg.norm(b - a) * fps / 2.0
    return out


def trunk_lean_deg(shoulder_mid, hip_mid):
    """Angle (deg) of trunk vector (hip→shoulder) from vertical (negative y in MediaPipe world)."""
    v = shoulder_mid - hip_mid  # ideally pointing up = -y
    # vertical reference vector
    vert = np.array([0.0, -1.0, 0.0])
    dot = v @ vert
    norm = np.linalg.norm(v, axis=1) * np.linalg.norm(vert) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="pose CSV from pose.py")
    p.add_argument("smash_json", help="smash phases JSON from detect_smash.py")
    p.add_argument("--out-dir", default=r"C:\claude\padel-coach\output")
    args = p.parse_args()

    base = os.path.splitext(os.path.basename(args.csv))[0].replace("_pose", "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{base}_metrics.csv")
    out_json = os.path.join(args.out_dir, f"{base}_metrics.json")
    out_png = os.path.join(args.out_dir, f"{base}_metrics.png")

    df = pd.read_csv(args.csv)
    with open(args.smash_json) as f:
        smash = json.load(f)
    dominant = smash["dominant"]              # "left_wrist" or "right_wrist"
    side = dominant.split("_")[0]             # "left" or "right"
    fps = smash["fps"]
    # Filter to the smasher's pose only.
    if "person_id" in df.columns and "person_id" in smash:
        df = df[df.person_id == smash["person_id"]]
    n_frames = int(df.frame.max()) + 1

    W = to_world_arrays(df, n_frames)

    # Per-frame metric arrays.
    dom_shoulder = W[f"{side}_shoulder"]
    dom_elbow = W[f"{side}_elbow"]
    dom_wrist = W[f"{side}_wrist"]
    elbow_angle = joint_angle_deg(dom_shoulder, dom_elbow, dom_wrist)
    wrist_speed = speed_ms(dom_wrist, fps)
    wrist_height = dom_wrist[:, 1]  # MediaPipe world y; -y = up

    l_knee_ang = joint_angle_deg(W["left_hip"], W["left_knee"], W["left_ankle"])
    r_knee_ang = joint_angle_deg(W["right_hip"], W["right_knee"], W["right_ankle"])

    shoulder_mid = (W["left_shoulder"] + W["right_shoulder"]) / 2.0
    hip_mid = (W["left_hip"] + W["right_hip"]) / 2.0
    trunk = trunk_lean_deg(shoulder_mid, hip_mid)
    hip_shoulder_sep = hip_mid[:, 1] - shoulder_mid[:, 1]  # positive = shoulders above hips

    # Foot separation: distance in the horizontal (xz) plane between ankles.
    ankle_diff = W["left_ankle"] - W["right_ankle"]
    feet_sep = np.sqrt(ankle_diff[:, 0] ** 2 + ankle_diff[:, 2] ** 2)

    # Long-format CSV.
    rows = []
    for f in range(n_frames):
        rows.append({
            "frame": f,
            "t": f / fps,
            "dominant_elbow_angle_deg": elbow_angle[f],
            "dominant_wrist_speed_ms": wrist_speed[f],
            "dominant_wrist_height_m": wrist_height[f],
            "left_knee_angle_deg": l_knee_ang[f],
            "right_knee_angle_deg": r_knee_ang[f],
            "trunk_lean_deg": trunk[f],
            "hip_shoulder_separation_m": hip_shoulder_sep[f],
            "feet_separation_m": feet_sep[f],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.4f")

    # Summary at key phases.
    def at(arr, idx):
        v = arr[idx] if 0 <= idx < len(arr) else float("nan")
        return None if not np.isfinite(v) else float(v)

    swing_slice = slice(smash["start"], smash["end"] + 1)
    ws = wrist_speed[swing_slice]
    peak_speed_local = int(np.nanargmax(ws)) if np.any(np.isfinite(ws)) else 0
    peak_speed_frame = smash["start"] + peak_speed_local

    summary = {
        "video": base,
        "dominant": dominant,
        "fps": fps,
        "phases": {
            "start": smash["start"],
            "apex_impact": smash["impact"],
            "end": smash["end"],
            "prep_to_impact_s": (smash["impact"] - smash["start"]) / fps,
            "follow_through_s": (smash["end"] - smash["impact"]) / fps,
        },
        "at_impact": {
            "elbow_angle_deg": at(elbow_angle, smash["impact"]),
            "wrist_speed_ms": at(wrist_speed, smash["impact"]),
            "wrist_height_m": at(wrist_height, smash["impact"]),
            "left_knee_angle_deg": at(l_knee_ang, smash["impact"]),
            "right_knee_angle_deg": at(r_knee_ang, smash["impact"]),
            "trunk_lean_deg": at(trunk, smash["impact"]),
            "feet_separation_m": at(feet_sep, smash["impact"]),
        },
        "at_start": {
            "left_knee_angle_deg": at(l_knee_ang, smash["start"]),
            "right_knee_angle_deg": at(r_knee_ang, smash["start"]),
            "wrist_height_m": at(wrist_height, smash["start"]),
            "feet_separation_m": at(feet_sep, smash["start"]),
        },
        "peak_wrist_speed": {
            "value_ms": at(wrist_speed, peak_speed_frame),
            "frame": int(peak_speed_frame),
            "offset_from_impact_frames": int(peak_speed_frame - smash["impact"]),
        },
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Plot time-series with phase markers.
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    t = np.arange(n_frames) / fps

    axes[0].plot(t, uniform_filter1d(np.nan_to_num(wrist_speed, nan=0.0), 3),
                 color="crimson")
    axes[0].set_ylabel("Velocidad muñeca (m/s)")
    axes[0].set_title(f"{base} — dominante: {dominant}")

    axes[1].plot(t, uniform_filter1d(np.nan_to_num(elbow_angle, nan=0.0), 3),
                 color="darkorange")
    axes[1].set_ylabel("Ángulo codo (°)\n180=extendido")

    axes[2].plot(t, uniform_filter1d(np.nan_to_num(l_knee_ang, nan=0.0), 3),
                 color="steelblue", label="rodilla izq")
    axes[2].plot(t, uniform_filter1d(np.nan_to_num(r_knee_ang, nan=0.0), 3),
                 color="seagreen", label="rodilla der")
    axes[2].set_ylabel("Ángulo rodilla (°)")
    axes[2].legend(loc="lower right", fontsize=8)

    axes[3].plot(t, uniform_filter1d(np.nan_to_num(hip_shoulder_sep, nan=0.0), 3),
                 color="purple")
    axes[3].set_ylabel("Cadera-hombro Δy (m)\n+ = extensión")

    axes[4].plot(t, uniform_filter1d(np.nan_to_num(feet_sep, nan=0.0), 3),
                 color="teal")
    axes[4].set_ylabel("Separación pies (m)\nbase horizontal")
    axes[4].set_xlabel("tiempo (s)")

    for ax in axes:
        for k, color in [("start", "gray"), ("apex_impact", "red"), ("end", "gray")]:
            ax.axvline(summary["phases"][k] / fps,
                       color=color, ls="--" if k != "apex_impact" else "-",
                       lw=2 if k == "apex_impact" else 1, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"csv  -> {out_csv}")
    print(f"json -> {out_json}")
    print(f"plot -> {out_png}")


if __name__ == "__main__":
    main()
