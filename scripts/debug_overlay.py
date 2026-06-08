"""Print path and compare pro CSV positions vs what overlay shows for the first N steps."""
import json
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from compare import dtw_align_anchored, load_pose_pixels, normalize_pose
from scipy.ndimage import uniform_filter1d

PRO_METRICS = r"C:\claude\padel-coach\output\garrido_smash_metrics.csv"
PRO_SMASH = r"C:\claude\padel-coach\output\garrido_smash_smash.json"
PRO_POSE = r"C:\claude\padel-coach\output\garrido_smash_pose.csv"
USER_METRICS = r"C:\claude\padel-coach\output\user_smash_metrics.csv"
USER_SMASH = r"C:\claude\padel-coach\output\user_smash_smash.json"
USER_POSE = r"C:\claude\padel-coach\output\user_smash_pose.csv"

pro_sm = json.load(open(PRO_SMASH))
user_sm = json.load(open(USER_SMASH))

pro_df = pd.read_csv(PRO_METRICS)
user_df = pd.read_csv(USER_METRICS)
pro_win = pro_df[(pro_df.frame >= pro_sm["start"]) & (pro_df.frame <= pro_sm["end"])].reset_index(drop=True)
user_win = user_df[(user_df.frame >= user_sm["start"]) & (user_df.frame <= user_sm["end"])].reset_index(drop=True)

pro_h = uniform_filter1d(pro_win["dominant_wrist_height_m"].to_numpy(), 3)
user_h = uniform_filter1d(user_win["dominant_wrist_height_m"].to_numpy(), 3)
pi = pro_sm["impact"] - pro_sm["start"]
ui = user_sm["impact"] - user_sm["start"]
path, d = dtw_align_anchored(pro_h, user_h, pi, ui)

print(f"pro window {pro_sm['start']}..{pro_sm['end']}  pro_impact_local={pi}")
print(f"user window {user_sm['start']}..{user_sm['end']}  user_impact_local={ui}")
print(f"path length = {len(path)}\n")

pro_poses = load_pose_pixels(PRO_POSE, pro_sm.get("person_id"), pro_sm)
LMS = ["L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle"]
LM_IDX = [23, 24, 25, 26, 27, 28]

for step, (i, j) in enumerate(path[:8]):
    pro_frame = pro_sm["start"] + i
    if pro_frame not in pro_poses:
        print(f"step {step}: i={i} j={j}  pro_frame={pro_frame}  NO POSE")
        continue
    pts = pro_poses[pro_frame]
    raw = ", ".join(f"{n}=({int(pts[idx][0])},{int(pts[idx][1])})"
                    for n, idx in zip(LMS, LM_IDX))
    norm = normalize_pose(pts, 450, 530, 220)
    norm_str = ", ".join(f"{n}=({int(norm[idx][0])},{int(norm[idx][1])})"
                         for n, idx in zip(LMS, LM_IDX))
    print(f"step {step:2d}: i={i} j={j}  pro_frame={pro_frame}")
    print(f"  RAW : {raw}")
    print(f"  NORM: {norm_str}")
