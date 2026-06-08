"""Generate a static index.html showing every output of the pipeline.

The HTML is self-contained (CSS inlined) and references the video/png files in
../output/ via relative paths. Open the file directly in any modern browser —
no server needed.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
from html import escape

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as cmp_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(ROOT, "output")
WEB_DIR = os.path.join(ROOT, "web")
WEB_VIDEOS = os.path.join(WEB_DIR, "videos")
WEB_PLOTS = os.path.join(WEB_DIR, "plots")
os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(WEB_VIDEOS, exist_ok=True)
os.makedirs(WEB_PLOTS, exist_ok=True)
INDEX_HTML = os.path.join(WEB_DIR, "demo.html")   # landing is the real index.html now
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def reencode_h264(src, dst):
    """Re-encode to browser-friendly H.264 if not already done (mtime check)."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    print(f"  re-encoding {os.path.basename(src)} -> H.264")
    subprocess.run(
        [FFMPEG, "-y", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "23",
         "-an", dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def copy_if_newer(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    shutil.copy2(src, dst)

PRO_BASE = "garrido_smash"
USER_BASE = "user_smash"

METRIC_LABELS = {
    "dominant_elbow_angle_deg": ("Codo (dominante)", "deg", 10.0),
    "dominant_wrist_speed_ms":  ("Velocidad muñeca", "m/s", 1.0),
    "dominant_wrist_height_m":  ("Altura muñeca",    "m",   0.10),
    "left_knee_angle_deg":      ("Rodilla izquierda","deg", 10.0),
    "right_knee_angle_deg":     ("Rodilla derecha",  "deg", 10.0),
    "trunk_lean_deg":           ("Inclinación tronco","deg",5.0),
    "hip_shoulder_separation_m":("Separación cadera-hombro","m",0.05),
    "feet_separation_m":        ("Separación pies",  "m",   0.05),
}


VIDEOS_TO_BUNDLE = [
    "comparison_overlay.mp4",
    "comparison_sxs.mp4",
    f"{PRO_BASE}_smash_marked.mp4",
    f"{USER_BASE}_smash_marked.mp4",
    f"{PRO_BASE}_pose.mp4",
    f"{USER_BASE}_pose.mp4",
]
PLOTS_TO_BUNDLE = [
    "comparison.png",
    f"{PRO_BASE}_metrics.png",
    f"{USER_BASE}_metrics.png",
    f"{PRO_BASE}_smash.png",
    f"{USER_BASE}_smash.png",
]
JSON_CSV_TO_BUNDLE = [
    "comparison.json",
    f"{PRO_BASE}_smash.json",
    f"{USER_BASE}_smash.json",
    f"{PRO_BASE}_pose.csv",
    f"{USER_BASE}_pose.csv",
    f"{PRO_BASE}_metrics.csv",
    f"{USER_BASE}_metrics.csv",
]


def bundle_assets():
    """Re-encode videos to H.264 and copy plots/JSON/CSV into web/ so the page is self-contained."""
    for v in VIDEOS_TO_BUNDLE:
        src = os.path.join(OUTPUT_DIR, v)
        if not os.path.exists(src):
            print(f"  ! missing {v}")
            continue
        reencode_h264(src, os.path.join(WEB_VIDEOS, v))
    for p in PLOTS_TO_BUNDLE:
        src = os.path.join(OUTPUT_DIR, p)
        if os.path.exists(src):
            copy_if_newer(src, os.path.join(WEB_PLOTS, p))
    data_dir = os.path.join(WEB_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    for j in JSON_CSV_TO_BUNDLE:
        src = os.path.join(OUTPUT_DIR, j)
        if os.path.exists(src):
            copy_if_newer(src, os.path.join(data_dir, j))


def build_skeleton_steps_json(out_path, canvas_size=900, torso_px=150, smooth_factor=8):
    """Pre-compute every DTW step's normalized poses + anchored racket bboxes,
    dump as a single JSON the browser can scrub through interactively."""
    pro_metrics = os.path.join(OUTPUT_DIR, f"{PRO_BASE}_metrics.csv")
    user_metrics = os.path.join(OUTPUT_DIR, f"{USER_BASE}_metrics.csv")
    pro_smash = os.path.join(OUTPUT_DIR, f"{PRO_BASE}_smash.json")
    user_smash = os.path.join(OUTPUT_DIR, f"{USER_BASE}_smash.json")
    pro_pose_csv = os.path.join(OUTPUT_DIR, f"{PRO_BASE}_pose.csv")
    user_pose_csv = os.path.join(OUTPUT_DIR, f"{USER_BASE}_pose.csv")
    pro_racket_csv = os.path.join(OUTPUT_DIR, f"{PRO_BASE}_racket.csv")
    user_racket_csv = os.path.join(OUTPUT_DIR, f"{USER_BASE}_racket.csv")

    pro_win, pro_sm = cmp_mod.load_window(pro_metrics, pro_smash)
    user_win, user_sm = cmp_mod.load_window(user_metrics, user_smash)
    pro_win, pro_sm, user_win, user_sm = cmp_mod.trim_to_common(
        pro_win, pro_sm, user_win, user_sm)

    pro_h = uniform_filter1d(pro_win["dominant_wrist_height_m"].to_numpy(), 3)
    user_h = uniform_filter1d(user_win["dominant_wrist_height_m"].to_numpy(), 3)
    pi = pro_sm["impact"] - pro_sm["start"]
    ui = user_sm["impact"] - user_sm["start"]
    path, _ = cmp_mod.dtw_align_anchored(pro_h, user_h, pi, ui)
    fine_path = cmp_mod.smooth_path(path, smooth_factor)
    impact_step = min(
        range(len(fine_path)),
        key=lambda k: abs(fine_path[k][0] - pi) + abs(fine_path[k][1] - ui),
    )

    pro_poses = cmp_mod.load_pose_pixels(pro_pose_csv, pro_sm.get("person_id"), pro_sm)
    user_poses = cmp_mod.load_pose_pixels(user_pose_csv, user_sm.get("person_id"), user_sm)
    pro_wrists = cmp_mod.smasher_wrist_per_frame(
        pro_pose_csv, pro_sm.get("person_id"), pro_sm["dominant"], pro_sm)
    user_wrists = cmp_mod.smasher_wrist_per_frame(
        user_pose_csv, user_sm.get("person_id"), user_sm["dominant"], user_sm)
    pro_rackets = cmp_mod.load_smasher_rackets(pro_racket_csv, pro_wrists) if os.path.exists(pro_racket_csv) else {}
    user_rackets = cmp_mod.load_smasher_rackets(user_racket_csv, user_wrists) if os.path.exists(user_racket_csv) else {}

    cx, cy = canvas_size // 2, canvas_size // 2 + 80
    pro_wrist_lm = 15 if pro_sm["dominant"].startswith("left") else 16
    user_wrist_lm = 15 if user_sm["dominant"].startswith("left") else 16

    def pack_step(pts, racket, sm, wrist_lm, frame_float):
        if pts is None:
            return None, None
        norm = cmp_mod.normalize_pose(pts, cx, cy, torso_px)
        if norm is None:
            return None, None
        lh, rh = pts[23], pts[24]
        ls, rs = pts[11], pts[12]
        hip_mid = (lh + rh) / 2.0
        torso_len = float(np.linalg.norm((ls + rs) / 2.0 - hip_mid))
        if torso_len < 1e-6:
            return [[float(round(p[0], 1)), float(round(p[1], 1))] for p in norm], None
        scale = torso_px / torso_len
        pose_out = [[float(round(p[0], 1)), float(round(p[1], 1))] for p in norm]
        racket_out = None
        if racket is not None:
            nx1 = (racket["x1"] - hip_mid[0]) * scale + cx
            ny1 = (racket["y1"] - hip_mid[1]) * scale + cy
            nx2 = (racket["x2"] - hip_mid[0]) * scale + cx
            ny2 = (racket["y2"] - hip_mid[1]) * scale + cy
            wx_n, wy_n = norm[wrist_lm][0], norm[wrist_lm][1]
            ax1, ay1, ax2, ay2, _, _ = cmp_mod.anchor_bbox_at_wrist(
                nx1, ny1, nx2, ny2, (wx_n, wy_n))
            racket_out = [round(ax1, 1), round(ay1, 1), round(ax2, 1), round(ay2, 1)]
        return pose_out, racket_out

    steps = []
    for i_f, j_f in fine_path:
        pro_pts = cmp_mod.interpolate_pose(pro_poses, i_f, pro_sm["start"])
        user_pts = cmp_mod.interpolate_pose(user_poses, j_f, user_sm["start"])
        pro_r = cmp_mod.interpolate_racket(pro_rackets, i_f, pro_sm["start"])
        user_r = cmp_mod.interpolate_racket(user_rackets, j_f, user_sm["start"])
        pro_pose, pro_box = pack_step(pro_pts, pro_r, pro_sm, pro_wrist_lm, i_f)
        user_pose, user_box = pack_step(user_pts, user_r, user_sm, user_wrist_lm, j_f)
        steps.append({
            "pro": pro_pose, "user": user_pose,
            "pro_racket": pro_box, "user_racket": user_box,
            "pro_frame": round(pro_sm["start"] + i_f, 1),
            "user_frame": round(user_sm["start"] + j_f, 1),
        })

    # Lowest racket-tip drop during the preparation (start→impact) for each player.
    # Calibrated in cm using the known padel-racket length (45.5cm) — for each clip
    # we take the max bbox diagonal across the full swing as that player's
    # "racket fully face-on" reference, then convert pixels→cm with it.
    RACKET_LENGTH_CM = 45.5

    def bbox_diag(racket):
        if racket is None:
            return 0.0
        x1, y1, x2, y2 = racket["x1"], racket["y1"], racket["x2"], racket["y2"]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def tip_corner_img(racket, wrist):
        if racket is None or wrist is None:
            return None
        x1, y1, x2, y2 = racket["x1"], racket["y1"], racket["x2"], racket["y2"]
        wx, wy = wrist
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        return max(corners, key=lambda c: (c[0] - wx) ** 2 + (c[1] - wy) ** 2)

    def lowest_tip_drop_cm(rackets, wrists, sm):
        if not rackets or not wrists:
            return None
        max_diag = max((bbox_diag(r) for r in rackets.values()), default=0.0)
        if max_diag < 1:
            return None
        px_per_cm = max_diag / RACKET_LENGTH_CM
        best_drop_px = -1e9
        best_frame = None
        for f in range(sm["start"], sm["impact"] + 1):
            r = rackets.get(f); w = wrists.get(f)
            tip = tip_corner_img(r, w) if (r and w) else None
            if tip is None:
                continue
            drop_px = tip[1] - w[1]  # positive = tip BELOW wrist in image
            if drop_px > best_drop_px:
                best_drop_px = drop_px
                best_frame = f
        if best_frame is None:
            return None
        return {
            "drop_cm": round(best_drop_px / px_per_cm, 1),
            "frame": best_frame,
            "px_per_cm": round(px_per_cm, 2),
        }

    pro_prep_drop = lowest_tip_drop_cm(pro_rackets, pro_wrists, pro_sm)
    user_prep_drop = lowest_tip_drop_cm(user_rackets, user_wrists, user_sm)
    prep_drop_compare = None
    if pro_prep_drop and user_prep_drop:
        delta = user_prep_drop["drop_cm"] - pro_prep_drop["drop_cm"]
        mag = abs(delta) / 10.0  # tolerance: 10 cm
        col = "#2ecc71" if mag < 1 else ("#f1c40f" if mag < 2 else "#ff5e5e")
        prep_drop_compare = {
            "pro_cm": pro_prep_drop["drop_cm"],
            "user_cm": user_prep_drop["drop_cm"],
            "delta_cm": round(delta, 1),
            "color": col,
        }

    # Impact-frame racket crops: encode the actual racket pixels from each clip's
    # impact frame as base64 PNGs so the browser can draw them at the anchored bbox.
    def racket_crop_b64(video_path, rackets, frame_idx):
        if not os.path.exists(video_path) or frame_idx not in rackets:
            return None
        cap = cv2.VideoCapture(video_path)
        i = 0
        frame = None
        while True:
            ok, f = cap.read()
            if not ok or i > frame_idx:
                break
            if i == frame_idx:
                frame = f
                break
            i += 1
        cap.release()
        if frame is None:
            return None
        r = rackets[frame_idx]
        sh, sw = frame.shape[:2]
        x1, y1 = max(0, int(r["x1"])), max(0, int(r["y1"]))
        x2, y2 = min(sw, int(r["x2"])), min(sh, int(r["y2"]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = frame[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    pro_video_path = os.path.join(ROOT, "videos", "pro", f"{PRO_BASE}.webm")
    user_video_path = os.path.join(ROOT, "videos", "user", f"{USER_BASE}.webm")
    pro_racket_img = racket_crop_b64(pro_video_path, pro_rackets, pro_sm["impact"])
    user_racket_img = racket_crop_b64(user_video_path, user_rackets, user_sm["impact"])

    # Racket-angle (handle direction, from vertical) at impact for both clips.
    racket_angle = {"pro": None, "user": None, "delta": None}
    pw = pro_wrists.get(pro_sm["impact"])
    uw = user_wrists.get(user_sm["impact"])
    pr = pro_rackets.get(pro_sm["impact"])
    ur = user_rackets.get(user_sm["impact"])
    if pw and uw and pr and ur:
        pa = cmp_mod.racket_angle_deg(pw, (pr["cx"], pr["cy"]))
        ua = cmp_mod.racket_angle_deg(uw, (ur["cx"], ur["cy"]))
        d = ua - pa
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        mag = abs(d) / 15.0
        d_col = "#2ecc71" if mag < 1 else ("#f1c40f" if mag < 2 else "#ff5e5e")
        racket_angle = {
            "pro": round(pa, 1), "user": round(ua, 1),
            "delta": round(d, 1), "color": d_col,
        }

    # Impact-frame badges: deltas at key joints for the visual overlay.
    impact_metrics = []
    user_elbow_lm = 13 if user_sm["dominant"].startswith("left") else 14
    metric_cfg = [
        (user_elbow_lm, "dominant_elbow_angle_deg", 10.0),
        (25,            "left_knee_angle_deg",      10.0),
        (26,            "right_knee_angle_deg",     10.0),
    ]
    pro_imp_row = pro_win[pro_win.frame == pro_sm["impact"]]
    user_imp_row = user_win[user_win.frame == user_sm["impact"]]
    if len(pro_imp_row) and len(user_imp_row):
        for lm_idx, col, tol in metric_cfg:
            pv = float(pro_imp_row[col].iloc[0])
            uv = float(user_imp_row[col].iloc[0])
            if not (np.isfinite(pv) and np.isfinite(uv)):
                continue
            d = uv - pv
            mag = abs(d) / tol
            color = "#2ecc71" if mag < 1 else ("#f1c40f" if mag < 2 else "#ff5e5e")
            impact_metrics.append({"lm": lm_idx, "delta": round(d, 1), "color": color})

    data = {
        "canvas_size": canvas_size,
        "torso_px": torso_px,
        "hip_center": [cx, cy],
        "impact_step": int(impact_step),
        "edges": cmp_mod.SKELETON_EDGES,
        "joints": list(range(33)),
        "pro_color": "#4ea1ff",
        "user_color": "#ff5e5e",
        "pro_wrist_lm": pro_wrist_lm,
        "user_wrist_lm": user_wrist_lm,
        "impact_badges": impact_metrics,
        "pro_racket_img": pro_racket_img,
        "user_racket_img": user_racket_img,
        "racket_angle": racket_angle,
        "prep_drop": prep_drop_compare,
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  skeleton steps -> {out_path} ({len(steps)} steps, {os.path.getsize(out_path)//1024} KB)")


def generate_coaching(per_metric, prep_drop, racket_angle):
    """Rule-based Spanish coaching advice from the comparison deltas.
    Returns list of {severity, title, body, priority} sorted by priority desc.
    Focus on impact-moment mechanics."""
    advice = []

    def add(priority, severity, title, body):
        advice.append({"priority": priority, "severity": severity,
                       "title": title, "body": body})

    # ====================================================================
    # IMPACT-MOMENT MECHANICS (the most important — these come first)
    # ====================================================================

    # --- Velocidad de la muñeca en el contacto ---
    ws = per_metric.get("dominant_wrist_speed_ms", {})
    d = ws.get("delta_at_impact")
    if d is not None:
        if d < -1.5:
            add(abs(d) * 12, "bad",
                "Tu muñeca llega lenta al impacto",
                f"En el momento del contacto tu muñeca va {abs(d):.1f} m/s más despacio que la del pro. "
                f"Esto es probablemente lo que más limita la potencia de tu smash. "
                f"Trabaja la aceleración progresiva: pausado al armar, explosivo en el último tercio del swing. "
                f"Imagina dar un 'latigazo' con la muñeca justo antes de tocar la bola.")
        elif d > 2:
            add(abs(d) * 4, "mid",
                "Velocidad muy alta — controla la tensión",
                f"Tu muñeca va {abs(d):.1f} m/s MÁS rápido que la del pro. Mucha velocidad cruda, pero si te falta precisión "
                f"puede ser que estés impactando con demasiada tensión muscular en lugar de soltar el latigazo.")

    # --- Altura del impacto (qué tan arriba contactas la bola) ---
    wh = per_metric.get("dominant_wrist_height_m", {})
    d = wh.get("delta_at_impact")
    if d is not None:
        # MediaPipe world y is positive DOWN. delta > 0 = user wrist is LOWER than pro's.
        d_cm = d * 100
        if d_cm > 8:
            add(d_cm * 4, "bad",
                "Impacta la bola más arriba",
                f"Tu muñeca está {d_cm:.0f} cm más baja que la del pro al contactar la bola. "
                f"Cuanto más alto golpees, mejor ángulo descendente le das al smash (la bola pica más cerca de tu línea y rebota más alto). "
                f"Para subir el impacto: estira completamente el brazo, sube los hombros, sube los talones y, si hace falta, salta.")
        elif d_cm < -8:
            add(abs(d_cm) * 2, "good",
                "Buen impacto en altura",
                f"Tu muñeca está {abs(d_cm):.0f} cm MÁS arriba que la del pro al contactar — bien hecho, esto te da ventaja en el ángulo de ataque.")

    # --- Codo dominante en el impacto ---
    el = per_metric.get("dominant_elbow_angle_deg", {})
    d = el.get("delta_at_impact")
    if d is not None and abs(d) >= 5:
        if d < 0:
            sev = "bad" if abs(d) > 20 else "mid"
            qualifier = "ligeramente" if abs(d) < 10 else ("notablemente" if abs(d) < 20 else "claramente")
            add(abs(d) * 2.0, sev,
                "Extiende más el codo al impactar",
                f"Tu codo está {abs(d):.0f}° {qualifier} más flexionado que el del pro en el contacto. "
                f"Un brazo más recto sube el punto de impacto y mejora el ángulo de la cara de la pala. "
                f"Sensación práctica: al impactar piensa en 'estirar' el brazo hasta arriba como si quisieras tocar el techo, no en empujar la pala con el codo doblado.")
        else:
            add(abs(d) * 1.2, "mid",
                "Codo más bloqueado que el del pro",
                f"Tu codo está {abs(d):.0f}° más estirado que el del pro en el impacto. Bloquear completamente el codo elimina el efecto látigo de la muñeca al final del swing. "
                f"Deja un poco de flexión para liberar la muñeca en el último instante.")

    # --- Extensión del codo durante TODO el swing (media de la diferencia absoluta) ---
    el_mean = el.get("mean_abs_delta")
    if el_mean is not None and el_mean > 12:
        add(el_mean * 0.7, "mid",
            "El pro extiende y flexiona el brazo de forma muy distinta a la tuya",
            f"Durante el swing entero, tu ángulo de codo difiere de media {el_mean:.0f}° respecto al del pro. "
            f"Esto indica que la dinámica del brazo (cuándo lo armas, cuándo lo extiendes, cuándo lo sueltas) "
            f"no sigue el mismo patrón que el del pro. Observa el overlay frame a frame: probablemente él extiende "
            f"el brazo más arriba y más recto durante la subida al impacto, mientras tú mantienes el codo más doblado.")

    # --- Pierna trasera (derecha si diestro) ---
    rk = per_metric.get("right_knee_angle_deg", {})
    d = rk.get("delta_at_impact")
    if d is not None:
        if d < -20:
            add(abs(d) * 1.0, "bad",
                "Extiende más la pierna trasera al impactar",
                f"Tu rodilla derecha está {abs(d):.0f}° más flexionada que la del pro en el impacto. "
                f"La pierna trasera casi recta = transferencia completa del peso del cuerpo al golpe = más potencia. "
                f"Empuja con el pie trasero al subir, casi como si fueras a saltar.")
        elif d > 20:
            add(abs(d) * 0.8, "mid",
                "Pierna trasera demasiado rígida",
                f"Tu rodilla derecha está {abs(d):.0f}° más extendida que la del pro. "
                f"Una pierna bloqueada pierde elasticidad — necesita algo de flexión en la carga para luego explotar.")

    # --- Pierna delantera ---
    lk = per_metric.get("left_knee_angle_deg", {})
    d = lk.get("delta_at_impact")
    if d is not None:
        if d < -15:
            add(abs(d) * 0.8, "mid",
                "Pierna delantera muy flexionada al impactar",
                f"Tu rodilla izquierda está {abs(d):.0f}° más flexionada que la del pro al contactar. "
                f"Idealmente en el impacto el cuerpo está casi totalmente extendido para máxima altura — extiende también esta pierna al subir.")
        elif d > 15:
            add(abs(d) * 0.7, "mid",
                "Pierna delantera bloqueada",
                f"Tu rodilla izquierda está {abs(d):.0f}° más extendida que la del pro. Mantén un poco de flexión para absorber el impacto y caer bien.")

    # --- Extensión del cuerpo cadera-hombro ---
    hs = per_metric.get("hip_shoulder_separation_m", {})
    d = hs.get("delta_at_impact")
    if d is not None:
        d_cm = d * 100
        if d_cm < -8:
            add(abs(d_cm) * 2, "mid",
                "Te falta extensión total del cuerpo en el impacto",
                f"La distancia vertical entre cadera y hombros es {abs(d_cm):.0f} cm menor que la del pro en el impacto. "
                f"Eso indica que no te estiras del todo — pruebas a 'crecer' con todo el cuerpo en el momento del contacto (rodillas, cadera, espalda, hombros). "
                f"Más extensión = más altura = mejor ángulo del golpe.")

    # --- Inclinación del tronco ---
    tr = per_metric.get("trunk_lean_deg", {})
    d = tr.get("delta_at_impact")
    if d is not None and abs(d) > 10:
        if d < 0:
            add(abs(d) * 0.6, "mid",
                "Tronco demasiado erguido al impactar",
                f"Tu tronco está {abs(d):.0f}° más vertical que el del pro. Inclinarte ligeramente hacia adelante al impactar transfiere mejor el peso a la bola y proyecta el cuerpo en la dirección del golpe.")
        else:
            add(abs(d) * 0.6, "mid",
                "Tronco demasiado inclinado al impactar",
                f"Tu tronco está {abs(d):.0f}° más inclinado hacia adelante que el del pro. Demasiada inclinación = pérdida de equilibrio. Busca el justo equilibrio para no quedarte 'pinchado' en el sitio.")

    # --- Base de pies ---
    feet = per_metric.get("feet_separation_m", {})
    d = feet.get("delta_at_impact")
    if d is not None:
        d_cm = d * 100
        if d_cm < -10:
            add(abs(d_cm) * 0.5, "mid",
                "Abre más la base de los pies",
                f"Tus pies están {abs(d_cm):.0f} cm más juntos que los del pro al impactar. "
                f"Una base más ancha te da estabilidad para rotar el tronco con fuerza sin perder el equilibrio.")
        elif d_cm > 15:
            add(abs(d_cm) * 0.4, "mid",
                "Base demasiado ancha",
                f"Tus pies están {abs(d_cm):.0f} cm más separados que los del pro. Excesiva apertura reduce la explosividad del salto/extensión — busca una postura más compacta.")

    # --- Ángulo del mango de la pala en el impacto ---
    if racket_angle and racket_angle.get("delta") is not None:
        d = racket_angle["delta"]
        if abs(d) > 15:
            sev = "mid" if abs(d) < 30 else "bad"
            if d < 0:
                add(abs(d) * 0.6, sev,
                    "Pala demasiado vertical en el impacto",
                    f"El mango de tu pala apunta {abs(d):.0f}° más vertical que el del pro. "
                    f"Una pala más vertical = bola más rápida hacia el suelo pero con menos margen de error. "
                    f"Si fallas mucho, inclínala ligeramente.")
            else:
                add(abs(d) * 0.6, sev,
                    "Pala demasiado horizontal en el impacto",
                    f"El mango de tu pala apunta {abs(d):.0f}° más horizontal que el del pro. "
                    f"Una pala más plana = más control y efecto pero menos pico de velocidad descendente. "
                    f"Verticaliza un poco si buscas un smash más definitivo.")

    # ====================================================================
    # PREPARATION
    # ====================================================================
    if prep_drop:
        delta_cm = prep_drop["delta_cm"]
        if delta_cm < -10:
            add(abs(delta_cm) * 0.6, "bad",
                "Aumenta el recorrido de la preparación",
                f"Tu punta de pala baja {abs(delta_cm):.0f} cm MENOS que la del pro durante la preparación. "
                f"Más recorrido = más espacio para acelerar = más potencia en el impacto. "
                f"Trabaja en armar la pala más atrás y abajo antes de subir.")
        elif delta_cm > 15:
            add(abs(delta_cm) * 0.5, "mid",
                "Preparación posiblemente excesiva",
                f"Tu punta baja {abs(delta_cm):.0f} cm MÁS que la del pro. "
                f"Una preparación muy larga puede hacerte perder tempo y precisión. "
                f"Más corta y compacta suele ser más eficaz en pádel (la pista no es tan grande).")

    advice.sort(key=lambda x: -x["priority"])
    return advice[:8]


def asset(name):
    """Relative path from web/index.html to a bundled asset."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".mp4":
        return f"videos/{name}"
    if ext == ".png":
        return f"plots/{name}"
    return f"data/{name}"


def delta_class(delta, tol):
    mag = abs(delta) / max(tol, 1e-9)
    if mag < 1.0:
        return "good"
    if mag < 2.0:
        return "mid"
    return "bad"


def fmt(v, unit):
    if v is None:
        return "—"
    if unit == "m":
        return f"{v:+.2f} m"
    if unit == "m/s":
        return f"{v:+.2f} m/s"
    return f"{v:+.1f} deg"


def render_metric_row(key, data):
    label, unit, tol = METRIC_LABELS.get(key, (key, "", 1.0))
    d_imp = data.get("delta_at_impact")
    d_mean = data.get("mean_abs_delta")
    d_max = data.get("max_abs_delta")
    cls = delta_class(d_imp or 0, tol)
    return f"""
        <tr class="metric-row {cls}">
            <td class="label">{escape(label)}</td>
            <td class="num">{fmt(d_imp, unit)}</td>
            <td class="num">{abs(d_mean or 0):.2f} {unit}</td>
            <td class="num">{abs(d_max or 0):.2f} {unit}</td>
        </tr>
    """


def main():
    print("bundling assets (re-encoding videos to H.264)...")
    bundle_assets()
    print("building interactive skeleton-steps JSON...")
    steps_json_path = os.path.join(WEB_DIR, "data", "skeleton_steps.json")
    build_skeleton_steps_json(steps_json_path)
    with open(steps_json_path) as f:
        skeleton_steps_json = f.read()

    # Coaching text: read prep_drop and racket_angle from the just-built data,
    # plus per_metric from comparison.json.
    steps_data = json.loads(skeleton_steps_json)
    with open(os.path.join(OUTPUT_DIR, "comparison.json")) as f:
        comparison_data = json.load(f)
    coaching = generate_coaching(
        comparison_data.get("per_metric", {}),
        steps_data.get("prep_drop"),
        steps_data.get("racket_angle"),
    )

    with open(os.path.join(OUTPUT_DIR, "comparison.json")) as f:
        cmp = json.load(f)

    pro_phases_json = os.path.join(OUTPUT_DIR, f"{PRO_BASE}_smash.json")
    user_phases_json = os.path.join(OUTPUT_DIR, f"{USER_BASE}_smash.json")
    # (filename pattern in output is base + "_smash.json" since detect_smash strips "_pose")
    pro_phases = json.load(open(pro_phases_json)) if os.path.exists(pro_phases_json) else {}
    user_phases = json.load(open(user_phases_json)) if os.path.exists(user_phases_json) else {}

    biggest = cmp.get("biggest_diffs", [])
    per = cmp.get("per_metric", {})

    if coaching:
        coaching_items = "\n".join(
            f'<div class="advice {escape(a["severity"])}">'
            f'<div class="advice-title">{escape(a["title"])}</div>'
            f'<div class="advice-body">{escape(a["body"])}</div>'
            f'</div>'
            for a in coaching
        )
    else:
        coaching_items = (
            '<div class="advice good">'
            '<div class="advice-title">Mecánica similar a la del pro</div>'
            '<div class="advice-body">Todas las métricas analizadas están dentro de la tolerancia respecto al pro. ¡Sigue así!</div>'
            '</div>'
        )

    metric_rows = "\n".join(
        render_metric_row(k, per[k]) for k in biggest if k in per
    )
    other_keys = [k for k in per if k not in biggest]
    metric_rows_other = "\n".join(
        render_metric_row(k, per[k]) for k in other_keys
    )

    pro_dom = pro_phases.get("dominant", "?")
    user_dom = user_phases.get("dominant", "?")
    fps = pro_phases.get("fps", 25)
    pro_dur = (pro_phases.get("end", 0) - pro_phases.get("start", 0)) / fps
    user_dur = (user_phases.get("end", 0) - user_phases.get("start", 0)) / fps

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>PadelCoach — Comparación de smash</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;800;900&display=swap" rel="stylesheet">
<style>
:root {{
    --cream: #EEE7DA;
    --red: #99040D;
    --red-bright: #d10518;
    --dark: #2F0103;
    --darker: #1a0001;
    --bg: var(--dark);
    --panel: rgba(238, 231, 218, 0.04);
    --panel-2: rgba(238, 231, 218, 0.07);
    --border: rgba(238, 231, 218, 0.12);
    --text: var(--cream);
    --muted: rgba(238, 231, 218, 0.55);
    --pro: #6CC8FF;
    --user: #ff5e5e;
    --good: #2ecc71;
    --mid: #f1c40f;
    --bad: #ff5e5e;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0; padding: 0;
    background:
        radial-gradient(ellipse at 70% -10%, rgba(153,4,13,0.35) 0%, transparent 50%),
        radial-gradient(ellipse at 10% 110%, rgba(47,1,3,0.85) 0%, transparent 60%),
        linear-gradient(180deg, var(--darker) 0%, var(--dark) 100%);
    background-attachment: fixed;
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    font-weight: 400;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}}
body::before {{
    /* film grain */
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 100;
    background-image: radial-gradient(rgba(238,231,218,0.06) 1px, transparent 1px);
    background-size: 3px 3px; opacity: 0.5; mix-blend-mode: overlay;
}}
header {{
    padding: 28px 48px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(10px);
    background: rgba(26, 0, 1, 0.55);
    position: sticky; top: 0; z-index: 50;
}}
header .brand {{
    font-size: 18px; font-weight: 800; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--cream);
}}
header .brand span {{ color: var(--red-bright); }}
header .sub {{
    color: var(--muted); font-size: 12px; letter-spacing: 0.18em;
    text-transform: uppercase; font-weight: 500;
    display: none;  /* hide on small screens, shown on wide */
}}
@media (min-width: 1100px) {{ header .sub {{ display: inline; }} }}
header a {{
    color: var(--cream); text-decoration: none;
    font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
    font-weight: 700;
    padding: 10px 20px; border-radius: 999px;
    border: 1px solid var(--border);
    transition: background .2s, border-color .2s;
}}
header a:hover {{ background: var(--panel-2); border-color: var(--cream); }}
main {{ padding: 48px 48px 120px; max-width: 1400px; margin: 0 auto; }}
section {{ margin-bottom: 72px; }}
section h2 {{
    font-size: clamp(24px, 3vw, 36px);
    font-weight: 900;
    margin: 0 0 24px;
    text-transform: uppercase;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--cream);
}}
section h2 .accent {{ color: var(--red-bright); font-style: italic; }}
.panel {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden;
    backdrop-filter: blur(4px);
}}
.panel-pad {{ padding: 16px 20px; }}
.video-block video {{ display: block; width: 100%; height: auto; background: #000; }}
.grid-2 {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}}
.legend {{
    display: flex; gap: 18px; font-size: 13px; color: var(--muted);
    padding: 10px 14px; border-top: 1px solid var(--border); background: var(--panel-2);
}}
.legend .dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}}
.dot.pro {{ background: var(--pro); }}
.dot.user {{ background: var(--user); }}
.player-card .title {{
    padding: 12px 16px; font-weight: 600; font-size: 14px;
    border-bottom: 1px solid var(--border); display: flex; align-items: center;
    justify-content: space-between;
}}
.player-card.pro .title {{ color: var(--pro); }}
.player-card.user .title {{ color: var(--user); }}
.player-card .meta {{
    color: var(--muted); font-size: 12px; font-weight: 400;
}}
.metrics-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.metrics-table th {{
    text-align: left; padding: 10px 14px; font-weight: 500; color: var(--muted);
    border-bottom: 1px solid var(--border); font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.6px;
}}
.metrics-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
.metrics-table tr:last-child td {{ border-bottom: none; }}
.metric-row .label {{ font-weight: 500; }}
.metric-row .num {{ font-variant-numeric: tabular-nums; color: var(--muted); }}
.metric-row.good .num:first-of-type {{ color: var(--good); font-weight: 600; }}
.metric-row.mid  .num:first-of-type {{ color: var(--mid);  font-weight: 600; }}
.metric-row.bad  .num:first-of-type {{ color: var(--bad);  font-weight: 600; }}
.assets {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; margin-top: 12px;
}}
.asset {{
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; font-size: 13px;
    transition: border-color .2s, background .2s;
}}
.asset:hover {{ border-color: var(--red-bright); background: rgba(153,4,13,0.08); }}
.asset a {{ color: var(--cream); text-decoration: none; font-weight: 700; }}
.asset a:hover {{ color: var(--red-bright); }}
.asset .desc {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
img.plot {{ display: block; width: 100%; height: auto; background: #fff; border-radius: 0; }}
.divider-hint {{
    text-align: center; color: var(--muted); font-size: 12px;
    padding: 8px; border-top: 1px solid var(--border);
}}
@media (max-width: 900px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
}}
.coaching-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px;
}}
.advice {{
    background: var(--panel); border: 1px solid var(--border);
    border-left: 4px solid var(--mid); border-radius: 12px;
    padding: 18px 22px;
    transition: border-color .25s, background .25s, transform .25s;
}}
.advice:hover {{ transform: translateY(-2px); background: var(--panel-2); }}
.advice.bad {{ border-left-color: var(--bad); }}
.advice.mid {{ border-left-color: var(--mid); }}
.advice.good {{ border-left-color: var(--good); }}
.advice-title {{ font-weight: 800; font-size: 16px; margin-bottom: 8px; }}
.advice-body {{ color: var(--muted); font-size: 14px; line-height: 1.55; }}
.scrub-panel {{ padding: 0; }}
.scrub-panel canvas {{
    display: block; width: 100%; max-width: 560px; height: auto;
    margin: 0 auto; background: #14181f;
    border-bottom: 1px solid var(--border);
}}
.scrub-controls {{
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: var(--panel-2);
    border-bottom: 1px solid var(--border);
}}
.scrub-controls input[type=range] {{ flex: 1; cursor: pointer; }}
.scrub-controls button {{
    background: var(--panel-2); color: var(--cream);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 8px 14px; cursor: pointer;
    font-family: inherit; font-size: 12px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    transition: background .2s, border-color .2s, color .2s;
}}
.scrub-controls button:hover {{
    background: var(--red); color: var(--cream);
    border-color: var(--red);
}}
.scrub-readout {{
    font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px;
    min-width: 280px;
}}
</style>
</head>
<body>
<header>
    <div class="brand">PADEL<span>COACH</span></div>
    <span class="sub">Comparación biomecánica · DTW anclada al impacto</span>
    <a href="/">← Volver</a>
</header>
<main>

<section style="margin-bottom: 56px;">
    <h2>Tu smash <span class="accent">vs</span> el del pro</h2>
    <p style="color: var(--muted); font-size: 15px; max-width: 720px; margin: 0;">
        Esqueletos sincronizados frame a frame, métricas calibradas en cm y consejos accionables a partir de los deltas.
    </p>
</section>

<section>
    <h2>Consejos para mejorar tu smash</h2>
    <div id="coaching" class="coaching-grid">
        {coaching_items}
    </div>
</section>

<section>
    <h2>Bajada de la punta de pala en la preparación</h2>
    <div class="panel panel-pad" id="prep-drop-panel">
        <div style="font-size: 13px; color: var(--muted); margin-bottom: 10px;">
            Distancia máxima a la que cae la punta de la pala por debajo de la muñeca durante la fase de preparación (antes del impacto). Calibrado en cm usando la longitud real de la pala (45.5 cm).
        </div>
        <div id="prep-drop-readout" style="font-size: 16px; line-height: 1.8;">
            <span style="color: var(--muted);">cargando…</span>
        </div>
    </div>
</section>

<section>
    <h2>Overlay interactivo (scroll con slider)</h2>
    <div class="panel scrub-panel">
        <canvas id="scrub-canvas" width="900" height="900"></canvas>
        <div class="scrub-controls">
            <button id="scrub-play" type="button">▶</button>
            <input id="scrub-slider" type="range" min="0" max="0" value="0" step="1">
            <span class="scrub-readout">
                step <span id="scrub-step">0</span>/<span id="scrub-total">0</span>
                · pro f<span id="scrub-pro-frame">—</span>
                · user f<span id="scrub-user-frame">—</span>
                <span id="scrub-impact-tag" style="display:none; color: var(--bad); font-weight:600; margin-left:8px;">IMPACTO</span>
            </span>
            <button id="scrub-impact-btn" type="button" title="ir al frame del impacto">→ impacto</button>
        </div>
        <div class="scrub-controls offset-controls">
            <label style="color: var(--pro); min-width: 70px; font-size: 13px;">PRO X</label>
            <input id="pro-offset" type="range" min="-300" max="300" value="0" step="1">
            <span id="pro-offset-val" class="scrub-readout" style="min-width: 60px;">0 px</span>
            <button id="reset-offsets" type="button" title="centrar ambos">⌖ reset</button>
        </div>
        <div class="scrub-controls offset-controls">
            <label style="color: var(--user); min-width: 70px; font-size: 13px;">USER X</label>
            <input id="user-offset" type="range" min="-300" max="300" value="0" step="1">
            <span id="user-offset-val" class="scrub-readout" style="min-width: 60px;">0 px</span>
        </div>
        <div class="legend">
            <span><span class="dot pro"></span>PRO</span>
            <span><span class="dot user"></span>USER</span>
            <span style="margin-left:auto;">Mueve el slider para ver cualquier frame del swing — esqueletos alineados por cadera + pala bbox · sin reproducción de vídeo</span>
        </div>
    </div>
</section>

<section>
    <h2>Overlay normalizado (cámara lenta 8×)</h2>
    <div class="panel video-block">
        <video src="{asset('comparison_overlay.mp4')}" controls preload="metadata" loop></video>
        <div class="legend">
            <span><span class="dot pro"></span>PRO ({escape(pro_dom)})</span>
            <span><span class="dot user"></span>USER ({escape(user_dom)})</span>
            <span style="margin-left: auto;">Esqueletos centrados en la cadera y escalados al mismo torso · pala como bbox YOLO anclado a la muñeca · freeze de 1 s en el impacto</span>
        </div>
    </div>
</section>

<section>
    <h2>Vídeos originales sincronizados</h2>
    <div class="panel video-block">
        <video src="{asset('comparison_sxs.mp4')}" controls preload="metadata" loop></video>
        <div class="legend">
            <span>Frames emparejados por DTW · métricas por frame con delta vs PRO color-codeado</span>
        </div>
    </div>
</section>

<section>
    <h2>Mayores diferencias biomecánicas</h2>
    <div class="panel">
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Métrica</th>
                    <th>Δ en impacto</th>
                    <th>Δ medio absoluto</th>
                    <th>Δ máximo absoluto</th>
                </tr>
            </thead>
            <tbody>
                {metric_rows}
            </tbody>
        </table>
    </div>
    <details style="margin-top: 16px;">
        <summary style="cursor: pointer; color: var(--muted); font-size: 13px;">Resto de métricas</summary>
        <div class="panel" style="margin-top: 10px;">
            <table class="metrics-table">
                <tbody>{metric_rows_other}</tbody>
            </table>
        </div>
    </details>
</section>

<section>
    <h2>Desglose por jugador</h2>
    <div class="grid-2">
        <div class="panel player-card pro">
            <div class="title">PRO — {escape(PRO_BASE)} <span class="meta">{pro_dur:.2f} s · {pro_dom}</span></div>
            <video src="{asset(PRO_BASE + '_smash_marked.mp4')}" controls preload="metadata" loop></video>
            <div class="legend"><span>Esqueleto verde + pala (bbox amarillo) + barra de fase (gris PRE → amarillo SWING → rojo IMPACT → verde FOLLOW)</span></div>
            <img class="plot" src="{asset(PRO_BASE + '_metrics.png')}" alt="Pro metrics">
        </div>
        <div class="panel player-card user">
            <div class="title">USER — {escape(USER_BASE)} <span class="meta">{user_dur:.2f} s · {user_dom}</span></div>
            <video src="{asset(USER_BASE + '_smash_marked.mp4')}" controls preload="metadata" loop></video>
            <div class="legend"><span>Idem para el clip del usuario</span></div>
            <img class="plot" src="{asset(USER_BASE + '_metrics.png')}" alt="User metrics">
        </div>
    </div>
</section>

<section>
    <h2>Curvas alineadas DTW (todas las métricas)</h2>
    <div class="panel panel-pad">
        <img class="plot" src="{asset('comparison.png')}" alt="DTW-aligned metric curves">
    </div>
</section>

</main>

<script id="skeleton-data" type="application/json">{skeleton_steps_json}</script>
<script>
(() => {{
    const data = JSON.parse(document.getElementById('skeleton-data').textContent);
    const canvas = document.getElementById('scrub-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = data.canvas_size;
    canvas.height = data.canvas_size;

    const slider = document.getElementById('scrub-slider');
    const stepReadout = document.getElementById('scrub-step');
    const totalReadout = document.getElementById('scrub-total');
    const proFrame = document.getElementById('scrub-pro-frame');
    const userFrame = document.getElementById('scrub-user-frame');
    const impactTag = document.getElementById('scrub-impact-tag');
    const impactBtn = document.getElementById('scrub-impact-btn');
    const playBtn = document.getElementById('scrub-play');
    const proOffsetSlider = document.getElementById('pro-offset');
    const userOffsetSlider = document.getElementById('user-offset');
    const proOffsetVal = document.getElementById('pro-offset-val');
    const userOffsetVal = document.getElementById('user-offset-val');
    const resetBtn = document.getElementById('reset-offsets');

    slider.max = data.steps.length - 1;
    totalReadout.textContent = data.steps.length - 1;

    // Fill the "lowest racket tip in preparation" panel.
    const prepEl = document.getElementById('prep-drop-readout');
    if (data.prep_drop) {{
        const p = data.prep_drop;
        prepEl.innerHTML =
            `<div><span style="color: var(--pro);">●</span> PRO  punta baja <b>${{p.pro_cm.toFixed(1)}} cm</b> por debajo de la muñeca</div>` +
            `<div><span style="color: var(--user);">●</span> USER punta baja <b>${{p.user_cm.toFixed(1)}} cm</b> por debajo de la muñeca</div>` +
            `<div style="margin-top: 6px;">Delta: <b style="color: ${{p.color}}; font-size: 18px;">${{p.delta_cm >= 0 ? '+' : ''}}${{p.delta_cm.toFixed(1)}} cm</b> ` +
            `<span style="color: var(--muted); font-size: 13px;">(positivo = tu punta baja MÁS que la del pro · negativo = baja MENOS)</span></div>`;
    }} else {{
        prepEl.textContent = 'No se pudo calcular (faltan detecciones de pala en la fase de preparación).';
    }}

    // Pre-load racket images so they paint instantly when scrubbing to the impact step.
    const proRacketImg = data.pro_racket_img ? new Image() : null;
    const userRacketImg = data.user_racket_img ? new Image() : null;
    let racketsReady = 0;
    const needed = (proRacketImg ? 1 : 0) + (userRacketImg ? 1 : 0);
    function maybeRender() {{ racketsReady++; if (racketsReady === needed) render(parseInt(slider.value)); }}
    if (proRacketImg) {{ proRacketImg.onload = maybeRender; proRacketImg.src = data.pro_racket_img; }}
    if (userRacketImg) {{ userRacketImg.onload = maybeRender; userRacketImg.src = data.user_racket_img; }}

    function drawSkeleton(pose, color) {{
        if (!pose) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        for (const [a, b] of data.edges) {{
            ctx.beginPath();
            ctx.moveTo(pose[a][0], pose[a][1]);
            ctx.lineTo(pose[b][0], pose[b][1]);
            ctx.stroke();
        }}
        ctx.fillStyle = color;
        for (const j of data.joints) {{
            ctx.beginPath();
            ctx.arc(pose[j][0], pose[j][1], 4, 0, Math.PI * 2);
            ctx.fill();
        }}
    }}

    function drawRacket(box, color) {{
        if (!box) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    }}


    function render(step) {{
        ctx.fillStyle = '#14181f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // light grid
        ctx.strokeStyle = '#1e242d'; ctx.lineWidth = 1;
        for (let k = 0; k < canvas.width; k += 60) {{
            ctx.beginPath(); ctx.moveTo(k, 0); ctx.lineTo(k, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, k); ctx.lineTo(canvas.width, k); ctx.stroke();
        }}
        // hip anchor dot
        ctx.fillStyle = '#444'; ctx.beginPath();
        ctx.arc(data.hip_center[0], data.hip_center[1], 4, 0, Math.PI*2); ctx.fill();

        const s = data.steps[step];
        const isImpact = (parseInt(step) === data.impact_step);
        const proDx = parseInt(proOffsetSlider.value);
        const userDx = parseInt(userOffsetSlider.value);

        // ---- PRO (with horizontal offset) ----
        ctx.save();
        ctx.translate(proDx, 0);
        drawSkeleton(s.pro, data.pro_color);
        drawRacket(s.pro_racket, data.pro_color);
        if (isImpact) {{
            if (s.pro_racket && proRacketImg && proRacketImg.complete) {{
                const [x1, y1, x2, y2] = s.pro_racket;
                ctx.drawImage(proRacketImg, x1, y1, x2 - x1, y2 - y1);
            }}
            if (s.pro) {{
                const w = s.pro[data.pro_wrist_lm];
                ctx.strokeStyle = data.pro_color; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(w[0], w[1]); ctx.lineTo(w[0], canvas.height); ctx.stroke();
            }}
        }}
        ctx.restore();

        // ---- USER (with horizontal offset) ----
        ctx.save();
        ctx.translate(userDx, 0);
        drawSkeleton(s.user, data.user_color);
        drawRacket(s.user_racket, data.user_color);
        if (isImpact) {{
            if (s.user_racket && userRacketImg && userRacketImg.complete) {{
                const [x1, y1, x2, y2] = s.user_racket;
                ctx.drawImage(userRacketImg, x1, y1, x2 - x1, y2 - y1);
            }}
            if (s.user) {{
                const w = s.user[data.user_wrist_lm];
                ctx.strokeStyle = data.user_color; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(w[0], w[1]); ctx.lineTo(w[0], canvas.height); ctx.stroke();
            }}
            if (s.user && data.impact_badges) {{
                ctx.font = 'bold 18px system-ui';
                for (const b of data.impact_badges) {{
                    const p = s.user[b.lm];
                    ctx.strokeStyle = b.color; ctx.lineWidth = 3;
                    ctx.beginPath(); ctx.arc(p[0], p[1], 14, 0, Math.PI*2); ctx.stroke();
                    const txt = (b.delta >= 0 ? '+' : '') + b.delta.toFixed(0) + ' deg';
                    ctx.strokeStyle = '#000'; ctx.lineWidth = 4;
                    ctx.strokeText(txt, p[0] + 18, p[1] + 6);
                    ctx.fillStyle = b.color; ctx.fillText(txt, p[0] + 18, p[1] + 6);
                }}
            }}
        }}
        ctx.restore();

        // ---- Fixed UI elements (not affected by offsets) ----
        if (isImpact) {{
            // Big IMPACTO label top-right.
            ctx.font = 'bold 28px system-ui';
            ctx.fillStyle = '#ff5e5e';
            ctx.strokeStyle = '#000'; ctx.lineWidth = 5;
            const lbl = 'IMPACTO';
            const w = ctx.measureText(lbl).width;
            ctx.strokeText(lbl, canvas.width - w - 24, 40);
            ctx.fillText(lbl, canvas.width - w - 24, 40);

            // Racket angle panel (top-left) — handle direction from vertical.
            if (data.racket_angle && data.racket_angle.pro !== null) {{
                const ra = data.racket_angle;
                const lines = [
                    ['Angulo pala (desde vertical)', '#cccccc'],
                    ['PRO  ' + (ra.pro >= 0 ? '+' : '') + ra.pro.toFixed(0) + ' deg', data.pro_color],
                    ['USER ' + (ra.user >= 0 ? '+' : '') + ra.user.toFixed(0) + ' deg', data.user_color],
                    ['Delta ' + (ra.delta >= 0 ? '+' : '') + ra.delta.toFixed(1) + ' deg', ra.color],
                ];
                ctx.font = 'bold 16px system-ui';
                let y = 36;
                for (const [text, col] of lines) {{
                    ctx.strokeStyle = '#000'; ctx.lineWidth = 4;
                    ctx.strokeText(text, 16, y);
                    ctx.fillStyle = col;
                    ctx.fillText(text, 16, y);
                    y += 24;
                }}
            }}
        }}

        stepReadout.textContent = step;
        proFrame.textContent = s.pro_frame ?? '—';
        userFrame.textContent = s.user_frame ?? '—';
        impactTag.style.display = (parseInt(step) === data.impact_step) ? 'inline' : 'none';
    }}

    function rerender() {{ render(parseInt(slider.value)); }}
    slider.addEventListener('input', rerender);
    impactBtn.addEventListener('click', () => {{
        slider.value = data.impact_step; rerender();
    }});
    proOffsetSlider.addEventListener('input', () => {{
        proOffsetVal.textContent = proOffsetSlider.value + ' px'; rerender();
    }});
    userOffsetSlider.addEventListener('input', () => {{
        userOffsetVal.textContent = userOffsetSlider.value + ' px'; rerender();
    }});
    resetBtn.addEventListener('click', () => {{
        proOffsetSlider.value = 0; userOffsetSlider.value = 0;
        proOffsetVal.textContent = '0 px'; userOffsetVal.textContent = '0 px';
        rerender();
    }});

    let playing = false, rafId = null;
    function tick() {{
        if (!playing) return;
        let v = parseInt(slider.value) + 1;
        if (v >= data.steps.length) v = 0;
        slider.value = v; render(v);
        rafId = setTimeout(() => requestAnimationFrame(tick), 40);  // ~25 fps
    }}
    playBtn.addEventListener('click', () => {{
        playing = !playing;
        playBtn.textContent = playing ? '⏸' : '▶';
        if (playing) tick();
        else if (rafId) clearTimeout(rafId);
    }});

    render(0);
}})();
</script>
</body>
</html>
"""
    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"web -> {INDEX_HTML}")
    print(f"open with: start {INDEX_HTML}")


if __name__ == "__main__":
    main()
