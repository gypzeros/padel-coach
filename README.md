# Padel Coach

Análisis biomecánico automático del smash de pádel. Sube tu vídeo, compáralo frame a frame con un jugador profesional, recibe consejos accionables a partir de las diferencias detectadas.

Demo web → [`web/index.html`](web/index.html) · landing → [`web/landing.html`](web/landing.html)

## Lo que hace

1. **Detección de pose** multi-persona con MediaPipe Pose Landmarker (heavy, 33 landmarks 2D + 3D)
2. **Tracking estable** de personas entre frames + auto-selección del smasher por mayor varianza de muñeca
3. **Detección automática del swing** (start → apex/impact → end) basada en la trayectoria de la muñeca dominante
4. **Detección de la pala** con YOLO11 (clase tennis racket) + pareo con la muñeca del smasher
5. **Métricas biomecánicas** por frame: ángulos de codo/rodillas, velocidad 3D de muñeca, separación cadera-hombro, separación de pies, inclinación del tronco
6. **Alineación temporal DTW** anclada al impacto entre los dos clips
7. **Comparación cuantitativa** frame a frame con calibración pixel→cm usando la longitud real de la pala (45.5 cm)
8. **Coaching automático**: el sistema convierte los deltas numéricos en consejos accionables en castellano
9. **Visualización**: overlay de esqueletos a cámara lenta, scrubber interactivo con sliders de separación horizontal, vídeo side-by-side, gráficas matplotlib

## Estructura

```
padel-coach/
├── scripts/                       # pipeline en Python
│   ├── pose.py                    # MediaPipe → CSV de landmarks
│   ├── racket.py                  # YOLO11 → CSV de detecciones de pala
│   ├── detect_smash.py            # localiza start/apex/impact/end + marked.mp4
│   ├── metrics.py                 # ángulos, velocidades, ratios → CSV/JSON/PNG
│   ├── compare.py                 # DTW anchored + overlays + sxs video
│   ├── build_web.py               # genera web/index.html (dashboard)
│   └── build_landing.py           # genera web/landing.html embebiendo el GLB
├── videos/
│   ├── pro/                       # clip de referencia del profesional
│   └── user/                      # clip del usuario
├── output/                        # CSVs, JSONs, PNGs y MP4s generados
├── models/
│   └── smash.glb                  # mesh 3D para la landing (generado con Meshy.ai)
├── web/                           # frontend
│   ├── index.html                 # dashboard de comparación (auto-generado)
│   ├── landing.html               # landing page con 3D rotativo (auto-generado)
│   ├── landing_template.html      # template antes de embeber el .glb
│   └── videos/, plots/, data/     # assets bundleados
└── README.md
```

## Pipeline de uso

Asumiendo que tienes:
- `videos/pro/garrido_smash.webm` — clip de referencia (smash de un pro, recortado a 3-5 s)
- `videos/user/user_smash.webm` — clip del usuario en las mismas condiciones de ángulo

Dependencias: Python 3.11, PyTorch+CUDA, ultralytics, mediapipe, opencv-python, scipy, pandas, dtaidistance, matplotlib, imageio-ffmpeg, yt-dlp (para descargar segmentos de YouTube).

```powershell
$py = "C:\path\to\python.exe"
$root = "C:\claude\padel-coach"

# 1. Pose extraction per clip
& $py $root\scripts\pose.py $root\videos\pro\garrido_smash.webm
& $py $root\scripts\pose.py $root\videos\user\user_smash.webm

# 2. Racket detection per clip
& $py $root\scripts\racket.py $root\videos\pro\garrido_smash.webm
& $py $root\scripts\racket.py $root\videos\user\user_smash.webm

# 3. Smash window detection + marked video
& $py $root\scripts\detect_smash.py $root\output\garrido_smash_pose.csv --video $root\videos\pro\garrido_smash.webm
& $py $root\scripts\detect_smash.py $root\output\user_smash_pose.csv --video $root\videos\user\user_smash.webm

# 4. Biomechanical metrics
& $py $root\scripts\metrics.py $root\output\garrido_smash_pose.csv $root\output\garrido_smash_smash.json
& $py $root\scripts\metrics.py $root\output\user_smash_pose.csv $root\output\user_smash_smash.json

# 5. Comparison (DTW anchored at impact + sxs + overlay)
& $py $root\scripts\compare.py `
    --pro-metrics $root\output\garrido_smash_metrics.csv `
    --pro-smash   $root\output\garrido_smash_smash.json `
    --pro-video   $root\videos\pro\garrido_smash.webm `
    --user-metrics $root\output\user_smash_metrics.csv `
    --user-smash   $root\output\user_smash_smash.json `
    --user-video   $root\videos\user\user_smash.webm

# 6. Build web (dashboard + interactive scrubber data)
& $py $root\scripts\build_web.py

# 7. Build landing (embeds smash.glb as base64 data URI)
& $py $root\scripts\build_landing.py
```

Después abre `web/index.html` o `web/landing.html` con doble-click.

## Limitaciones conocidas

- **Cámara lateral** es el ángulo óptimo. Cámara trasera funciona pero hombros y lag cadera-hombro pierden precisión.
- **No detecta rebotes ni tracking de pelota** en este proyecto (es solo pose + pala). El proyecto hermano `padelvision` cubre eso.
- **Una sola pelota / un solo smasher** por clip.
- **El SMPL/3D mesh fitting** se probó (ROMP) pero da resultados pobres desde cámara trasera por la ambigüedad 2D→3D inherente.

## Créditos

- MediaPipe Pose Landmarker (Google) — pose 2D/3D
- YOLO11x (Ultralytics) — racket detection
- dtaidistance — DTW anchored alignment
- Three.js — landing 3D
- Modelo 3D del jugador en la landing → generado con [Meshy.ai](https://www.meshy.ai/)
