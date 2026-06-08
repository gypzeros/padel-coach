import cv2
import os

VIDEO = r"C:\claude\padel-coach\videos\user\user_smash.webm"
OUT_DIR = r"C:\claude\padel-coach\output"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps}  total_frames={total}  duration={total/fps:.2f}s")

# Sample 10 evenly spaced frames so we can see the swing trajectory.
sample_indices = [int(i * total / 10) for i in range(10)]
print("sampling at frames:", sample_indices)

i = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i in sample_indices:
        path = os.path.join(OUT_DIR, f"thumb_f{i:03d}_t{i/fps:.1f}s.jpg")
        cv2.imwrite(path, frame)
        print("  saved", path)
        saved += 1
    i += 1
cap.release()
print(f"done, {saved} thumbs in {OUT_DIR}")
