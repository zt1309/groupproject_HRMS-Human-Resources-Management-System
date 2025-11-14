import os
import cv2
import random
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ====== Configuration ======
VIDEO_DIR = "data/videos"  # directory containing 'real' and 'fake' subdirectories
OUTPUT_DIR = ("data/frames_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

interval = int(input("Enter frame interval:") or 5)
train_ratio, val_ratio = 0.7, 0.15
test_ratio = 0.15

# ====== List videos ======
videos = {"real": [], "fake": []}
for cls in ["real", "fake"]:
    p = Path(VIDEO_DIR) / cls
    for v in p.glob("*.mp4"):
        videos[cls].append(str(v))

# ====== Split videos into sets ======
def split_videos(video_list):
    random.shuffle(video_list)
    n = len(video_list)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)
    return video_list[:n_train], video_list[n_train : n_train + n_val], video_list[n_train + n_val :]

# ====== Extract frames from a video ======
def extract_frames(video_path, output_folder, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    count, saved = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            fname = f"{Path(video_path).stem}_f{count:06d}.jpg"
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
        count += 1
    cap.release()
    return saved

# ====== Extract frames and build labels.csv ======
rows = []
for cls in ["real", "fake"]:
    train_v, val_v, test_v = split_videos(videos[cls])
    mapping = {v: "train" for v in train_v}
    mapping.update({v: "val" for v in val_v})
    mapping.update({v: "test" for v in test_v})

    print(f"\nProcessing class '{cls}' ({len(videos[cls])} videos)...")
    for video_path in tqdm(videos[cls]):
        split = mapping[video_path]
        out_dir = Path(OUTPUT_DIR) / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = extract_frames(video_path, out_dir, interval)
        for f in frames:
            rel_path = os.path.relpath(f, OUTPUT_DIR)
            rows.append(
                {
                    "path": rel_path.replace("\\", "/"),
                    "label": 1 if cls == "real" else 0,
                    "split": split,
                    "class": cls,
                }
            )

# ====== Save labels.csv ======
df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
df.to_csv(csv_path, index=False)
print(f"Done. Extracted {len(df)} images — saved labels at: {csv_path}")
