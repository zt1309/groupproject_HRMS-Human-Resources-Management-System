import cv2
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

from src.detect_faces import detect_and_crop_faces
from src.extract_embeddings import get_embedding

# === Paths ===
CSV_PATH = "db/data_employee.csv"
DB_PATH = "db/employees.json"
IMG_SAVE_DIR = "data/employees"
AVATAR_NAME = "avatar.jpg"

# === Capture Configuration ===
CAPTURE_DURATION = 18
CAPTURE_INTERVAL = 0.4
MAX_SAMPLES = 30
STAGE_DURATION = 6


def load_csv():
    return pd.read_csv(CSV_PATH)


def _ensure_dirs(emp_id):
    save_dir = os.path.join(IMG_SAVE_DIR, str(emp_id))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)


def enroll_employee(emp_id: str):
    """Enroll a new employee by capturing multiple images, embeddings, and avatar."""

    df = load_csv()
    try:
        row = df[df["Employee ID"] == int(emp_id)].iloc[0]
    except Exception:
        print(f"[ERROR] Employee ID {emp_id} not found in CSV file.")
        return

    full_name, department, position = row["Full Name"], row["Department"], row["Position"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the camera.")
        return

    print(f"[INFO] Starting enrollment for {full_name} (ID {emp_id})")

    save_dir = _ensure_dirs(emp_id)
    embeddings = []
    saved = 0
    start_time = time.time()
    last_capture = 0.0

    stages = [
        ("Look straight at the camera", (0, 255, 0)),
        ("Slowly turn your head LEFT", (255, 255, 0)),
        ("Slowly turn your head RIGHT", (255, 0, 255))
    ]

    avatar_saved = False  # <– to store first clear face

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        elapsed = now - start_time

        if elapsed >= CAPTURE_DURATION or saved >= MAX_SAMPLES:
            break

        stage_idx = int(elapsed // STAGE_DURATION)
        stage_idx = min(stage_idx, len(stages) - 1)
        instruction, color = stages[stage_idx]

        # Overlay text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(overlay, f"Capturing ({saved}/{MAX_SAMPLES}) | {CAPTURE_DURATION - elapsed:.1f}s left",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, f"Instruction: {instruction}", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Enrollment", overlay)

        if now - last_capture >= CAPTURE_INTERVAL:

            faces = detect_and_crop_faces(frame)

            if faces:
                # pick largest face
                face = sorted(faces, key=lambda f: f.shape[0] * f.shape[1], reverse=True)[0]

                emb = get_embedding(face)
                if emb is not None:
                    embeddings.append(emb)

                # Save dataset sample
                filename = os.path.join(save_dir, f"{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
                cv2.imwrite(filename, face)

                # Save avatar if not saved yet (FIRST GOOD FACE)
                if not avatar_saved:
                    avatar_path = os.path.join(save_dir, AVATAR_NAME)
                    cv2.imwrite(avatar_path, face)
                    avatar_saved = True

                saved += 1
                last_capture = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Enrollment aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # === Save DB Entry ===
    if not embeddings:
        print("[WARN] No valid embeddings captured.")
        return

    mean_emb = (sum(embeddings) / len(embeddings)).tolist()

    db = _load_db()
    db[str(emp_id)] = {
        "name": full_name,
        "department": department,
        "position": position,
        "embedding": mean_emb,
        "avatar": f"{save_dir}/{AVATAR_NAME}"  # <— EXACTLY WHAT REALTIME USES
    }
    _save_db(db)

    print(f"[INFO] Enrollment completed for {full_name}.")
    print(f"[INFO] Avatar saved at: {save_dir}/{AVATAR_NAME}")
    print(f"[INFO] {saved} face samples stored.")
