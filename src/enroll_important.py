import cv2
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

from src.detect_faces import detect_and_crop_faces
from src.extract_embeddings import get_embedding


# === Data paths ===
CSV_PATH = "db/important_employee.csv"          # VIP employee list (CSV)
DB_PATH = "db/important_employees.json"         # JSON DB storing VIP embeddings
IMG_SAVE_DIR = "data/employees_important"       # directory to save face images


# === Capture configuration ===
CAPTURE_DURATION = 18        # total capture duration (seconds)
CAPTURE_INTERVAL = 0.5       # interval between captures (seconds)
MAX_SAMPLES = 30             # maximum number of images to collect


# === Helper functions ===
def load_csv():
    """Read VIP employee list from CSV"""
    return pd.read_csv(CSV_PATH)


def _ensure_dirs(emp_id):
    """Create directory to save employee images"""
    save_dir = os.path.join(IMG_SAVE_DIR, str(emp_id))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _load_db():
    """Load JSON database (returns empty dict if missing)"""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # corrupted or empty file -> reinitialize
                print(f"[WARN] {DB_PATH} is invalid. Reinitializing.")
                with open(DB_PATH, "w", encoding="utf-8") as fw:
                    fw.write("{}")
                return {}
    return {}


def _save_db(db):
    """Save database to JSON"""
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)


# === Main enroll function ===
def enroll_important(emp_id):
    """Enroll a VIP employee by collecting face crops and embeddings."""
    df = load_csv()
    try:
        row = df[df["Employee ID"] == int(emp_id)].iloc[0]
    except Exception:
        print(f"[ERROR] Employee ID {emp_id} not found in CSV.")
        return

    full_name = row["Full Name"]
    department = row["Department"]
    position = row["Position"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    print(f"[INFO] Starting VIP enrollment for {full_name} (ID {emp_id})...")
    print("[INFO] Please look at the camera. Press 'q' to quit early.")

    embeddings = []
    saved = 0
    last_cap = 0.0
    start_time = time.time()
    save_dir = _ensure_dirs(emp_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed > CAPTURE_DURATION or saved >= MAX_SAMPLES:
            break

        # overlay status text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"Collecting VIP data... {saved}/{MAX_SAMPLES}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Enroll Important (VIP Mode)", overlay)

        now = time.time()
        if now - last_cap >= CAPTURE_INTERVAL and saved < MAX_SAMPLES:
            faces = detect_and_crop_faces(frame)
            if faces:
                # choose the largest detected face
                faces_sorted = sorted(faces, key=lambda f: f.shape[0] * f.shape[1], reverse=True)
                face = faces_sorted[0]

                emb = get_embedding(face)
                if emb is not None and np.linalg.norm(emb) > 0:
                    embeddings.append(emb)
                    saved += 1

                    # save cropped face image
                    img_name = os.path.join(
                        save_dir, f"{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )
                    cv2.imwrite(img_name, face)

            last_cap = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("[WARN] No embeddings were collected.")
        return

    # average embedding
    mean_emb = (sum(embeddings) / len(embeddings)).tolist()

    # update DB
    db = _load_db()
    db[str(emp_id)] = {
        "name": full_name,
        "department": department,
        "position": position,
        "embedding": mean_emb,
    }
    _save_db(db)

    print(f"[INFO] VIP enrollment completed for {full_name} (ID {emp_id}) — {saved} images saved at {save_dir}")


if __name__ == "__main__":
    emp_id = input("Enter Employee ID (VIP): ")
    enroll_important(emp_id)