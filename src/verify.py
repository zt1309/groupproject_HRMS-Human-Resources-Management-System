import cv2
import os
import json
import numpy as np
from datetime import datetime
from src.detect_faces import yolo
from src.extract_embeddings import get_embedding
from src.antispoof import check_liveness

# =======================
# Configuration
# =======================
DB_PATH = "db/important_employees.json"
ACCESS_LOG = "db/access_logs.csv"
SNAPSHOT_DIR = "snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def load_db():
    """Load employee database"""
    if not os.path.exists(DB_PATH):
        print("[WARN] Database not found! Creating new...")
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def verify_access(face_img, threshold=0.55):
    """Verify employee access using one-to-one matching"""
    db = load_db()
    if not db:
        print("[WARN] Empty database.")
        return None, None, 0

    emb = get_embedding(face_img)
    if emb is None:
        return None, None, 0

    best_id, best_name, best_score = None, None, -1

    for emp_id, emp_data in db.items():
        db_emb = np.array(emp_data["embedding"])
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_id = emp_id
            best_name = emp_data["name"]

    if best_score > threshold:
        return (best_id, best_name, best_score)
    else:
        return (None, None, best_score)

def log_access(emp_id, name, status, liveness="Unknown"):
    """Log access attempts with timestamp"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    new_row = f"{date_str},{time_str},{emp_id},{name},{status},{liveness}\n"

    if not os.path.exists(ACCESS_LOG):
        with open(ACCESS_LOG, "w", encoding="utf-8") as f:
            f.write("Date,Time,Employee ID,Name,Status,Liveness\n")

    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(new_row)

def save_snapshot(emp_id, name, frame):
    """Save snapshot of verification attempt"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_")
    emp_dir = os.path.join(SNAPSHOT_DIR, str(emp_id))
    os.makedirs(emp_dir, exist_ok=True)
    filename = f"{emp_id}_{safe_name}_{timestamp}.jpg"
    filepath = os.path.join(emp_dir, filename)
    cv2.imwrite(filepath, frame)

def one_to_one_verification():
    """Main verification loop"""
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting One-to-One Access Control... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame)
        annotated = frame.copy()

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face = frame[y1:y2, x1:x2]

                if face.size <= 0:
                    continue

                # Step 1: Liveness detection
                is_real = check_liveness(face)
                if not is_real:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated,
                        "Fake Face Detected",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    print("[ACCESS DENIED] Spoof detected")
                    log_access("Unknown", "Unknown", "Denied", "Fake")
                    continue

                # Step 2: Face matching
                emp_id, name, score = verify_access(face)

                if emp_id is not None:
                    print(f"[ACCESS GRANTED] {name} (ID {emp_id}) | score={score:.2f}")
                    log_access(emp_id, name, "Granted", "Real")
                    save_snapshot(emp_id, name, frame)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{name} - Access Granted",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    print(f"[ACCESS DENIED] | score={score:.2f}")
                    log_access("Unknown", "Unknown", "Denied", "Real")
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated,
                        "Access Denied",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

        cv2.imshow("One-to-One Verification", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    one_to_one_verification()