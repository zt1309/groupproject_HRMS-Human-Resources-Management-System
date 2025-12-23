import cv2
import time
import os
import json
from datetime import datetime
from collections import defaultdict

from src.video_stream import WebcamStream
from src.detect_faces import detect_faces
from src.recognize import recognize
from src.attendance import log_attendance
from src.antispoof import check_liveness
from config import (
    EMPLOYEES_JSON,
    SNAPSHOT_DIR,
    DISPLAY_DURATION,
    AVATAR_SIZE,
    LIVENESS_CHECK_INTERVAL,
    RECOGNITION_INTERVAL,
    SAVE_SNAPSHOTS
)

DB_PATH = EMPLOYEES_JSON

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ===== STATUS BAR DATA =====
last_display_name = None
last_display_time = None
last_display_timestamp = 0.0
last_avatar = None
last_status_text = None
last_status_color = (255, 255, 255)


# -----------------------
# Load DB
# -----------------------
def load_db():
    if not os.path.exists(DB_PATH):
        print("[WARN] employees.json not found.")
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# Load Avatar
# -----------------------
def load_avatar(emp_id):
    db = load_db()
    try:
        avatar_path = db[emp_id].get("avatar", None)
        if avatar_path and os.path.exists(avatar_path):
            img = cv2.imread(avatar_path)
            return cv2.resize(img, AVATAR_SIZE)
    except:
        pass
    return None


# -----------------------
# MAIN REALTIME FUNCTION
# -----------------------
def realtime_attendance():
    global last_display_name, last_display_time, last_display_timestamp
    global last_avatar, last_status_text, last_status_color

    cap = WebcamStream(src=0).start()
    print("[INFO] Realtime Attendance Started — Press Q to quit.\n")

    frame_count = 0
    prev_time = time.time()

    while True:
        frame = cap.read()
        if frame is None:
            continue

        now = time.time()
        frame_count += 1

        # ===== FPS =====
        diff = now - prev_time
        fps = 1 / diff if diff > 0 else 0
        prev_time = now
        print(f"[FPS] {fps:.1f}")

        # ===== Resize =====
        frame_small = cv2.resize(frame, (640, 480))

        # ===== YOLO DETECT =====
        start_detect = time.time()
        boxes = detect_faces(frame_small)
        detect_ms = (time.time() - start_detect) * 1000
        print(f"[YOLO] {len(boxes)} face(s) — {detect_ms:.2f} ms")

        annotated = frame_small.copy()

        for (x1, y1, x2, y2) in boxes:

            face = frame_small[y1:y2, x1:x2]
            if face.size <= 0:
                continue

            # ===== LIVENESS =====
            if frame_count % LIVENESS_CHECK_INTERVAL == 0:
                start_live = time.time()
                is_real = check_liveness(face)
                live_ms = (time.time() - start_live) * 1000
                print(f"[LIVENESS] real={is_real} — {live_ms:.2f} ms")

                if not is_real:
                    # Update status bar for FAKE
                    last_display_name = "Unknown"
                    last_status_text = "FAKE"
                    last_status_color = (0, 0, 255)
                    last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    last_display_timestamp = now
                    last_avatar = cv2.resize(face, AVATAR_SIZE)
                    continue

            # ===== RECOGNITION =====
            if frame_count % RECOGNITION_INTERVAL != 0:
                continue

            start_rec = time.time()
            emp_id, name = recognize(face)
            rec_ms = (time.time() - start_rec) * 1000
            print(f"[RECOGNIZE] ID={emp_id}, Name={name}, {rec_ms:.2f} ms")

            if emp_id is None:
                # Unknown but real person
                last_display_name = "Unknown"
                last_status_text = "REAL (Unknown)"
                last_status_color = (0, 255, 255)
                last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_display_timestamp = now
                last_avatar = cv2.resize(face, AVATAR_SIZE)
                continue

            # ===== LOG ATTENDANCE =====
            print(f"[LOG] Attendance recorded → {emp_id} - {name}\n")
            log_attendance(emp_id)
            
            # ===== SAVE SNAPSHOT =====
            if SAVE_SNAPSHOTS:
                try:
                    emp_snapshot_dir = os.path.join(SNAPSHOT_DIR, str(emp_id))
                    os.makedirs(emp_snapshot_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"snapshot_{timestamp}.jpg"
                    snapshot_path = os.path.join(emp_snapshot_dir, snapshot_filename)
                    
                    cv2.imwrite(snapshot_path, face)
                    print(f"[SNAPSHOT] Saved: {snapshot_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save snapshot: {e}")

            # ===== UPDATE STATUS BAR =====
            last_display_name = name
            last_status_text = "REAL"
            last_status_color = (0, 255, 0)
            last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_display_timestamp = now

            avatar = load_avatar(emp_id)
            if avatar is not None:
                last_avatar = avatar
            else:
                last_avatar = cv2.resize(face, AVATAR_SIZE)

            break

        # ===== STATUS BAR =====
        elapsed = now - last_display_timestamp
        if last_display_name and elapsed <= DISPLAY_DURATION:

            h, w, _ = annotated.shape
            bar_h = 100

            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

            # Avatar
            ax, ay = 20, h - bar_h + 10
            annotated[ay:ay+70, ax:ax+70] = last_avatar

            tx = ax + 90

            # NAME
            cv2.putText(annotated, f"Employee: {last_display_name}",
                        (tx, h - bar_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)

            # STATUS (REAL / FAKE / UNKNOWN)
            cv2.putText(annotated, f"Status: {last_status_text}",
                        (tx, h - bar_h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        last_status_color, 2)

            # TIME
            cv2.putText(annotated, f"Time: {last_display_time}",
                        (tx, h - bar_h + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (200, 200, 200), 2)

        # ===== SHOW =====
        cv2.imshow("Realtime Attendance", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()








