import cv2
import time
import os
import json
from datetime import datetime

import mediapipe as mp

from src.video_stream import WebcamStream
from src.detect_faces import detect_faces
from src.recognize import recognize
from src.attendance import log_attendance
from src.antispoof import check_liveness

# ==========================
# CONFIG
# ==========================
DB_PATH = "db/employees.json"
SNAPSHOT_DIR = "snapshots"

DISPLAY_DURATION = 3.0
AVATAR_SIZE = (70, 70)

LIVENESS_CHECK_INTERVAL = 5
RECOGNITION_INTERVAL = 3

SHOW_TERMINAL_LOG = True

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ==========================
# MediaPipe Face Mesh
# ==========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# UI STATE
# ==========================
last_display_name = None
last_display_time = None
last_display_timestamp = 0.0
last_avatar = None
last_status_text = None
last_status_color = (255, 255, 255)

# ==========================
# UTILS
# ==========================
def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_avatar(emp_id):
    db = load_db()
    try:
        avatar_path = db[str(emp_id)].get("avatar")
        if avatar_path and os.path.exists(avatar_path):
            img = cv2.imread(avatar_path)
            return cv2.resize(img, AVATAR_SIZE)
    except:
        pass
    return None


def draw_corner_bbox(frame, x1, y1, x2, y2, color=(0, 255, 255)):
    t = 2
    l = 20
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, t)
    cv2.line(frame, (x2, y1), (x2 - l, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, t)
    cv2.line(frame, (x1, y2), (x1 + l, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, t)
    cv2.line(frame, (x2, y2), (x2 - l, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - l), color, t)


def get_square_face(frame, x1, y1, x2, y2):
    h = y2 - y1
    w = x2 - x1
    size = max(h, w)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1n = max(cx - size // 2, 0)
    y1n = max(cy - size // 2, 0)
    x2n = min(x1n + size, frame.shape[1])
    y2n = min(y1n + size, frame.shape[0])

    return frame[y1n:y2n, x1n:x2n], x1n, y1n


def draw_face_mesh(frame, face_img, offset_x, offset_y):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return

    h, w, _ = face_img.shape
    for lm in results.multi_face_landmarks[0].landmark:
        px = int(lm.x * w) + offset_x
        py = int(lm.y * h) + offset_y
        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)


# ==========================
# MAIN
# ==========================
def realtime_attendance():
    global last_display_name, last_display_time, last_display_timestamp
    global last_avatar, last_status_text, last_status_color

    cap = WebcamStream(src=0).start()
    print("[INFO] Realtime Attendance Started — Press Q to quit")

    frame_count = 0
    prev_time = time.time()

    while True:
        frame = cap.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        annotated = frame.copy()

        now = time.time()
        frame_count += 1

        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        start_detect = time.time()
        boxes = detect_faces(frame)
        detect_ms = (time.time() - start_detect) * 1000

        if SHOW_TERMINAL_LOG:
            print(f"[FPS] {fps:.1f} | Faces={len(boxes)} | Detect={detect_ms:.2f}ms")

        if boxes:
            x1, y1, x2, y2 = max(
                boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

            face_sq, fx, fy = get_square_face(frame, x1, y1, x2, y2)
            if face_sq.size > 0:
                draw_corner_bbox(annotated, x1, y1, x2, y2)
                draw_face_mesh(annotated, face_sq, fx, fy)

                if frame_count % LIVENESS_CHECK_INTERVAL == 0:
                    t0 = time.time()
                    is_real = check_liveness(face_sq)
                    live_ms = (time.time() - t0) * 1000

                    if SHOW_TERMINAL_LOG:
                        print(f"[LIVENESS] real={is_real} | {live_ms:.2f}ms")

                    if not is_real:
                        last_display_name = "Unknown"
                        last_status_text = "FAKE"
                        last_status_color = (0, 0, 255)
                        last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        last_display_timestamp = now
                        last_avatar = cv2.resize(face_sq, AVATAR_SIZE)

                if frame_count % RECOGNITION_INTERVAL == 0:
                    t1 = time.time()
                    emp_id, name = recognize(face_sq)
                    rec_ms = (time.time() - t1) * 1000

                    if SHOW_TERMINAL_LOG:
                        print(f"[RECOGNIZE] ID={emp_id} | {rec_ms:.2f}ms")

                    if emp_id is None:
                        last_display_name = "Unknown"
                        last_status_text = "REAL (Unknown)"
                        last_status_color = (0, 255, 255)
                        last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        last_display_timestamp = now
                        last_avatar = cv2.resize(face_sq, AVATAR_SIZE)
                    else:
                        log_attendance(emp_id)

                        last_display_name = name
                        last_status_text = "REAL"
                        last_status_color = (0, 255, 0)
                        last_display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        last_display_timestamp = now

                        avatar = load_avatar(emp_id)
                        last_avatar = avatar if avatar is not None else cv2.resize(face_sq, AVATAR_SIZE)

        if last_display_name and (now - last_display_timestamp) <= DISPLAY_DURATION:
            h, w, _ = annotated.shape
            bar_h = 100

            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

            ax, ay = 20, h - bar_h + 15
            if last_avatar is not None:
                annotated[ay:ay+70, ax:ax+70] = last_avatar

            tx = ax + 90
            cv2.putText(annotated, f"Employee: {last_display_name}",
                        (tx, h - bar_h + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.putText(annotated, f"Status: {last_status_text}",
                        (tx, h - bar_h + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, last_status_color, 2)

            cv2.putText(annotated, f"Time: {last_display_time}",
                        (tx, h - bar_h + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.imshow("Realtime Attendance", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    cv2.destroyAllWindows()









