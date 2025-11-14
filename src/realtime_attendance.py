import cv2
import time
import os
from datetime import datetime
from collections import defaultdict

from src.detect_faces import yolo
from src.recognize import recognize
from src.attendance import log_attendance
from src.antispoof import check_liveness  # Add anti-spoofing

# ======================
# Time Configuration
# ======================
GLOBAL_COOLDOWN = 1.2    # delay between 2 persons (seconds)
PER_EMP_COOLDOWN = 5.0   # prevent duplicate logs for same employee (seconds)
DISPLAY_DURATION = 2.0   # keep name displayed after check (seconds)

# ======================
# Snapshot Configuration
# ======================
SNAPSHOT_DIR = "snapshots"  # snapshots/<emp_id>/*.jpg
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(emp_id, face):
    """Save cropped face image to snapshots/<emp_id>/"""
    try:
        emp_dir = os.path.join(SNAPSHOT_DIR, str(emp_id))
        os.makedirs(emp_dir, exist_ok=True)
        filename = f"{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(emp_dir, filename), face)
    except Exception as e:
        print(f"[WARN] Failed to save snapshot for {emp_id}: {e}")


def realtime_attendance():
    cap = cv2.VideoCapture(0)
    print("[INFO] Realtime Attendance System Started (press 'q' to quit)")

    # Store timestamps for logging and display
    last_any_log = 0.0
    last_emp_log = defaultdict(lambda: 0.0)
    last_display = defaultdict(lambda: 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        results = yolo(frame)
        annotated = results[0].plot()

        # Check cooldown between persons (queue)
        global_ready = (now - last_any_log) >= GLOBAL_COOLDOWN

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face = frame[y1:y2, x1:x2]
                if face.size <= 0:
                    continue

                # Anti-spoofing check
                is_real = check_liveness(face)
                if not is_real:
                    cv2.putText(
                        annotated,
                        "FAKE FACE DETECTED!",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    continue

                # Face recognition
                emp_id, name = recognize(face)
                if emp_id is None or name == "Unknown":
                    continue

                # Per-employee cooldown
                if (now - last_emp_log[emp_id]) < PER_EMP_COOLDOWN:
                    if now - last_display[emp_id] <= DISPLAY_DURATION:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            annotated,
                            f"{emp_id} - {name}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    continue

                # Not ready for next person
                if not global_ready:
                    cv2.putText(
                        annotated,
                        "Please wait... next person in queue",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    continue

                # Log attendance (Check-in / Check-out)
                log_attendance(emp_id)
                save_snapshot(emp_id, face)

                # Display info
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{emp_id} - {name}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Update timestamps
                last_emp_log[emp_id] = now
                last_display[emp_id] = now
                last_any_log = now
                global_ready = False
                break  # prevent duplicate logs in same frame

        # Display frame
        cv2.imshow("Realtime Attendance (Queue Mode)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_attendance()

