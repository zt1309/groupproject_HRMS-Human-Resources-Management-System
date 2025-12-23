import cv2
import time
import os
from datetime import datetime
from collections import defaultdict
from typing import Tuple, Optional
from enum import Enum
import numpy as np

from src.video_stream import WebcamStream
from src.detect_faces import detect_faces
from src.recognize import recognize
from src.attendance import log_attendance
from src.antispoof import check_liveness

# Constants
GLOBAL_COOLDOWN = 1.2
PER_EMP_COOLDOWN = 5.0
DISPLAY_DURATION = 5.0
LIVENESS_CHECK_INTERVAL = 5
RECOGNITION_INTERVAL = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_FACE_SIZE = 20
SNAPSHOT_DIR = "snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class VerificationStatus(Enum):
    """Verification status with display properties."""
    REAL = ("REAL", (0, 255, 0))
    FAKE = ("FAKE", (0, 0, 255))
    COOLDOWN = ("COOLDOWN", (0, 255, 255))
    WAITING = ("WAITING", (255, 255, 0))
    
    def __init__(self, text: str, color: Tuple[int, int, int]):
        self.text = text
        self.color = color


class SnapshotManager:
    """Handles snapshot saving."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
    
    def save(self, emp_id: str, face: np.ndarray) -> bool:
        """Save employee face snapshot."""
        try:
            emp_dir = os.path.join(self.base_dir, str(emp_id))
            os.makedirs(emp_dir, exist_ok=True)
            filename = f"{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(emp_dir, filename), face)
            return True
        except Exception as e:
            print(f"[ERROR] Save snapshot failed: {e}")
            return False


class CooldownManager:
    """Manages cooldown timers for attendance logging."""
    
    def __init__(self, global_cd: float, per_emp_cd: float, display_duration: float):
        self.global_cooldown = global_cd
        self.per_emp_cooldown = per_emp_cd
        self.display_duration = display_duration
        
        self.last_any_log = 0.0
        self.last_emp_log = defaultdict(lambda: 0.0)
        self.last_display = defaultdict(lambda: 0.0)
    
    def is_global_ready(self, now: float) -> bool:
        """Check if global cooldown has elapsed."""
        return (now - self.last_any_log) >= self.global_cooldown
    
    def is_employee_ready(self, emp_id: str, now: float) -> bool:
        """Check if employee-specific cooldown has elapsed."""
        return (now - self.last_emp_log[emp_id]) >= self.per_emp_cooldown
    
    def is_display_active(self, emp_id: str, now: float) -> bool:
        """Check if display period is still active for employee."""
        return (now - self.last_display[emp_id]) <= self.display_duration
    
    def update(self, emp_id: str, now: float):
        """Update cooldown timers after successful logging."""
        self.last_emp_log[emp_id] = now
        self.last_display[emp_id] = now
        self.last_any_log = now


class StatusBarRenderer:
    """Renders clean status bar with employee info and status."""
    
    def __init__(self, duration: float = DISPLAY_DURATION):
        self.duration = duration
        self.display_name = None
        self.display_id = None
        self.display_time = None
        self.display_timestamp = 0.0
        self.status = None
        self.bar_height = 80
    
    def update(self, emp_id: str, name: str, status: VerificationStatus, timestamp: float):
        """Update status bar with new verification result."""
        self.display_id = emp_id
        self.display_name = name
        self.display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display_timestamp = timestamp
        self.status = status
    
    def render(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Render status bar with ID, name, status, and time."""
        elapsed = current_time - self.display_timestamp
        
        if self.display_name is None or elapsed > self.duration:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - self.bar_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        
        # Draw employee ID and name (left side)
        cv2.putText(frame, f"ID: {self.display_id}",
                    (20, h - self.bar_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        cv2.putText(frame, f"Name: {self.display_name}",
                    (20, h - self.bar_height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        # Draw time (right side, top)
        time_text = f"Time: {self.display_time}"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        time_x = w - text_size[0] - 20
        cv2.putText(frame, time_text,
                    (time_x, h - self.bar_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)
        
        # Draw status (right side, bottom)
        status_text = f"Status: {self.status.text}"
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        status_x = w - status_size[0] - 20
        cv2.putText(frame, status_text,
                    (status_x, h - self.bar_height + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.status.color, 2)
        
        return frame


class FPSCounter:
    """Efficient FPS tracking with smoothing."""
    
    def __init__(self, smoothing: float = 0.9):
        self.prev_time = time.time()
        self.fps = 0.0
        self.smoothing = smoothing
    
    def update(self) -> float:
        """Calculate smoothed FPS."""
        now = time.time()
        diff = now - self.prev_time
        instant_fps = 1.0 / diff if diff > 0 else 0.0
        
        # Exponential moving average
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        self.prev_time = now
        return self.fps


class RealtimeAttendanceSystem:
    """Optimized realtime attendance system with clean bbox display."""
    
    def __init__(self):
        self.snapshot_mgr = SnapshotManager(SNAPSHOT_DIR)
        self.cooldown_mgr = CooldownManager(GLOBAL_COOLDOWN, PER_EMP_COOLDOWN, DISPLAY_DURATION)
        self.status_bar = StatusBarRenderer(DISPLAY_DURATION)
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        self.cap = None
    
    def _is_valid_face(self, face: np.ndarray) -> bool:
        """Validate face region."""
        if face.size == 0:
            return False
        h, w = face.shape[:2]
        return h >= MIN_FACE_SIZE and w >= MIN_FACE_SIZE
    
    def _check_liveness_throttled(self, face: np.ndarray) -> bool:
        """Check liveness with throttling."""
        if self.frame_count % LIVENESS_CHECK_INTERVAL == 0:
            is_real = check_liveness(face)
            print(f"[LIVENESS] real={is_real}")
            return is_real
        return True  # Skip check, assume real
    
    def _draw_face_box(self, frame: np.ndarray, box: Tuple[int, int, int, int], 
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        """Draw clean bounding box around face."""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    def _process_face(self, face: np.ndarray, box: Tuple[int, int, int, int],
                      annotated: np.ndarray, now: float) -> Tuple[np.ndarray, bool]:
        """Process single detected face."""
        # Validate face
        if not self._is_valid_face(face):
            return annotated, False
        
        # Liveness check (throttled)
        is_real = self._check_liveness_throttled(face)
        if not is_real:
            self._draw_face_box(annotated, box, color=(0, 0, 255))  # Red for fake
            self.status_bar.update("Unknown", "Unknown", VerificationStatus.FAKE, now)
            return annotated, False
        
        # Recognition (throttled)
        if self.frame_count % RECOGNITION_INTERVAL != 0:
            self._draw_face_box(annotated, box, color=(128, 128, 128))  # Gray - processing
            return annotated, False
        
        start = time.time()
        emp_id, name = recognize(face)
        duration = (time.time() - start) * 1000
        print(f"[RECOGNIZE] ID={emp_id}, Name={name}, {duration:.2f} ms")
        
        if emp_id is None:
            self._draw_face_box(annotated, box, color=(0, 165, 255))  # Orange - unknown
            return annotated, False
        
        # Check per-employee cooldown
        if not self.cooldown_mgr.is_employee_ready(emp_id, now):
            if self.cooldown_mgr.is_display_active(emp_id, now):
                self._draw_face_box(annotated, box, color=(0, 255, 255))  # Yellow - cooldown
            return annotated, False
        
        # Check global cooldown
        if not self.cooldown_mgr.is_global_ready(now):
            self._draw_face_box(annotated, box, color=(255, 255, 0))  # Cyan - waiting
            return annotated, False
        
        # Log attendance
        print(f"[LOG] Attendance recorded → {emp_id} - {name}\n")
        log_attendance(emp_id)
        self.snapshot_mgr.save(emp_id, face)
        
        # Update timers and UI
        self.cooldown_mgr.update(emp_id, now)
        self.status_bar.update(emp_id, name, VerificationStatus.REAL, now)
        
        # Draw green box for successful verification
        self._draw_face_box(annotated, box, color=(0, 255, 0), thickness=3)
        
        return annotated, True
    
    def run(self):
        """Main attendance loop."""
        self.cap = WebcamStream(src=0).start()
        print("[INFO] Realtime Attendance — Started\n")
        
        try:
            while True:
                frame = self.cap.read()
                if frame is None:
                    continue
                
                self.frame_count += 1
                now = time.time()
                
                # Calculate FPS
                fps = self.fps_counter.update()
                print(f"[FPS] {fps:.1f}")
                
                # Resize frame
                frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Detect faces
                start = time.time()
                boxes = detect_faces(frame_small)
                duration = (time.time() - start) * 1000
                print(f"[YOLO] {len(boxes)} face(s) — {duration:.2f} ms")
                
                annotated = frame_small.copy()
                
                # Process faces (stop after first successful attendance)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    face = frame_small[y1:y2, x1:x2]
                    
                    annotated, logged = self._process_face(face, box, annotated, now)
                    if logged:
                        break  # Only one attendance per frame
                
                # Render status bar
                annotated = self.status_bar.render(annotated, now)
                
                # Display
                cv2.imshow("Realtime Attendance", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.stop()
        cv2.destroyAllWindows()
        print("[INFO] System shut down successfully.")


def realtime_attendance():
    """Main entry point for backward compatibility."""
    system = RealtimeAttendanceSystem()
    system.run()


if __name__ == "__main__":
    realtime_attendance()