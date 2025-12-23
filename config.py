"""
FINOVA ATTENDANCE SYSTEM - Configuration File
Cấu hình cho hệ thống chấm công
"""

# ==================== AI MODELS ====================

# Anti-spoofing Configuration
ENABLE_ANTISPOOF = True  # Set to False to disable anti-spoofing check
ANTISPOOF_THRESHOLD = 0.5  # Score threshold (0.0 - 1.0)
# Higher threshold = stricter (more likely to reject)
# Lower threshold = more lenient (more likely to accept)

# Face Recognition Configuration
RECOGNITION_THRESHOLD = 0.6  # Similarity threshold for face matching
# Higher threshold = stricter matching (fewer false positives)
# Lower threshold = more lenient (may have false positives)

# Face Detection Configuration
YOLO_CONFIDENCE = 0.5  # Confidence threshold for face detection
YOLO_IOU = 0.4  # IoU threshold for NMS
YOLO_MAX_DETECTIONS = 5  # Maximum number of faces to detect

# ==================== SYSTEM SETTINGS ====================

# Camera Settings
CAMERA_INDEX = 0  # Default camera index (0 = first camera)
FRAME_WIDTH = 640  # Camera frame width
FRAME_HEIGHT = 480  # Camera frame height

# Recognition Intervals (frames)
LIVENESS_CHECK_INTERVAL = 5  # Check liveness every N frames
RECOGNITION_INTERVAL = 3  # Run recognition every N frames

# Cooldown Settings (seconds)
GLOBAL_COOLDOWN = 1.2  # Minimum time between any attendance logs
PER_EMPLOYEE_COOLDOWN = 5.0  # Minimum time between logs for same employee
DISPLAY_DURATION = 5.0  # How long to display recognition result

# ==================== FILE PATHS ====================

# Database Files
DATA_EMPLOYEE_CSV = "db/data_employee.csv"
IMPORTANT_EMPLOYEE_CSV = "db/important_employee.csv"
ATTENDANCE_CSV = "logs/attendance.csv"
ACCESS_LOGS_CSV = "db/access_logs.csv"
EMPLOYEES_JSON = "db/employees.json"
IMPORTANT_EMPLOYEES_JSON = "db/important_employees.json"

# Directories
SNAPSHOT_DIR = "snapshots"
ENROLLMENT_DIR = "data/employees"
LOGS_DIR = "logs"
MODELS_DIR = "models"

# Model Files
YOLO_MODEL = "models/yolov8n-face-lindevs.pt"
ARCFACE_MODEL = "models/w600k_r50.onnx"
ANTISPOOF_MODEL = "models/antispoof_resnet18.pt"

# ==================== WEB INTERFACE ====================

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# Database Settings
DATABASE_URL = "sqlite:///finova_attendance.db"

# ==================== LOGGING ====================

# Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Enable/Disable Logging
ENABLE_CONSOLE_LOG = True
ENABLE_FILE_LOG = True
LOG_FILE = "logs/system.log"

# ==================== ADVANCED SETTINGS ====================

# Performance
USE_GPU = False  # Set to True if CUDA is available
NUM_THREADS = 4  # Number of threads for CPU inference

# Snapshot Settings
SAVE_SNAPSHOTS = True  # Save face snapshots for audit
SNAPSHOT_QUALITY = 95  # JPEG quality (0-100)
SNAPSHOT_CLEANUP_DAYS = 30  # Delete snapshots older than N days

# Avatar Settings
AVATAR_SIZE = (70, 70)  # Avatar display size in pixels

# ==================== NOTES ====================
"""
Configuration Tips:

1. Anti-spoofing:
   - Set ENABLE_ANTISPOOF = False if you want to disable fake detection
   - Increase ANTISPOOF_THRESHOLD (e.g., 0.7) for stricter checking
   - Decrease ANTISPOOF_THRESHOLD (e.g., 0.3) for more lenient checking

2. Face Recognition:
   - Increase RECOGNITION_THRESHOLD (e.g., 0.7) to reduce false positives
   - Decrease RECOGNITION_THRESHOLD (e.g., 0.5) to be more lenient

3. Performance:
   - Increase LIVENESS_CHECK_INTERVAL to check less frequently (faster)
   - Increase RECOGNITION_INTERVAL to recognize less frequently (faster)
   - Set USE_GPU = True if you have CUDA-capable GPU

4. Cooldown:
   - Increase PER_EMPLOYEE_COOLDOWN to prevent duplicate logs
   - Decrease for more frequent logging (not recommended)
"""
