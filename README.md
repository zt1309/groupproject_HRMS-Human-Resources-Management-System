# Face Attendance System

## Overview
AI-powered face recognition attendance system using YOLOv8 face detection, ArcFace embeddings, and anti-spoofing technology for secure employee verification.

## Features
- YOLOv8 Face Detection: Advanced face detection with high accuracy
- ArcFace Embeddings: Face recognition technology
- Anti-spoofing: Prevents fake face attacks and liveness detection
- Real-time Recognition: Live camera feed processing
- Employee Management: Complete enrollment and management system
- CSV Database: Attendance and access logs
- Multiple Recognition Modes: One-to-many and one-to-one verification

## System Components

### AI Models
- YOLOv8: models/yolov8n-face-lindevs.pt - Face detection
- ArcFace: models/w600k_r50.onnx - Face embedding extraction
- Anti-spoofing: models/antispoof_resnet18.pt - Liveness detection

### Database Files
- db/data_employee.csv - All employees for recognition
- db/important_employee.csv - VIP employees for verification
- db/employees.json - Employee information and embeddings
- db/important_employees.json - VIP employee data
- logs/attendance.csv - Daily attendance records
- logs/access_logs.csv - Security access logs

### Data Storage
- data/employees/ - Enrollment photos (multiple frames per employee)
- snapshots/ - Attendance photos for HR review
- models/ - AI model files
- db/ - CSV and JSON database files

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or IP Camera

### Setup
1. Create Python environment
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Menu (Console)
Run the main program:
```bash
python main.py
```

Menu options:
1. Enroll new employee - Capture employee photos and create embeddings
2. Run realtime attendance - One-to-many attendance verification
3. Verify face - One-to-one VIP verification
4. Generate report - Create attendance reports
5. Exit - Quit the program

### Command Line Tools
- Real-time Attendance: python src/realtime_attendance.py
- VIP Verification: python src/verify.py
- Generate Reports: python src/report.py
- Employee Enrollment: python src/enroll.py
- Extract Embeddings: python src/extract_embeddings.py
- Detect Faces: python src/detect_faces.py
- Check Liveness: python src/antispoof.py
- Recognize Faces: python src/recognize.py

## Recognition Modes

### One-to-Many Attendance
- Recognizes any enrolled employee
- Uses db/data_employee.csv
- Logs to logs/attendance.csv
- Saves attendance photos to snapshots/

### One-to-One Verification
- Verifies specific VIP employee identity
- Uses db/important_employee.csv
- Logs to logs/access_logs.csv
- Higher security threshold for verification

## File Structure
```
system_for_gr_prj/
├── main.py                  # Entry point and main menu
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── detect_faces.py     # YOLOv8 face detection
│   ├── extract_embeddings.py # ArcFace embeddings
│   ├── antispoof.py        # Liveness detection
│   ├── recognize.py        # Face recognition
│   ├── verify.py           # One-to-one verification
│   ├── attendance.py       # Attendance logging
│   ├── enroll.py           # Employee enrollment
│   ├── realtime_attendance.py # Live recognition
│   ├── report.py           # Report generation
│   ├── video_stream.py     # Video capture
│   ├── check.py            # System checks
│   └── frames.py           # Frame processing
├── models/
│   ├── yolov8n-face-lindevs.pt # Face detection model
│   ├── w600k_r50.onnx          # ArcFace model
│   └── antispoof_resnet18.pt   # Anti-spoofing model
├── db/
│   ├── data_employee.csv       # Employee data
│   ├── important_employee.csv  # VIP employee data
│   ├── employees.json          # JSON employee database
│   └── important_employees.json # JSON VIP database
├── data/employees/             # Enrollment photos
├── snapshots/                  # Attendance photos
├── logs/
│   ├── attendance.csv          # Attendance records
│   └── access_logs.csv         # Verification logs
└── __pycache__/
```

## Workflow

### Enrollment Process
1. Run main.py and select "Enroll new employee"
2. Enter employee ID
3. Capture 30 photos from different angles
4. System extracts face embeddings automatically
5. Employee is added to database

### Attendance Process
1. Run main.py and select "Run realtime attendance"
2. System detects faces in camera feed
3. Performs liveness check (real vs fake)
4. Recognizes employee and logs attendance
5. Saves attendance photo to snapshots/

### Verification Process
1. Run main.py and select "Verify face"
2. System detects face and checks liveness
3. Verifies against VIP employee database
4. Logs access attempt with timestamp
5. Grants or denies access

## Key Features

### Face Detection
- YOLOv8-nano model for real-time detection
- Optimized for speed and accuracy
- Handles multiple faces in frame

### Face Recognition
- ArcFace embeddings for 1:N recognition
- Cosine similarity for matching
- Configurable confidence threshold

### Liveness Detection
- ResNet18 anti-spoofing model
- Detects fake face attacks
- Frame-based verification

### Attendance Logging
- Automatic check-in/check-out
- Timestamp recording
- Photo snapshots for verification
- CSV format for easy analysis

## Performance
- Face Detection: Real-time at 30 FPS
- Face Recognition: <100ms per face
- Liveness Check: <50ms per face
- System throughput: Multiple employees per second
