# Finova Attendance System - AI Face Recognition

## Overview
Professional attendance management system with AI-powered face recognition, featuring YOLOv8 face detection, ArcFace embeddings, and anti-spoofing technology.

## Features
- **YOLOv8 Face Detection**: Advanced face detection with high accuracy
- **ArcFace Embeddings**: State-of-the-art face recognition technology  
- **Anti-spoofing**: Prevents fake face attacks
- **Real-time Recognition**: Live camera feed processing
- **Modern Web Interface**: Glassmorphism design with responsive layout
- **Employee Management**: Complete enrollment and management system
- **Database Architecture**: Professional SQLite database with 4 tables
- **Dual API System**: Legacy CSV + Modern Database APIs

## System Architecture

### AI Models
- **YOLOv8**: `models/yolov8n-face-lindevs.pt` - Face detection
- **ArcFace**: `models/w600k_r50.onnx` - Face embedding extraction
- **Anti-spoofing**: `models/antispoof_resnet18.pt` - Liveness detection

### Database Structure (New)
**SQLite Database with 4 tables:**
- `employees` - Employee information
- `face_embeddings` - Face recognition vectors
- `attendance_logs` - Check-in/out records
- `ai_inference_logs` - AI model performance logs

### Legacy CSV Files (Maintained)
- `db/data_employee.csv` - All employees (one-to-many recognition)
- `db/important_employee.csv` - VIP employees (one-to-one verification)
- `logs/attendance.csv` - Daily attendance records
- `db/access_logs.csv` - Security access logs

### Data Storage
- `data/employees/` - Enrollment photos (30 frames per employee)
- `snapshots/` - Daily attendance photos for HR review
- `models/` - AI model files
- `web/` - Modern web interface

## Installation

### Prerequisites
- Python 3.8+
- Miniconda/Anaconda
- Webcam/IP Camera

### Setup
1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd finova-attendance
   ```

2. **Create conda environment**
   ```bash
   conda create -n hrms_attendance python=3.9
   conda activate hrms_attendance
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Console Interface
```bash
# Start console menu
start_console.bat
# Or manually:
conda activate hrms_attendance
python main.py
```

**Console Menu Features:**
1. 🎯 Chấm công Real-time (One-to-Many)
2. 🔐 Xác thực nhân viên quan trọng (One-to-One)
3. 👤 Đăng ký nhân viên mới (Enrollment)
4. 👥 Đăng ký nhân viên quan trọng
5. 📊 Xem báo cáo chấm công
6. 🌐 Khởi động Web Interface
7. 🗄️ Khởi động Database API
8. 🧪 Test AI Models
9. ℹ️ Thông tin hệ thống

### Quick Start - Legacy System
```bash
# Start legacy CSV-based system
start_system.bat
# Or manually:
conda activate hrms_attendance
python finova_api.py
```

### Quick Start - Modern Database System
```bash
# Start modern database system
start_backend.bat
# Or manually:
conda activate hrms_attendance
python backend/api.py
```

### Web Interface
- **Main Dashboard**: http://localhost:5000/web/modern-index.html
- **Employee Enrollment**: http://localhost:5000/web/enroll.html
- **API Documentation**: http://localhost:5000/

### Command Line Tools
- **Real-time Attendance**: `python src/realtime_attendance.py`
- **VIP Verification**: `python src/verify.py`
- **Generate Reports**: `python src/report.py`
- **Employee Enrollment**: `python src/enroll.py`

## API Systems

### Legacy API (finova_api.py)
- CSV-based data storage
- Compatible with existing src/ modules
- Endpoints: `/api/employees`, `/api/recognize`, `/api/verify`

### Modern Database API (backend/api.py)
- SQLite database with professional schema
- Advanced analytics and logging
- Endpoints: `/api/employees`, `/api/face/recognize`, `/api/attendance/today`

## File Structure
```
finova-attendance/
├── backend/                # Modern database system
│   ├── models.py          # SQLAlchemy database models
│   ├── services.py        # Business logic services
│   └── api.py             # REST API endpoints
├── src/                   # AI processing modules
│   ├── detect_faces.py    # YOLOv8 face detection
│   ├── extract_embeddings.py # ArcFace embeddings
│   ├── antispoof.py       # Anti-spoofing detection
│   ├── recognize.py       # Face recognition
│   ├── verify.py          # One-to-one verification
│   ├── attendance.py      # Attendance logging
│   ├── enroll.py          # Employee enrollment
│   └── realtime_attendance.py # Live recognition
├── models/                # AI model files
│   ├── yolov8n-face-lindevs.pt
│   ├── w600k_r50.onnx
│   └── antispoof_resnet18.pt
├── web/                   # Modern web interface
│   ├── modern-index.html  # Main dashboard
│   ├── enroll.html        # Employee enrollment
│   ├── css/modern-style.css # Glassmorphism design
│   └── js/                # JavaScript modules
├── db/                    # CSV database files
├── data/employees/        # Enrollment photos
├── snapshots/             # Daily attendance photos
├── logs/                  # System logs
├── finova_api.py          # Legacy API server
├── start_system.bat       # Start legacy system
├── start_backend.bat      # Start database system
└── requirements.txt       # Dependencies
```

## Recognition Modes

### One-to-Many (General Attendance)
- Uses `db/data_employee.csv` or database `employees` table
- Recognizes any enrolled employee
- Logs to `logs/attendance.csv` or `attendance_logs` table
- Saves snapshots for HR review

### One-to-One (VIP Verification)
- Uses `db/important_employee.csv`
- Verifies specific employee identity
- Logs to `db/access_logs.csv`
- Higher security threshold

## Development

### Adding New Employees
1. Use web interface: `/web/enroll.html`
2. Or command line: `python src/enroll.py`
3. Capture 30 photos from different angles
4. System automatically extracts embeddings

### Database Migration
- Legacy CSV files maintained for compatibility
- New database system provides advanced features
- Both systems can run simultaneously

## License
Professional attendance management system for enterprise use.