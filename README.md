# HRMS
# HRMS – Human Resource Management System

## Overview
**HRMS (Human Resource Management System)** is an AI-powered platform designed to automate employee attendance and human resource management.  
It integrates **face recognition technology** with traditional HR workflows, enabling contactless attendance logging, work-time analytics, and secure access control.

This project is developed by my team in ICT Department, as part of the academic curriculum and research initiatives in intelligent systems.

---

## Main Features

### AI-based Face Recognition
- **Face detection:** YOLOv8-Face for real-time, multi-person detection.  
- **Face embedding:** ArcFace model for high-precision identity representation.  
- **Verification modes:**
  - **One-to-One:** Validate one user against stored embedding (e.g., single-entry kiosk).  
  - **One-to-Many:** Identify among multiple users simultaneously (e.g., group check-in).

### Attendance System
- Automatic check-in/check-out when the employee’s face is recognized.
- Logging of late arrivals, absences, or leave requests.
- Exportable daily/weekly/monthly attendance reports in CSV or Excel format.

### HR Management
- CRUD (Create, Read, Update, Delete) operations for employee data.  
- Role-based access: **Admin**, **HR Manager**, **Employee**.  
- Leave request submission and approval workflow.  
- Dashboard to visualize employee statistics, department metrics, and working hours.

### Access Control & Security
- Authentication with encrypted credentials.  
- Session management with JWT.  
- AI-based access restriction: allow entry only when recognized as a registered employee.

---

## System Architecture

1. AI Module (ai_module/)
- This module contains all computer vision and deep learning logic used for facial recognition and attendance tracking.

- The models/ directory stores pretrained model weights and configuration files.

- The src/ folder includes scripts for face detection (detect_faces.py), embedding extraction (extract_embeddings.py), one-to-many recognition, and one-to-one verification (for future extension).

- The file realtime_attendance.py handles live attendance logging using the camera feed.

- The enroll.py script is responsible for registering new employee faces into the database.

- All supporting utilities (for preprocessing, I/O, and face alignment) are located in src/utils/.

2. Backend (backend/)
- This directory contains the core API server built with Flask or FastAPI, managing business logic, database interaction, and AI integration.

- app.py is the main entry point of the application.

- The routes/ folder defines RESTful endpoints for different features:

- auth.py handles login, logout, and registration.

- employee.py manages employee profiles and photo uploads.

- attendance.py manages check-in/check-out and attendance logs.

- leave.py processes leave requests and approvals.

- announcement.py handles internal company announcements.

- payroll.py manages payroll viewing and salary details.

- admin.py supports user-role management and access control.

- ai_integration.py connects the backend to the AI face recognition service.

- The models/ folder defines ORM models (e.g., SQLAlchemy schemas) for users, attendance logs, leave requests, announcements, and payroll records.

- Business logic is implemented in services/, where each service corresponds to a functional area (attendance, leave, payroll, etc.).

- Common helper functions such as date formatting or email utilities are stored in utils/.

- The config.py file contains application settings like database URLs, secret keys, and AI module paths.

3. Frontend (frontend/)
- The frontend is built using Vue.js or React, providing a clean interface for HR managers and employees.

- public/ holds static assets such as logos and icons.

- In src/pages/, each page represents a functional module: login, dashboard, attendance, leave requests, announcements, payroll, and profile.

- Reusable UI components (headers, sidebars, employee cards, etc.) are stored in src/components/.

- The services/ folder contains API wrappers (api.js) for backend communication and authentication logic (auth.js).

Application-wide state management (using Vuex or Redux) resides in src/store/.

4. Configuration (config/)
- This directory holds environment and logging configurations.

- settings.yaml defines environment variables for development and production.

- logging.conf specifies logging levels and handlers for monitoring system activity.

5. Scripts (scripts/)
- This folder includes deployment scripts, database migration tools, and data seeding utilities that simplify setup and maintenance.
### Python Environment

The project runs in a Conda environment named **hrms_attendance**, using **Python 3.10**.

To set it up:
```bash
conda create -n hrms_attendance python=3.10.18
conda activate hrms_attendance
pip install -r requirements.txt
torch==2.2.2
torchvision==0.17.2
ultralytics==8.3.203
facenet-pytorch==2.6.0
opencv-python==4.11.0.86
pandas==2.3.2
matplotlib==3.10.6
scikit-learn==1.7.2
mediapipe==0.10.21
