@echo off
echo ========================================
echo    FINOVA ATTENDANCE SYSTEM
echo    AI Face Recognition System
echo ========================================
echo.

echo Activating conda environment...
call conda activate hrms_attendance

echo.
echo Starting Finova API Server...
echo Web Interface: http://localhost:5000/web/modern-index.html
echo API Documentation: http://localhost:5000/
echo.

python finova_api.py

pause