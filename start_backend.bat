@echo off
echo ========================================
echo    FINOVA BACKEND DATABASE API
echo    Professional Database System
echo ========================================
echo.

echo Activating conda environment...
call conda activate hrms_attendance

echo.
echo Starting Backend Database API...
echo API Server: http://localhost:5000/
echo Web Interface: http://localhost:5000/web/modern-index.html
echo Database: SQLite with 4 tables
echo.

python backend/api.py

pause