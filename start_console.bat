@echo off
echo ========================================
echo    FINOVA ATTENDANCE SYSTEM
echo    Console Interface
echo ========================================
echo.

echo Activating conda environment...
call conda activate hrms_attendance

echo.
echo Starting Console Interface...
echo.

python main.py

pause