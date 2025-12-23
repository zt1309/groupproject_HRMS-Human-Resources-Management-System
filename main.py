#!/usr/bin/env python3
"""
FINOVA ATTENDANCE SYSTEM - Main Console Interface
Face Recognition Attendance System with AI
"""

import os
import sys
from datetime import datetime

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print system header"""
    clear_screen()
    print("=" * 70)
    print("           FINOVA ATTENDANCE SYSTEM - AI FACE RECOGNITION")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

def print_menu():
    """Print main menu"""
    print("=== MAIN MENU ===")
    print()
    print("  1. Realtime Attendance (One-to-Many)")
    print("  2. Verify Important Employee (One-to-One)")
    print("  3. Enroll New Employee")
    print("  4. Enroll Important Employee")
    print("  5. Generate Attendance Report")
    print("  6. Start Web Interface")
    print("  7. Start Database API")
    print("  8. Test AI Models")
    print("  9. System Information")
    print("  0. Exit")
    print()

def run_realtime_attendance():
    """Run realtime attendance recognition"""
    print_header()
    print("=== REALTIME ATTENDANCE (ONE-TO-MANY) ===")
    print("-" * 70)
    print("System will recognize all employees in the company")
    print("Press 'q' to quit camera")
    print("-" * 70)
    input("\nPress Enter to start...")
    
    try:
        from src.realtime_attendance import realtime_attendance
        realtime_attendance()
    except ImportError as e:
        print(f"Error: Cannot import module - {e}")
        print("Check file src/realtime_attendance.py")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def run_verify():
    """Run one-to-one verification"""
    print_header()
    print("=== VERIFY IMPORTANT EMPLOYEE (ONE-TO-ONE) ===")
    print("-" * 70)
    print("System verifies identity for executives and managers")
    print("Press 'q' to quit camera")
    print("-" * 70)
    input("\nPress Enter to start...")
    
    try:
        from src.verify import realtime_attendance as verify_attendance
        verify_attendance()
    except ImportError as e:
        print(f"Error: Cannot import module - {e}")
        print("Check file src/verify.py")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def run_enrollment():
    """Run employee enrollment"""
    print_header()
    print("=== ENROLL NEW EMPLOYEE ===")
    print("-" * 70)
    print("System will capture 30 face photos from multiple angles")
    print("-" * 70)
    
    try:
        from src.enroll import enroll_employee
        
        # Get employee information
        print("\nEnter employee information:")
        emp_id = input("Employee ID: ").strip()
        if not emp_id:
            print("Error: Employee ID cannot be empty!")
            input("\nPress Enter to return to menu...")
            return
        
        full_name = input("Full Name: ").strip()
        if not full_name:
            print("Error: Full Name cannot be empty!")
            input("\nPress Enter to return to menu...")
            return
        
        department = input("Department: ").strip()
        position = input("Position: ").strip()
        
        print("\n" + "-" * 70)
        print("Starting photo capture...")
        print("Look at the camera and turn your head slightly for different angles")
        print("-" * 70)
        input("\nPress Enter to start capture...")
        
        # Call enrollment function
        enroll_employee(emp_id, full_name, department, position)
        
        print("\nEmployee enrolled successfully!")
        
    except ImportError as e:
        print(f"Error: Cannot import module - {e}")
        print("Check file src/enroll.py")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def run_enrollment_important():
    """Run important employee enrollment"""
    print_header()
    print("=== ENROLL IMPORTANT EMPLOYEE ===")
    print("-" * 70)
    print("Enrollment for executives and managers (One-to-One Verification)")
    print("-" * 70)
    
    try:
        from src.enroll_important import enroll_important_employee
        
        # Get employee information
        print("\nEnter important employee information:")
        emp_id = input("Employee ID: ").strip()
        if not emp_id:
            print("Error: Employee ID cannot be empty!")
            input("\nPress Enter to return to menu...")
            return
        
        full_name = input("Full Name: ").strip()
        if not full_name:
            print("Error: Full Name cannot be empty!")
            input("\nPress Enter to return to menu...")
            return
        
        department = input("Department: ").strip()
        position = input("Position: ").strip()
        
        print("\n" + "-" * 70)
        print("Starting photo capture...")
        print("-" * 70)
        input("\nPress Enter to start capture...")
        
        # Call enrollment function
        enroll_important_employee(emp_id, full_name, department, position)
        
        print("\nImportant employee enrolled successfully!")
        
    except ImportError as e:
        print(f"Error: Cannot import module - {e}")
        print("Check file src/enroll_important.py")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def run_report():
    """Generate attendance report"""
    print_header()
    print("=== ATTENDANCE REPORT ===")
    print("-" * 70)
    
    try:
        from src.report import generate_report
        
        print("\nSelect report type:")
        print("1. Today's report")
        print("2. Date range report")
        print("3. Employee report")
        
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "-" * 70)
            generate_report("today")
        elif choice == "2":
            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            print("\n" + "-" * 70)
            generate_report("range", start_date, end_date)
        elif choice == "3":
            emp_id = input("Employee ID: ").strip()
            print("\n" + "-" * 70)
            generate_report("employee", emp_id)
        else:
            print("Invalid choice!")
        
    except ImportError as e:
        print(f"Error: Cannot import module - {e}")
        print("Check file src/report.py")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def run_web_interface():
    """Start web interface"""
    print_header()
    print("=== START WEB INTERFACE ===")
    print("-" * 70)
    print("System will start web server with modern interface")
    print("-" * 70)
    
    print("\nSelect system:")
    print("1. Legacy System (CSV-based) - finova_api.py")
    print("2. Modern Database System (SQLite) - backend/api.py")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == "1":
        print("\nStarting Legacy System...")
        print("Web Interface: http://localhost:5000/web/modern-index.html")
        print("API Docs: http://localhost:5000/")
        print("\nPress Ctrl+C to stop server")
        print("-" * 70)
        
        try:
            import finova_api
        except KeyboardInterrupt:
            print("\n\nServer stopped")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == "2":
        print("\nStarting Modern Database System...")
        print("Web Interface: http://localhost:5000/web/modern-index.html")
        print("API Docs: http://localhost:5000/")
        print("Database: SQLite with 4 tables")
        print("\nPress Ctrl+C to stop server")
        print("-" * 70)
        
        try:
            from backend import api
        except KeyboardInterrupt:
            print("\n\nServer stopped")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice!")
    
    input("\nPress Enter to return to menu...")

def run_database_api():
    """Start database API"""
    print_header()
    print("=== START DATABASE API ===")
    print("-" * 70)
    print("Professional database-driven attendance system")
    print("-" * 70)
    print("\nStarting Backend Database API...")
    print("API Server: http://localhost:5000/")
    print("Web Interface: http://localhost:5000/web/modern-index.html")
    print("Database: SQLite with 4 tables")
    print("\nPress Ctrl+C to stop server")
    print("-" * 70)
    
    try:
        from backend import api
    except KeyboardInterrupt:
        print("\n\nServer stopped")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def test_ai_models():
    """Test AI models"""
    print_header()
    print("=== TEST AI MODELS ===")
    print("-" * 70)
    
    try:
        import test_ai_modules
        test_ai_modules.test_ai_modules()
    except ImportError as e:
        print(f"Error: Cannot import test module - {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to menu...")

def show_system_info():
    """Show system information"""
    print_header()
    print("=== SYSTEM INFORMATION ===")
    print("-" * 70)
    
    print("\nAI Models:")
    models = [
        ("YOLOv8 Face Detection", "models/yolov8n-face-lindevs.pt"),
        ("ArcFace Embedding", "models/w600k_r50.onnx"),
        ("Anti-spoofing", "models/antispoof_resnet18.pt")
    ]
    
    for name, path in models:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {name}: {size_mb:.1f} MB")
        else:
            print(f"  [MISSING] {name}: NOT FOUND")
    
    print("\nDatabase Files:")
    db_files = [
        ("All Employees", "db/data_employee.csv"),
        ("Important Employees", "db/important_employee.csv"),
        ("Attendance Logs", "logs/attendance.csv"),
        ("Access Logs", "db/access_logs.csv"),
        ("Embeddings (All)", "db/employees.json"),
        ("Embeddings (VIP)", "db/important_employees.json")
    ]
    
    for name, path in db_files:
        if os.path.exists(path):
            print(f"  [OK] {name}: {path}")
        else:
            print(f"  [WARN] {name}: {path} (will be created)")
    
    print("\nData Folders:")
    folders = [
        ("Enrollment Photos", "data/employees/"),
        ("Snapshots", "snapshots/"),
        ("Logs", "logs/")
    ]
    
    for name, path in folders:
        if os.path.exists(path):
            try:
                count = len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
                print(f"  [OK] {name}: {path} ({count} folders)")
            except:
                print(f"  [OK] {name}: {path}")
        else:
            print(f"  [WARN] {name}: {path} (will be created)")
    
    print("\nWeb Interface:")
    print("  URL: http://localhost:5000/web/modern-index.html")
    print("  API: http://localhost:5000/")
    
    print("\nSystem Requirements:")
    print("  - Python 3.8+")
    print("  - OpenCV, PyTorch, ONNX Runtime")
    print("  - Flask, SQLAlchemy")
    print("  - Webcam/IP Camera")
    
    print("\nDocumentation:")
    print("  - README.md - System overview")
    print("  - DATABASE_STRUCTURE.md - Database documentation")
    print("  - FIXES_APPLIED.md - Recent fixes")
    
    print("-" * 70)
    input("\nPress Enter to return to menu...")

def main():
    """Main function"""
    while True:
        print_header()
        print_menu()
        
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == "1":
            run_realtime_attendance()
        elif choice == "2":
            run_verify()
        elif choice == "3":
            run_enrollment()
        elif choice == "4":
            run_enrollment_important()
        elif choice == "5":
            run_report()
        elif choice == "6":
            run_web_interface()
        elif choice == "7":
            run_database_api()
        elif choice == "8":
            test_ai_models()
        elif choice == "9":
            show_system_info()
        elif choice == "0":
            print_header()
            print("Thank you for using Finova Attendance System!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\nInvalid choice! Please select from 0-9")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSystem stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
