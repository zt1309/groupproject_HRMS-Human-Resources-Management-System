from src import enroll
from src import report
from src import realtime_attendance
from src import verify

def main():
    while True:
        print("\n=== FACE ATTENDANCE SYSTEM ===")
        print("1. Enroll new employee")
        print("2. Run realtime attendance")
        print("3. Verify face")
        print("4. Generate report")
        print("5. Exit")
        choice = input("Choose option: ")

        if choice == "1":
            emp_id = input("Enter Employee ID: ")
            enroll.enroll_employee(emp_id)
        elif choice == "2":
            realtime_attendance.realtime_attendance()
        elif choice == "3":
            verify.realtime_attendance()
        elif choice == "4":
            report.generate_report()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please select again.")


if __name__ == "__main__":
    main()