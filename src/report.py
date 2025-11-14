import pandas as pd
import matplotlib.pyplot as plt
import os

ATTENDANCE_PATH = "logs/attendance.csv"


def generate_report():
    if not os.path.exists(ATTENDANCE_PATH):
        print("[WARN] Attendance log not found.")
        return

    df = pd.read_csv(ATTENDANCE_PATH)

    # chuẩn hóa tên cột về lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # change alias to "employeeid" and "time"
    if "employee id" in df.columns:
        emp_col = "employee id"
    elif "emp_id" in df.columns:
        emp_col = "emp_id"
    else:
        print("[ERROR] Không tìm thấy cột EmployeeID/emp_id trong attendance.csv")
        print("Các cột hiện tại:", df.columns)
        return

    if "date" in df.columns and "checkin" in df.columns:
        # Đếm số lần check-in theo nhân viên
        summary = df.groupby(emp_col).size().reset_index(name="CheckIns")
    else:
        print("[ERROR] Không tìm thấy cột date/checkin trong attendance.csv")
        print("Các cột hiện tại:", df.columns)
        return

    print(summary)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(summary[emp_col].astype(str), summary["CheckIns"], color="skyblue")
    plt.xlabel("Employee ID")
    plt.ylabel("Check-ins")
    plt.title("Attendance Report")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_report()
