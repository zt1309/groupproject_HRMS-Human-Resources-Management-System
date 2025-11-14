import os
import csv
import json
from datetime import datetime

DB_PATH = "db/employees.json"
ATTENDANCE_PATH = "logs/attendance.csv"


def load_db():
    """Load employees.json"""
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_csv():
    """Tạo file attendance.csv nếu chưa có"""
    if not os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Employee ID", "Full Name", "Department", "Position",
                "Date", "CheckIn", "CheckOut"
            ])
            writer.writeheader()


def log_attendance(emp_id):
    """Log attendance: lần đầu -> CheckIn, lần sau -> CheckOut"""
    init_csv()
    db = load_db()

    if str(emp_id) not in db:
        print(f"[WARN] Employee {emp_id} không có trong database.")
        return

    emp = db[str(emp_id)]
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    rows = []
    found = False

    # đọc tất cả record hiện tại
    with open(ATTENDANCE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # nếu đã có CheckIn cho hôm nay, thì lần này cập nhật CheckOut
            if row["Employee ID"] == str(emp_id) and row["Date"] == date_str:
                if row["CheckOut"] == "":
                    row["CheckOut"] = time_str
                    print(f"[INFO] {emp['name']} đã CheckOut lúc {time_str}")
                found = True
            rows.append(row)

    # nếu chưa có record hôm nay -> thêm CheckIn mới
    if not found:
        new_row = {
            "Employee ID": emp_id,
            "Full Name": emp["name"],
            "Department": emp["department"],
            "Position": emp["position"],
            "Date": date_str,
            "CheckIn": time_str,
            "CheckOut": ""
        }
        rows.append(new_row)
        print(f"[INFO] {emp['name']} đã CheckIn lúc {time_str}")

    # ghi lại tất cả record
    with open(ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Employee ID", "Full Name", "Department", "Position",
            "Date", "CheckIn", "CheckOut"
        ])
        writer.writeheader()
        writer.writerows(rows)




