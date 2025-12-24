import os
import csv
import json
from datetime import datetime

DB_PATH = "db/employees.json"
ATTENDANCE_PATH = "logs/attendance.csv"


def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_csv():
    os.makedirs(os.path.dirname(ATTENDANCE_PATH), exist_ok=True)

    if not os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Employee ID",
                    "Full Name",
                    "Department",
                    "Position",
                    "Date",
                    "CheckIn",
                    "CheckOut",
                    "LivenessResult",
                    "LivenessScore",
                    "Note"
                ]
            )
            writer.writeheader()


def log_attendance(
    emp_id,
    liveness_result="REAL",
    liveness_score=None,
    note=""
):
    init_csv()
    db = load_db()

    if str(emp_id) not in db:
        print(f"[WARN] Employee {emp_id} không có trong database.")
        return

    emp = db[str(emp_id)]
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    rows = []
    found_today = False

    with open(ATTENDANCE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Employee ID"] == str(emp_id) and row["Date"] == date_str:
                if row["CheckOut"] == "":
                    row["CheckOut"] = time_str
                    row["LivenessResult"] = liveness_result
                    row["LivenessScore"] = liveness_score if liveness_score is not None else ""
                    row["Note"] = note
                    print(f"[INFO] {emp['name']} CheckOut in {time_str}")
                found_today = True
            rows.append(row)

    if not found_today:
        new_row = {
            "Employee ID": emp_id,
            "Full Name": emp["name"],
            "Department": emp["department"],
            "Position": emp["position"],
            "Date": date_str,
            "CheckIn": time_str,
            "CheckOut": "",
            "LivenessResult": liveness_result,
            "LivenessScore": liveness_score if liveness_score is not None else "",
            "Note": note
        }
        rows.append(new_row)
        print(f"[INFO] {emp['name']} CheckIn lúc {time_str}")

    with open(ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Employee ID",
                "Full Name",
                "Department",
                "Position",
                "Date",
                "CheckIn",
                "CheckOut",
                "LivenessResult",
                "LivenessScore",
                "Note"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)





