# Finova Attendance System - Database Structure

## 📁 File Database Structure

### 1. **CSV Files** (db/)

#### `db/data_employee.csv` - Toàn bộ nhân viên công ty
- **Mục đích**: Database chính cho ONE-TO-MANY recognition
- **Sử dụng**: Chấm công hàng ngày cho tất cả nhân viên
- **Cấu trúc**:
  ```csv
  Employee ID, Full Name, Position, Department, Date of Birth, Phone/ID, Username, Password
  ```
- **Số lượng**: 200+ nhân viên
- **API endpoint**: `GET /api/employees`

#### `db/important_employee.csv` - Nhân viên quan trọng
- **Mục đích**: Database cho ONE-TO-ONE verification
- **Sử dụng**: Xác thực danh tính cho lãnh đạo, quản lý cấp cao
- **Cấu trúc**: Giống `data_employee.csv`
- **Số lượng**: ~60 nhân viên (lãnh đạo, trưởng phòng, phó phòng)
- **API endpoint**: `GET /api/employees/important`

#### `db/access_logs.csv` - Log truy cập (OUTPUT của verify)
- **Mục đích**: Ghi lại kết quả ONE-TO-ONE verification
- **Sử dụng**: Audit trail cho nhân viên quan trọng
- **Cấu trúc**:
  ```csv
  Date, Time, Employee ID, Name, Status, Liveness
  ```
- **Status**: `Granted` (thành công) hoặc `Denied` (thất bại)
- **Liveness**: `Real` hoặc `Fake`
- **API endpoint**: `GET /api/access-logs`

#### `logs/attendance.csv` - Log chấm công (OUTPUT của recognize)
- **Mục đích**: Ghi lại chấm công hàng ngày
- **Sử dụng**: Báo cáo chấm công, tính lương
- **Cấu trúc**:
  ```csv
  Date, Employee ID, Full Name, Department, CheckIn, CheckOut, Status
  ```
- **Status**: `on-time`, `late`, `absent`
- **API endpoint**: `GET /api/attendance/today`

### 2. **JSON Files** (db/)

#### `db/employees.json` - Face embeddings toàn bộ nhân viên
- **Mục đích**: Lưu trữ face embeddings cho ONE-TO-MANY
- **Sử dụng**: So sánh khuôn mặt với toàn bộ database
- **Cấu trúc**:
  ```json
  {
    "250": {
      "name": "Trương Việt Đông",
      "department": "Members' Council",
      "position": "Chairman of Members' Council",
      "embedding": [512 float values],
      "avatar": "data/employees/250/avatar.jpg",
      "enrolled_date": "2025-12-07T22:26:47",
      "photos_count": 30
    }
  }
  ```
- **Embedding**: 512-dimensional ArcFace vector
- **Được tạo bởi**: `src/enroll.py` hoặc `/api/enroll`

#### `db/important_employees.json` - Face embeddings nhân viên quan trọng
- **Mục đích**: Lưu trữ face embeddings cho ONE-TO-ONE
- **Sử dụng**: Verify danh tính cụ thể
- **Cấu trúc**: Giống `employees.json`
- **Subset**: Chỉ chứa nhân viên trong `important_employee.csv`

### 3. **Image Folders**

#### `data/employees/{employee_id}/` - Ảnh enrollment
- **Mục đích**: Lưu 30 ảnh khuôn mặt khi nhân viên đăng ký
- **Sử dụng**: Training data, backup, re-enrollment
- **Cấu trúc**:
  ```
  data/employees/250/
  ├── 250_20251207_222634_805580.jpg  (frame 1)
  ├── 250_20251207_222635_112867.jpg  (frame 2)
  ├── ...
  ├── 250_20251207_222647_175645.jpg  (frame 30)
  └── avatar.jpg                       (ảnh đại diện)
  ```
- **avatar.jpg**: Ảnh đại diện hiển thị trên web interface
- **Naming**: `{employee_id}_{timestamp}.jpg`

#### `snapshots/{employee_id}/` - Ảnh chấm công
- **Mục đích**: Lưu ảnh khuôn mặt mỗi lần chấm công
- **Sử dụng**: Admin/HR kiểm tra, audit trail
- **Cấu trúc**:
  ```
  snapshots/250/
  ├── snapshot_20251223_080512.jpg  (check-in sáng)
  ├── snapshot_20251223_173045.jpg  (check-out chiều)
  └── ...
  ```
- **Naming**: `snapshot_{timestamp}.jpg`
- **API endpoint**: `GET /api/snapshots`

## 🔄 Data Flow

### ONE-TO-MANY Recognition (Chấm công hàng ngày)
```
1. Camera capture → YOLOv8 detect face
2. Face → ArcFace extract embedding
3. Embedding → Compare với db/employees.json
4. Match found → Log vào logs/attendance.csv
5. Save snapshot → snapshots/{employee_id}/
```

### ONE-TO-ONE Verification (Xác thực lãnh đạo)
```
1. Input: employee_id + camera capture
2. YOLOv8 detect face → ArcFace extract embedding
3. Compare với db/important_employees.json[employee_id]
4. Verify match → Log vào db/access_logs.csv
5. Save snapshot → snapshots/{employee_id}/
```

### Enrollment (Đăng ký nhân viên mới)
```
1. Capture 30 photos → Save to data/employees/{id}/
2. Extract 30 embeddings → Average embedding
3. Save to db/employees.json
4. If important → Also save to db/important_employees.json
5. Update db/data_employee.csv
```

## 🔗 Backend API Integration

### Current Issues:
1. ❌ Backend API (SQLite) chưa đọc từ CSV files
2. ❌ Avatar images không hiển thị trên web
3. ❌ Chưa có endpoint để serve avatar images
4. ❌ Verify endpoint chưa log vào access_logs.csv

### Required Fixes:

#### 1. Load CSV data vào SQLite database
```python
# Cần migration script để import CSV → SQLite
def migrate_csv_to_database():
    # Import data_employee.csv → employees table
    # Import employees.json → face_embeddings table
    # Import attendance.csv → attendance_logs table
```

#### 2. Serve avatar images
```python
@app.route('/api/employees/<int:employee_id>/avatar')
def get_employee_avatar(employee_id):
    avatar_path = f"data/employees/{employee_id}/avatar.jpg"
    if os.path.exists(avatar_path):
        return send_file(avatar_path)
    return send_file("web/assets/placeholder-avatar.svg")
```

#### 3. Update verify endpoint to log access
```python
@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    # ... existing verification code ...
    
    # Log to access_logs.csv
    with open('db/access_logs.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            date.today().strftime('%Y-%m-%d'),
            datetime.now().strftime('%H:%M:%S'),
            employee_id,
            employee_name,
            'Granted' if verified else 'Denied',
            'Real' if is_real else 'Fake'
        ])
```

#### 4. Dual system support
```python
# Backend API nên hỗ trợ cả 2:
# 1. SQLite database (mới, professional)
# 2. CSV/JSON files (legacy, compatibility)

# Config để chọn mode
USE_DATABASE = True  # or False for CSV mode
```

## 📊 Web Interface Avatar Display

### Current Problem:
- Avatar không hiển thị vì path không đúng
- Web đang dùng: `assets/placeholder-avatar.svg`
- Cần dùng: `/api/employees/{id}/avatar`

### Fix:
```javascript
// web/modern-index.html
function displayEmployee(employee) {
    const avatarUrl = `/api/employees/${employee.employee_id}/avatar`;
    return `
        <img src="${avatarUrl}" 
             alt="${employee.name}"
             onerror="this.src='assets/placeholder-avatar.svg'">
    `;
}
```

## ✅ Summary

### File Purposes:
- **data_employee.csv**: INPUT cho one-to-many (toàn công ty)
- **important_employee.csv**: INPUT cho one-to-one (lãnh đạo)
- **access_logs.csv**: OUTPUT của verify (one-to-one)
- **attendance.csv**: OUTPUT của recognize (one-to-many)
- **employees.json**: Embeddings cho one-to-many
- **important_employees.json**: Embeddings cho one-to-one
- **data/employees/**: Ảnh enrollment (30 frames + avatar)
- **snapshots/**: Ảnh chấm công (audit trail)

### Next Steps:
1. ✅ Thêm endpoint serve avatar images
2. ✅ Update web interface để hiển thị avatar
3. ✅ Thêm CSV logging vào verify endpoint
4. ⏳ Tạo migration script CSV → SQLite
5. ⏳ Dual mode support (Database + CSV)
