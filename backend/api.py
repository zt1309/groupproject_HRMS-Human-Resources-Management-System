"""
Finova Attendance System - REST API
API endpoints theo database schema
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import cv2
import time
import os
import csv
from datetime import datetime, date
import logging
from typing import Dict, Any

# Import services
from .models import db_manager
from .services import EmployeeService, FaceEmbeddingService, AttendanceService, AIInferenceService

# Import AI modules
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.detect_faces import detect_and_crop_faces
    from src.extract_embeddings import get_embedding
    from src.antispoof import check_liveness
    print("✅ AI modules loaded successfully!")
except ImportError as e:
    print(f"❌ Lỗi import AI modules: {e}")
    detect_and_crop_faces = None
    get_embedding = None
    check_liveness = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize database
db_manager.create_tables()

@app.route('/')
def index():
    """API Documentation"""
    return jsonify({
        "name": "Finova Attendance System API v2.0",
        "description": "Professional database-driven attendance system",
        "database_schema": {
            "employees": "Employee information",
            "face_embeddings": "Face recognition vectors", 
            "attendance_logs": "Check-in/out records",
            "ai_inference_logs": "AI model performance logs"
        },
        "endpoints": {
            "employees": {
                "GET /api/employees": "List all employees",
                "POST /api/employees": "Create new employee",
                "GET /api/employees/{id}": "Get employee details",
                "PUT /api/employees/{id}": "Update employee",
                "DELETE /api/employees/{id}": "Delete employee"
            },
            "attendance": {
                "GET /api/attendance/today": "Today's attendance",
                "GET /api/attendance/employee/{id}": "Employee attendance history",
                "POST /api/attendance/checkin": "Check-in via face recognition",
                "GET /api/attendance/statistics": "Attendance statistics"
            },
            "face_recognition": {
                "POST /api/face/enroll": "Enroll new face",
                "POST /api/face/recognize": "Recognize face",
                "GET /api/face/embeddings/{employee_id}": "Get employee embeddings"
            },
            "ai_logs": {
                "GET /api/ai/logs": "AI inference logs",
                "GET /api/ai/stats": "AI performance statistics",
                "GET /api/ai/test": "Test AI modules"
            },
            "snapshots": {
                "GET /api/snapshots": "List all snapshots",
                "GET /api/snapshots/employee/{id}": "Get employee snapshots",
                "GET /api/snapshots/{employee_id}/{filename}": "Serve snapshot file",
                "POST /api/snapshots/cleanup": "Cleanup old snapshots"
            }
        }
    })

# ==================== EMPLOYEE ENDPOINTS ====================

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Lấy danh sách nhân viên"""
    session = db_manager.get_session()
    try:
        status = request.args.get('status', 'active')
        employees = EmployeeService.get_all_employees(session, status)
        
        return jsonify({
            "success": True,
            "count": len(employees),
            "employees": [emp.to_dict() for emp in employees]
        })
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/employees', methods=['POST'])
def create_employee():
    """Tạo nhân viên mới"""
    session = db_manager.get_session()
    try:
        data = request.get_json()
        
        if not data or not data.get('full_name'):
            return jsonify({"error": "full_name is required"}), 400
        
        employee = EmployeeService.create_employee(
            session=session,
            full_name=data['full_name'],
            department_id=data.get('department_id'),
            status=data.get('status', 'active')
        )
        
        return jsonify({
            "success": True,
            "employee": employee.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    """Lấy thông tin nhân viên"""
    session = db_manager.get_session()
    try:
        employee = EmployeeService.get_employee_by_id(session, employee_id)
        
        if not employee:
            return jsonify({"error": "Employee not found"}), 404
        
        return jsonify({
            "success": True,
            "employee": employee.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting employee {employee_id}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/employees/search', methods=['GET'])
def search_employees():
    """Tìm kiếm nhân viên"""
    session = db_manager.get_session()
    try:
        keyword = request.args.get('q', '')
        if not keyword:
            return jsonify({"error": "Search keyword required"}), 400
        
        employees = EmployeeService.search_employees(session, keyword)
        
        return jsonify({
            "success": True,
            "count": len(employees),
            "employees": [emp.to_dict() for emp in employees]
        })
        
    except Exception as e:
        logger.error(f"Error searching employees: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

# ==================== ATTENDANCE ENDPOINTS ====================

@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    """Lấy chấm công hôm nay"""
    session = db_manager.get_session()
    try:
        attendance_data = AttendanceService.get_today_attendance(session)
        
        return jsonify({
            "success": True,
            "date": date.today().isoformat(),
            "count": len(attendance_data),
            "attendance": attendance_data
        })
        
    except Exception as e:
        logger.error(f"Error getting today attendance: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/attendance/statistics', methods=['GET'])
def get_attendance_statistics():
    """Thống kê chấm công"""
    session = db_manager.get_session()
    try:
        target_date = request.args.get('date')
        if target_date:
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            target_date = date.today()
        
        stats = AttendanceService.get_attendance_statistics(session, target_date)
        
        return jsonify({
            "success": True,
            "date": target_date.isoformat(),
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance statistics: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/attendance/employee/<int:employee_id>', methods=['GET'])
def get_employee_attendance(employee_id):
    """Lấy lịch sử chấm công nhân viên"""
    session = db_manager.get_session()
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        logs = AttendanceService.get_employee_attendance(session, employee_id, start_date, end_date)
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "count": len(logs),
            "attendance": [log.to_dict() for log in logs]
        })
        
    except Exception as e:
        logger.error(f"Error getting employee attendance: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

# ==================== FACE RECOGNITION ENDPOINTS ====================

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    """Nhận diện khuôn mặt và chấm công"""
    if not all([detect_and_crop_faces, get_embedding, check_liveness]):
        return jsonify({"error": "AI modules not available"}), 503
    
    session = db_manager.get_session()
    start_time = time.time()
    
    try:
        # Kiểm tra file ảnh
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400
        
        # Đọc ảnh
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Log AI inference - Face Detection
        detect_start = time.time()
        faces = detect_and_crop_faces(frame)
        detect_time = int((time.time() - detect_start) * 1000)
        
        AIInferenceService.log_inference(
            session, 'YOLOv8_Face', 'image', detect_time,
            {'faces_detected': len(faces)}
        )
        
        if not faces:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "No face detected"
            })
        
        # Lấy khuôn mặt lớn nhất
        largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
        
        # Anti-spoofing check
        spoof_start = time.time()
        is_real = check_liveness(largest_face)
        spoof_time = int((time.time() - spoof_start) * 1000)
        
        AIInferenceService.log_inference(
            session, 'AntiSpoof_ResNet18', 'image', spoof_time,
            {'is_real': is_real}
        )
        
        if not is_real:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Spoofing detected",
                "anti_spoof": False
            })
        
        # Face embedding extraction
        embed_start = time.time()
        embedding = get_embedding(largest_face)
        embed_time = int((time.time() - embed_start) * 1000)
        
        if embedding is None:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Failed to extract face embedding"
            })
        
        AIInferenceService.log_inference(
            session, 'ArcFace_R50', 'image', embed_time,
            {'embedding_extracted': True}
        )
        
        # Tìm khuôn mặt tương tự
        match_start = time.time()
        match_result = FaceEmbeddingService.find_similar_face(session, embedding, threshold=0.6)
        match_time = int((time.time() - match_start) * 1000)
        
        if not match_result:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Face not recognized",
                "anti_spoof": True
            })
        
        # Lấy thông tin nhân viên
        employee = EmployeeService.get_employee_by_id(session, match_result['employee_id'])
        if not employee:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "Employee not found in database"
            })
        
        # Chấm công
        camera_id = request.form.get('camera_id', 'web_camera')
        attendance_log = AttendanceService.check_in(
            session, employee.id, camera_id, match_result['similarity']
        )
        
        # Lưu snapshot cho admin/HR kiểm tra
        snapshot_path = None
        try:
            snapshot_dir = f"snapshots/{employee.id}"
            os.makedirs(snapshot_dir, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"snapshot_{timestamp}.jpg"
            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
            
            cv2.imwrite(snapshot_path, largest_face)
            logger.info(f"Saved snapshot: {snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
        
        # Tổng thời gian xử lý
        total_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            "success": True,
            "recognized": True,
            "employee_id": employee.id,
            "employee_name": employee.full_name,
            "similarity": match_result['similarity'],
            "anti_spoof": True,
            "attendance_log_id": attendance_log.id,
            "check_in_time": attendance_log.check_in_time.isoformat() if attendance_log.check_in_time else None,
            "check_out_time": attendance_log.check_out_time.isoformat() if attendance_log.check_out_time else None,
            "snapshot_path": snapshot_path,
            "processing_time_ms": total_time,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    """One-to-one face verification cho nhân viên quan trọng"""
    if not all([detect_and_crop_faces, get_embedding, check_liveness]):
        return jsonify({"error": "AI modules not available"}), 503
    
    session = db_manager.get_session()
    start_time = time.time()
    
    try:
        # Lấy employee_id từ form data
        employee_id = request.form.get('employee_id')
        if not employee_id:
            return jsonify({"error": "employee_id required"}), 400
        
        employee_id = int(employee_id)
        
        # Kiểm tra file ảnh
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400
        
        # Đọc ảnh
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Detect faces
        faces = detect_and_crop_faces(frame)
        if not faces:
            return jsonify({
                "success": False,
                "verified": False,
                "message": "No face detected"
            })
        
        largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
        
        # Anti-spoofing check
        is_real = check_liveness(largest_face)
        if not is_real:
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Spoofing detected",
                "anti_spoof": False
            })
        
        # Extract embedding
        embedding = get_embedding(largest_face)
        if embedding is None:
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Failed to extract face embedding"
            })
        
        # Lấy embeddings của nhân viên cần verify
        employee_embeddings = FaceEmbeddingService.get_employee_embeddings(session, employee_id)
        if not employee_embeddings:
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Employee not enrolled in face recognition system"
            })
        
        # So sánh với embedding của nhân viên cụ thể
        best_similarity = 0.0
        for face_emb in employee_embeddings:
            stored_embedding = FaceEmbeddingService.load_embedding_vector(face_emb)
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
        
        # Threshold cao hơn cho one-to-one verification
        verified = best_similarity >= 0.7
        
        # Lấy thông tin nhân viên
        employee = EmployeeService.get_employee_by_id(session, employee_id)
        
        # Log verification attempt
        AIInferenceService.log_inference(
            session, 'One_to_One_Verification', 'image', 
            int((time.time() - start_time) * 1000),
            {
                'employee_id': employee_id,
                'verified': verified,
                'similarity': float(best_similarity)
            }
        )
        
        # Log to access_logs.csv for compatibility
        try:
            import csv
            access_log_path = "db/access_logs.csv"
            
            # Create file with header if not exists
            if not os.path.exists(access_log_path):
                with open(access_log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Time', 'Employee ID', 'Name', 'Status', 'Liveness'])
            
            # Append log
            with open(access_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    date.today().strftime('%Y-%m-%d'),
                    datetime.utcnow().strftime('%H:%M:%S'),
                    f' {employee_id}',  # Space prefix for consistency with existing data
                    employee.full_name if employee else 'Unknown',
                    'Granted' if verified else 'Denied',
                    'Real' if is_real else 'Fake'
                ])
            logger.info(f"Logged verification to access_logs.csv: {employee_id} - {'Granted' if verified else 'Denied'}")
        except Exception as e:
            logger.warning(f"Failed to log to access_logs.csv: {e}")
        
        return jsonify({
            "success": True,
            "verified": verified,
            "employee_id": employee_id,
            "employee_name": employee.full_name if employee else "Unknown",
            "similarity": float(best_similarity),
            "anti_spoof": True,
            "threshold": 0.7,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in face verification: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/face/enroll', methods=['POST'])
def enroll_face():
    """Enroll khuôn mặt mới"""
    if not all([detect_and_crop_faces, get_embedding]):
        return jsonify({"error": "AI modules not available"}), 503
    
    session = db_manager.get_session()
    try:
        # Lấy thông tin
        employee_id = request.form.get('employee_id')
        if not employee_id:
            return jsonify({"error": "employee_id required"}), 400
        
        employee_id = int(employee_id)
        
        # Kiểm tra nhân viên tồn tại
        employee = EmployeeService.get_employee_by_id(session, employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404
        
        # Xử lý nhiều ảnh
        embeddings = []
        processed_photos = 0
        
        for i in range(30):  # Tối đa 30 ảnh
            photo_key = f'photo_{i}'
            if photo_key in request.files:
                photo = request.files[photo_key]
                if photo.filename:
                    # Đọc ảnh
                    image_bytes = photo.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Detect face
                        faces = detect_and_crop_faces(image)
                        if faces:
                            # Lấy khuôn mặt lớn nhất
                            largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
                            
                            # Extract embedding
                            embedding = get_embedding(largest_face)
                            if embedding is not None:
                                embeddings.append(embedding)
                                processed_photos += 1
        
        if not embeddings:
            return jsonify({"error": "No valid face found in uploaded photos"}), 400
        
        # Tính embedding trung bình
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Lưu vào database
        face_embedding = FaceEmbeddingService.save_embedding(
            session, employee_id, avg_embedding, 'ArcFace_R50'
        )
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee.full_name,
            "embedding_id": face_embedding.id,
            "photos_processed": processed_photos,
            "embeddings_count": len(embeddings),
            "message": "Face enrolled successfully"
        })
        
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/face/embeddings/<int:employee_id>', methods=['GET'])
def get_employee_embeddings(employee_id):
    """Lấy embeddings của nhân viên"""
    session = db_manager.get_session()
    try:
        embeddings = FaceEmbeddingService.get_employee_embeddings(session, employee_id)
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "count": len(embeddings),
            "embeddings": [emb.to_dict() for emb in embeddings]
        })
        
    except Exception as e:
        logger.error(f"Error getting employee embeddings: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

# ==================== AI LOGS ENDPOINTS ====================

@app.route('/api/ai/logs', methods=['GET'])
def get_ai_logs():
    """Lấy AI inference logs"""
    session = db_manager.get_session()
    try:
        limit = int(request.args.get('limit', 50))
        logs = AIInferenceService.get_recent_logs(session, limit)
        
        return jsonify({
            "success": True,
            "count": len(logs),
            "logs": [log.to_dict() for log in logs]
        })
        
    except Exception as e:
        logger.error(f"Error getting AI logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/ai/stats', methods=['GET'])
def get_ai_stats():
    """Thống kê AI performance"""
    session = db_manager.get_session()
    try:
        hours = int(request.args.get('hours', 24))
        stats = AIInferenceService.get_inference_stats(session, hours)
        
        return jsonify({
            "success": True,
            "period_hours": hours,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting AI stats: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db_manager.close_session(session)

@app.route('/api/ai/test', methods=['GET'])
def test_ai_modules():
    """Test tất cả AI modules"""
    try:
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "modules": {}
        }
        
        # Test YOLOv8
        try:
            if detect_and_crop_faces is not None:
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                faces = detect_and_crop_faces(test_image)
                results["modules"]["yolov8"] = {
                    "status": "OK",
                    "message": f"Loaded successfully, detected {len(faces)} faces in test image"
                }
            else:
                results["modules"]["yolov8"] = {
                    "status": "ERROR", 
                    "message": "Module not loaded"
                }
        except Exception as e:
            results["modules"]["yolov8"] = {
                "status": "ERROR",
                "message": str(e)
            }
        
        # Test ArcFace
        try:
            if get_embedding is not None:
                test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                embedding = get_embedding(test_face)
                if embedding is not None:
                    results["modules"]["arcface"] = {
                        "status": "OK",
                        "message": f"Loaded successfully, embedding shape: {embedding.shape}"
                    }
                else:
                    results["modules"]["arcface"] = {
                        "status": "ERROR",
                        "message": "Failed to extract embedding"
                    }
            else:
                results["modules"]["arcface"] = {
                    "status": "ERROR",
                    "message": "Module not loaded"
                }
        except Exception as e:
            results["modules"]["arcface"] = {
                "status": "ERROR", 
                "message": str(e)
            }
        
        # Test Anti-spoofing
        try:
            if check_liveness is not None:
                test_face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
                is_real = check_liveness(test_face)
                results["modules"]["antispoof"] = {
                    "status": "OK",
                    "message": f"Loaded successfully, test result: {is_real}"
                }
            else:
                results["modules"]["antispoof"] = {
                    "status": "ERROR",
                    "message": "Module not loaded"
                }
        except Exception as e:
            results["modules"]["antispoof"] = {
                "status": "ERROR",
                "message": str(e)
            }
        
        # Check model files
        model_files = {
            "yolov8_model": "models/yolov8n-face-lindevs.pt",
            "arcface_model": "models/w600k_r50.onnx",
            "antispoof_model": "models/antispoof_resnet18.pt"
        }
        
        results["model_files"] = {}
        for name, path in model_files.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                results["model_files"][name] = {
                    "status": "OK",
                    "path": path,
                    "size_mb": round(size_mb, 1)
                }
            else:
                results["model_files"][name] = {
                    "status": "MISSING",
                    "path": path
                }
        
        return jsonify({
            "success": True,
            "test_results": results
        })
        
    except Exception as e:
        logger.error(f"Error testing AI modules: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== WEB FILES SERVING ====================

@app.route('/web/<path:filename>')
def serve_web_files(filename):
    """Phục vụ file web"""
    import os
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')
    return send_from_directory(web_dir, filename)

# ==================== AVATAR IMAGES SERVING ====================

@app.route('/api/employees/<int:employee_id>/avatar')
def get_employee_avatar(employee_id):
    """Serve avatar image của nhân viên"""
    try:
        # Thử tìm avatar.jpg trong thư mục enrollment
        avatar_path = f"data/employees/{employee_id}/avatar.jpg"
        
        if os.path.exists(avatar_path):
            from flask import send_file
            return send_file(avatar_path, mimetype='image/jpeg')
        
        # Nếu không có avatar, tìm ảnh đầu tiên trong thư mục
        employee_dir = f"data/employees/{employee_id}"
        if os.path.exists(employee_dir):
            files = [f for f in os.listdir(employee_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                first_image = os.path.join(employee_dir, sorted(files)[0])
                from flask import send_file
                return send_file(first_image, mimetype='image/jpeg')
        
        # Fallback to placeholder
        placeholder_path = "web/assets/placeholder-avatar.svg"
        if os.path.exists(placeholder_path):
            from flask import send_file
            return send_file(placeholder_path, mimetype='image/svg+xml')
        
        return jsonify({"error": "Avatar not found"}), 404
        
    except Exception as e:
        logger.error(f"Error serving avatar for employee {employee_id}: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== SNAPSHOTS ENDPOINTS ====================

@app.route('/api/snapshots', methods=['GET'])
def get_snapshots():
    """Lấy danh sách snapshots"""
    try:
        snapshots_dir = "snapshots"
        if not os.path.exists(snapshots_dir):
            return jsonify({
                "success": True,
                "snapshots": [],
                "message": "No snapshots directory found"
            })
        
        snapshots = []
        for employee_dir in os.listdir(snapshots_dir):
            employee_path = os.path.join(snapshots_dir, employee_dir)
            if os.path.isdir(employee_path):
                for filename in os.listdir(employee_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(employee_path, filename)
                        file_stat = os.stat(file_path)
                        
                        snapshots.append({
                            "employee_id": employee_dir,
                            "filename": filename,
                            "path": file_path.replace('\\', '/'),
                            "size": file_stat.st_size,
                            "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                            "url": f"/api/snapshots/{employee_dir}/{filename}"
                        })
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        snapshots.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(snapshots),
            "snapshots": snapshots
        })
        
    except Exception as e:
        logger.error(f"Error getting snapshots: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/snapshots/<employee_id>/<filename>')
def serve_snapshot(employee_id, filename):
    """Phục vụ file snapshot"""
    try:
        snapshots_dir = os.path.join("snapshots", employee_id)
        return send_from_directory(snapshots_dir, filename)
    except Exception as e:
        logger.error(f"Error serving snapshot: {e}")
        return jsonify({"error": "Snapshot not found"}), 404

@app.route('/api/snapshots/employee/<int:employee_id>', methods=['GET'])
def get_employee_snapshots(employee_id):
    """Lấy snapshots của nhân viên cụ thể"""
    try:
        employee_dir = os.path.join("snapshots", str(employee_id))
        if not os.path.exists(employee_dir):
            return jsonify({
                "success": True,
                "employee_id": employee_id,
                "snapshots": [],
                "message": "No snapshots found for this employee"
            })
        
        snapshots = []
        for filename in os.listdir(employee_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(employee_dir, filename)
                file_stat = os.stat(file_path)
                
                snapshots.append({
                    "filename": filename,
                    "path": file_path.replace('\\', '/'),
                    "size": file_stat.st_size,
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "url": f"/api/snapshots/{employee_id}/{filename}"
                })
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        snapshots.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "count": len(snapshots),
            "snapshots": snapshots
        })
        
    except Exception as e:
        logger.error(f"Error getting employee snapshots: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/snapshots/date/<date_str>', methods=['GET'])
def get_snapshots_by_date(date_str):
    """Lấy snapshots theo ngày (YYYY-MM-DD)"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        snapshots_dir = "snapshots"
        if not os.path.exists(snapshots_dir):
            return jsonify({
                "success": True,
                "date": date_str,
                "snapshots": [],
                "message": "No snapshots directory found"
            })
        
        snapshots = []
        for employee_dir in os.listdir(snapshots_dir):
            employee_path = os.path.join(snapshots_dir, employee_dir)
            if os.path.isdir(employee_path):
                for filename in os.listdir(employee_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(employee_path, filename)
                        file_stat = os.stat(file_path)
                        file_date = datetime.fromtimestamp(file_stat.st_ctime).date()
                        
                        if file_date == target_date:
                            snapshots.append({
                                "employee_id": employee_dir,
                                "filename": filename,
                                "path": file_path.replace('\\', '/'),
                                "size": file_stat.st_size,
                                "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                                "url": f"/api/snapshots/{employee_dir}/{filename}"
                            })
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        snapshots.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            "success": True,
            "date": date_str,
            "count": len(snapshots),
            "snapshots": snapshots
        })
        
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        logger.error(f"Error getting snapshots by date: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/snapshots/cleanup', methods=['POST'])
def cleanup_snapshots():
    """Dọn dẹp snapshots cũ"""
    try:
        data = request.get_json() or {}
        days_old = data.get('days_old', 30)  # Xóa snapshots cũ hơn 30 ngày
        
        snapshots_dir = "snapshots"
        if not os.path.exists(snapshots_dir):
            return jsonify({
                "success": True,
                "message": "No snapshots directory found",
                "deleted_count": 0
            })
        
        deleted_count = 0
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for employee_dir in os.listdir(snapshots_dir):
            employee_path = os.path.join(snapshots_dir, employee_dir)
            if os.path.isdir(employee_path):
                for filename in os.listdir(employee_path):
                    file_path = os.path.join(employee_path, filename)
                    if os.path.isfile(file_path):
                        file_stat = os.stat(file_path)
                        if file_stat.st_ctime < cutoff_time:
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"Deleted old snapshot: {file_path}")
                
                # Xóa thư mục rỗng
                if not os.listdir(employee_path):
                    os.rmdir(employee_path)
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up snapshots older than {days_old} days",
            "deleted_count": deleted_count
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up snapshots: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Tạo thư mục cần thiết
    os.makedirs('logs', exist_ok=True)
    os.makedirs('snapshots', exist_ok=True)
    os.makedirs('db', exist_ok=True)
    
    print("� FINOVyA ATTENDANCE SYSTEM - DATABASE API")
    print("📊 Database Schema: employees, face_embeddings, attendance_logs, ai_inference_logs")
    print("🌐 API Server: http://localhost:5000")
    print("📖 API Docs: http://localhost:5000/")
    print("🌐 Web Interface: http://localhost:5000/web/modern-index.html")
    print("👤 Employee Enrollment: http://localhost:5000/web/enroll.html")
    print("📸 Snapshots: Enabled for attendance verification")
    
    app.run(host='0.0.0.0', port=5000, debug=True)