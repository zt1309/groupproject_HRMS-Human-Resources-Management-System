"""
Finova Attendance System - Business Logic Services
Xử lý logic nghiệp vụ cho từng bảng
"""

import numpy as np
import json
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from .models import Employee, FaceEmbedding, AttendanceLog, AIInferenceLog, db_manager

class EmployeeService:
    """Service xử lý nhân viên"""
    
    @staticmethod
    def create_employee(session: Session, full_name: str, department_id: int = None, status: str = 'active') -> Employee:
        """Tạo nhân viên mới"""
        employee = Employee(
            full_name=full_name,
            department_id=department_id,
            status=status
        )
        session.add(employee)
        session.commit()
        session.refresh(employee)
        return employee
    
    @staticmethod
    def get_employee_by_id(session: Session, employee_id: int) -> Optional[Employee]:
        """Lấy nhân viên theo ID"""
        return session.query(Employee).filter(Employee.id == employee_id).first()
    
    @staticmethod
    def get_all_employees(session: Session, status: str = None) -> List[Employee]:
        """Lấy tất cả nhân viên"""
        query = session.query(Employee)
        if status:
            query = query.filter(Employee.status == status)
        return query.order_by(Employee.full_name).all()
    
    @staticmethod
    def update_employee_status(session: Session, employee_id: int, status: str) -> bool:
        """Cập nhật trạng thái nhân viên"""
        employee = session.query(Employee).filter(Employee.id == employee_id).first()
        if employee:
            employee.status = status
            session.commit()
            return True
        return False
    
    @staticmethod
    def search_employees(session: Session, keyword: str) -> List[Employee]:
        """Tìm kiếm nhân viên theo tên"""
        return session.query(Employee).filter(
            Employee.full_name.ilike(f'%{keyword}%')
        ).all()

class FaceEmbeddingService:
    """Service xử lý face embeddings"""
    
    @staticmethod
    def save_embedding(session: Session, employee_id: int, embedding_vector: np.ndarray, model_name: str = 'ArcFace_R50') -> FaceEmbedding:
        """Lưu face embedding"""
        # Convert numpy array to bytes
        embedding_bytes = embedding_vector.tobytes()
        
        face_embedding = FaceEmbedding(
            employee_id=employee_id,
            embedding_vector=embedding_bytes,
            model_name=model_name
        )
        session.add(face_embedding)
        session.commit()
        session.refresh(face_embedding)
        return face_embedding
    
    @staticmethod
    def get_employee_embeddings(session: Session, employee_id: int) -> List[FaceEmbedding]:
        """Lấy tất cả embeddings của nhân viên"""
        return session.query(FaceEmbedding).filter(
            FaceEmbedding.employee_id == employee_id
        ).all()
    
    @staticmethod
    def get_all_embeddings(session: Session) -> List[FaceEmbedding]:
        """Lấy tất cả embeddings để so sánh"""
        return session.query(FaceEmbedding).all()
    
    @staticmethod
    def load_embedding_vector(face_embedding: FaceEmbedding) -> np.ndarray:
        """Chuyển bytes thành numpy array"""
        return np.frombuffer(face_embedding.embedding_vector, dtype=np.float32)
    
    @staticmethod
    def find_similar_face(session: Session, query_embedding: np.ndarray, threshold: float = 0.6) -> Optional[Dict]:
        """Tìm khuôn mặt tương tự"""
        all_embeddings = FaceEmbeddingService.get_all_embeddings(session)
        
        best_match = None
        best_score = -1
        
        for face_emb in all_embeddings:
            stored_embedding = FaceEmbeddingService.load_embedding_vector(face_emb)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = {
                    'employee_id': face_emb.employee_id,
                    'similarity': float(similarity),
                    'embedding_id': face_emb.id
                }
        
        return best_match

class AttendanceService:
    """Service xử lý chấm công"""
    
    @staticmethod
    def check_in(session: Session, employee_id: int, camera_id: str = None, recognition_score: float = None) -> AttendanceLog:
        """Chấm công vào"""
        # Kiểm tra đã check-in hôm nay chưa
        today = date.today()
        existing_log = session.query(AttendanceLog).filter(
            and_(
                AttendanceLog.employee_id == employee_id,
                func.date(AttendanceLog.created_at) == today,
                AttendanceLog.check_in_time.isnot(None)
            )
        ).first()
        
        if existing_log:
            # Đã check-in rồi, cập nhật check-out
            existing_log.check_out_time = datetime.utcnow()
            if recognition_score:
                existing_log.recognition_score = recognition_score
            session.commit()
            return existing_log
        else:
            # Check-in mới
            attendance_log = AttendanceLog(
                employee_id=employee_id,
                check_in_time=datetime.utcnow(),
                camera_id=camera_id,
                recognition_score=recognition_score,
                method='face_recognition'
            )
            session.add(attendance_log)
            session.commit()
            session.refresh(attendance_log)
            return attendance_log
    
    @staticmethod
    def get_today_attendance(session: Session) -> List[Dict]:
        """Lấy chấm công hôm nay"""
        today = date.today()
        
        logs = session.query(AttendanceLog, Employee).join(Employee).filter(
            func.date(AttendanceLog.created_at) == today
        ).order_by(desc(AttendanceLog.check_in_time)).all()
        
        result = []
        for log, employee in logs:
            result.append({
                'id': log.id,
                'employee_id': employee.id,
                'employee_name': employee.full_name,
                'check_in_time': log.check_in_time.isoformat() if log.check_in_time else None,
                'check_out_time': log.check_out_time.isoformat() if log.check_out_time else None,
                'recognition_score': log.recognition_score,
                'camera_id': log.camera_id,
                'status': AttendanceService._get_attendance_status(log)
            })
        
        return result
    
    @staticmethod
    def get_employee_attendance(session: Session, employee_id: int, start_date: date = None, end_date: date = None) -> List[AttendanceLog]:
        """Lấy lịch sử chấm công của nhân viên"""
        query = session.query(AttendanceLog).filter(AttendanceLog.employee_id == employee_id)
        
        if start_date:
            query = query.filter(func.date(AttendanceLog.created_at) >= start_date)
        if end_date:
            query = query.filter(func.date(AttendanceLog.created_at) <= end_date)
            
        return query.order_by(desc(AttendanceLog.created_at)).all()
    
    @staticmethod
    def get_attendance_statistics(session: Session, target_date: date = None) -> Dict:
        """Thống kê chấm công"""
        if not target_date:
            target_date = date.today()
        
        # Tổng nhân viên active
        total_employees = session.query(Employee).filter(Employee.status == 'active').count()
        
        # Nhân viên đã chấm công hôm nay
        present_count = session.query(AttendanceLog).filter(
            and_(
                func.date(AttendanceLog.created_at) == target_date,
                AttendanceLog.check_in_time.isnot(None)
            )
        ).count()
        
        # Nhân viên muộn (sau 8:15)
        late_threshold = datetime.combine(target_date, datetime.min.time().replace(hour=8, minute=15))
        late_count = session.query(AttendanceLog).filter(
            and_(
                func.date(AttendanceLog.created_at) == target_date,
                AttendanceLog.check_in_time > late_threshold
            )
        ).count()
        
        return {
            'total_employees': total_employees,
            'present_today': present_count,
            'late_today': late_count,
            'absent_today': total_employees - present_count,
            'attendance_rate': round((present_count / total_employees * 100) if total_employees > 0 else 0, 1)
        }
    
    @staticmethod
    def _get_attendance_status(log: AttendanceLog) -> str:
        """Xác định trạng thái chấm công"""
        if not log.check_in_time:
            return 'absent'
        
        # Kiểm tra muộn (sau 8:15)
        work_start = log.check_in_time.replace(hour=8, minute=15, second=0, microsecond=0)
        if log.check_in_time > work_start:
            return 'late'
        
        return 'on_time'

class AIInferenceService:
    """Service xử lý AI inference logs"""
    
    @staticmethod
    def log_inference(session: Session, model_version: str, input_type: str, inference_time_ms: int, result: Dict = None) -> AIInferenceLog:
        """Ghi log AI inference"""
        log = AIInferenceLog(
            model_version=model_version,
            input_type=input_type,
            inference_time_ms=inference_time_ms,
            result=json.dumps(result) if result else None
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return log
    
    @staticmethod
    def get_inference_stats(session: Session, hours: int = 24) -> Dict:
        """Thống kê AI inference"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        logs = session.query(AIInferenceLog).filter(
            AIInferenceLog.created_at >= since
        ).all()
        
        if not logs:
            return {
                'total_inferences': 0,
                'avg_inference_time': 0,
                'model_usage': {},
                'success_rate': 0
            }
        
        # Thống kê theo model
        model_stats = {}
        total_time = 0
        success_count = 0
        
        for log in logs:
            model = log.model_version
            if model not in model_stats:
                model_stats[model] = {'count': 0, 'avg_time': 0, 'total_time': 0}
            
            model_stats[model]['count'] += 1
            model_stats[model]['total_time'] += log.inference_time_ms
            total_time += log.inference_time_ms
            
            # Đếm success (có result)
            if log.result:
                success_count += 1
        
        # Tính average time cho từng model
        for model in model_stats:
            model_stats[model]['avg_time'] = round(
                model_stats[model]['total_time'] / model_stats[model]['count'], 2
            )
        
        return {
            'total_inferences': len(logs),
            'avg_inference_time': round(total_time / len(logs), 2),
            'model_usage': model_stats,
            'success_rate': round((success_count / len(logs) * 100), 1)
        }
    
    @staticmethod
    def get_recent_logs(session: Session, limit: int = 50) -> List[AIInferenceLog]:
        """Lấy log gần đây"""
        return session.query(AIInferenceLog).order_by(
            desc(AIInferenceLog.created_at)
        ).limit(limit).all()