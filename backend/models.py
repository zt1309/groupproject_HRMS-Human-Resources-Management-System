"""
Finova Attendance System - Database Models
Schema theo thiết kế của bạn
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class Employee(Base):
    """Bảng employees - Thông tin nhân viên"""
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    department_id = Column(Integer, nullable=True)  # Có thể link với bảng departments
    status = Column(String(50), default='active')  # active, inactive, terminated
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="employee")
    attendance_logs = relationship("AttendanceLog", back_populates="employee")
    
    def to_dict(self):
        return {
            'id': self.id,
            'full_name': self.full_name,
            'department_id': self.department_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class FaceEmbedding(Base):
    """Bảng face_embeddings - Vector đặc trưng khuôn mặt"""
    __tablename__ = 'face_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)  # Lưu numpy array dạng bytes
    model_name = Column(String(100), default='ArcFace_R50')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="face_embeddings")
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class AttendanceLog(Base):
    """Bảng attendance_logs - Log chấm công"""
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    camera_id = Column(String(50), nullable=True)  # ID camera thực hiện chấm công
    recognition_score = Column(Float, nullable=True)  # Độ tin cậy nhận diện
    method = Column(String(50), default='face_recognition')  # face_recognition, manual, card
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_logs")
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'check_in_time': self.check_in_time.isoformat() if self.check_in_time else None,
            'check_out_time': self.check_out_time.isoformat() if self.check_out_time else None,
            'camera_id': self.camera_id,
            'recognition_score': self.recognition_score,
            'method': self.method,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class AIInferenceLog(Base):
    """Bảng ai_inference_logs - Log các lần chạy AI"""
    __tablename__ = 'ai_inference_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(100), nullable=False)  # YOLOv8, ArcFace, AntiSpoof
    input_type = Column(String(50), nullable=False)  # image, video, stream
    inference_time_ms = Column(Integer, nullable=False)  # Thời gian xử lý (ms)
    result = Column(Text, nullable=True)  # JSON result
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_version': self.model_version,
            'input_type': self.input_type,
            'inference_time_ms': self.inference_time_ms,
            'result': json.loads(self.result) if self.result else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Database connection
class DatabaseManager:
    def __init__(self, database_url="sqlite:///finova_attendance.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Tạo tất cả bảng"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ Database tables created successfully!")
        
    def get_session(self):
        """Lấy database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Đóng session"""
        session.close()

# Singleton instance
db_manager = DatabaseManager()