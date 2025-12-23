#!/usr/bin/env python3
"""
Test script để kiểm tra tất cả AI modules
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

def test_ai_modules():
    """Test tất cả AI modules"""
    print(" TESTING AI MODULES")
    print("=" * 50)
    
    # Test 1: YOLOv8 Face Detection
    print("\n1️ Testing YOLOv8 Face Detection...")
    try:
        from src.detect_faces import detect_and_crop_faces, detect_faces
        
        # Tạo ảnh test đơn giản
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)
        
        faces = detect_and_crop_faces(test_image)
        boxes = detect_faces(test_image)
        
        print(f"    YOLOv8 loaded successfully")
        print(f"    Detected {len(faces)} faces, {len(boxes)} boxes")
        
    except Exception as e:
        print(f"    YOLOv8 Error: {e}")
    
    # Test 2: ArcFace Embedding
    print("\n2️ Testing ArcFace Embedding...")
    try:
        from src.extract_embeddings import get_embedding
        
        # Tạo ảnh khuôn mặt giả
        face_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        embedding = get_embedding(face_img)
        
        if embedding is not None:
            print(f"    ArcFace loaded successfully")
            print(f"    Embedding shape: {embedding.shape}")
            print(f"    Embedding norm: {np.linalg.norm(embedding):.3f}")
        else:
            print(f"    ArcFace failed to extract embedding")
            
    except Exception as e:
        print(f"    ArcFace Error: {e}")
    
    # Test 3: Anti-spoofing
    print("\n3️ Testing Anti-spoofing...")
    try:
        from src.antispoof import check_liveness
        
        # Tạo ảnh test
        face_img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        
        is_real = check_liveness(face_img)
        
        print(f"    Anti-spoofing loaded successfully")
        print(f"    Liveness result: {is_real}")
        
    except Exception as e:
        print(f"    Anti-spoofing Error: {e}")
    
    # Test 4: Face Recognition
    print("\n Testing Face Recognition...")
    try:
        from src.recognize import recognize, load_db
        
        # Load database
        db = load_db()
        print(f"    Database loaded: {len(db)} employees")
        
        # Test recognition với ảnh giả
        face_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emp_id, name = recognize(face_img, threshold=0.5)
        
        print(f"    Recognition module loaded successfully")
        print(f"    Recognition result: ID={emp_id}, Name={name}")
        
    except Exception as e:
        print(f"    Recognition Error: {e}")
    
    # Test 5: Model Files
    print("\n Checking Model Files...")
    model_files = [
        "models/yolov8n-face-lindevs.pt",
        "models/w600k_r50.onnx", 
        "models/antispoof_resnet18.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"   ✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {model_file} - NOT FOUND")
    
    # Test 6: Database Files
    print("\n6️⃣ Checking Database Files...")
    db_files = [
        "db/data_employee.csv",
        "db/important_employee.csv", 
        "db/employees.json",
        "logs/attendance.csv"
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"   ✅ {db_file}")
        else:
            print(f"   ⚠️ {db_file} - NOT FOUND (will be created)")
    
    print("\n" + "=" * 50)
    print("🎯 AI MODULES TEST COMPLETED")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_ai_modules()