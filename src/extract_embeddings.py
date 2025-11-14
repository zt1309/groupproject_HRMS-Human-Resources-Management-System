# src/extract_embeddings.py
import cv2
import numpy as np
import onnxruntime as ort
import os

# === Load ArcFace model ===
MODEL_PATH = os.path.join("models", "w600k_r50.onnx")


session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess_face(face_img):
    """
    Chuẩn hoá ảnh khuôn mặt trước khi đưa vào ArcFace.
    Input: ảnh BGR (numpy)
    Output: tensor (1, 3, 112, 112)
    """
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face / 127.5 - 1.0).astype(np.float32)
    face = np.transpose(face, (2, 0, 1))  # (HWC → CHW)
    face = np.expand_dims(face, axis=0)   # (1, 3, 112, 112)
    return face

def get_embedding(face_img):
    """
    Trích xuất embedding từ ảnh khuôn mặt.
    Input: face_img (numpy BGR)
    Output: vector 512 chiều (numpy)
    """
    try:
        input_blob = preprocess_face(face_img)
        emb = session.run(None, {input_name: input_blob})[0].flatten()
        emb = emb / np.linalg.norm(emb)  # chuẩn hóa vector để so cosine similarity
        return emb
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None



