# models/build.py
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch

def load_yolo_model():
    """
    Load YOLO model dùng để detect khuôn mặt.
    Bạn cần chắc chắn đã có file models/yolov8n-face.pt
    """
    model = YOLO("models/yolov8n-face-lindevs.pt")
    return model

def load_facenet_model(device="cpu"):
    """
    Load FaceNet (InceptionResnetV1) để sinh embedding khuôn mặt.
    Pretrained trên VGGFace2.
    """
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model

def get_device():
    """
    Chọn device: GPU nếu có, không thì dùng CPU
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
