# models/build.py
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch

def load_yolo_model():
    model = YOLO("models/yolov8n-face-lindevs.pt")
    return model

def load_facenet_model(device="cpu"):
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
