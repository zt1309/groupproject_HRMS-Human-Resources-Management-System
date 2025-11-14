# ...existing code...
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 anti-spoofing model (real / fake)
def load_antispoof_model():
    try:
        model = YOLO("models/anticheking.pt")
        print("[INFO] Anti-spoofing YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load anti-spoof model: {e}")
        return None


# Check if a face is real or fake
def check_liveness(face_img, threshold=0.5):
    """
    Input: face_img (numpy array, BGR from OpenCV)
    Output: True if real, False if fake
    """
    model = load_antispoof_model()
    if model is None:
        print("[WARN] Anti-spoof disabled (model not loaded).")
        return True  # fallback: allow if model not loaded

    try:
        # Convert to RGB because YOLO expects RGB input
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, verbose=False)

        # Get label & confidence
        names = results[0].names
        boxes = results[0].boxes

        if len(boxes) == 0:
            return True  # if nothing detected, allow

        conf = boxes.conf.cpu().numpy()[0]
        cls = int(boxes.cls.cpu().numpy()[0])
        label = names[cls].lower()

        if label == "fake" and conf > threshold:
            return False
        return True

    except Exception as e:
        print(f"[ERROR] Liveness check failed: {e}")
        return True  # fallback: allow on unexpected error
# ...existing code...
