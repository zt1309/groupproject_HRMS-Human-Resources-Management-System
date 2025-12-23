import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from config import ENABLE_ANTISPOOF, ANTISPOOF_THRESHOLD, ANTISPOOF_MODEL

MODEL_PATH = ANTISPOOF_MODEL

# -------- Load model --------
try:
    state = torch.load(MODEL_PATH, map_location="cpu")

    if "state_dict" in state:
        state = state["state_dict"]

    class AntiSpoofNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.model(x)

    model = AntiSpoofNet()
    model.load_state_dict(state, strict=False)
    model.eval()

    print("[OK] Anti-spoof model loaded successfully!")
    if not ENABLE_ANTISPOOF:
        print("[WARN] Anti-spoof is DISABLED in config")

except Exception as e:
    print(f"[ERROR] Failed to load anti-spoof model: {e}")
    print("[WARN] Anti-spoof disabled.")
    model = None


# -------- Check liveness --------
def check_liveness(face_img):
    """
    Kiểm tra tính thật của khuôn mặt
    Input: face_img (numpy BGR)
    Output: True nếu thật, False nếu giả
    """
    # Check if anti-spoofing is disabled
    if not ENABLE_ANTISPOOF:
        return True
    
    # Check if model is loaded
    if model is None:
        print("[WARN] Anti-spoof model not loaded, returning True")
        return True
    
    try:
        # Resize và preprocess
        face = cv2.resize(face_img, (80, 80))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face).unsqueeze(0)

        with torch.no_grad():
            score = model(face).item()
        
        # Score > threshold = real, <= threshold = fake
        is_real = score > ANTISPOOF_THRESHOLD
        print(f"[ANTISPOOF] Score: {score:.3f}, Threshold: {ANTISPOOF_THRESHOLD}, Real: {is_real}")
        return is_real
        
    except Exception as e:
        print(f"[ERROR] Anti-spoof check failed: {e}")
        return True  # Default to real if error
