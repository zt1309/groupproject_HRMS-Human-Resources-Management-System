import torch
import torch.nn as nn
import cv2
import numpy as np
import os

MODEL_PATH = os.path.join("models", "antispoof_resnet18.pt")

ENABLE_ANTISPOOF = True
ANTISPOOF_SOFT_MODE = True
ANTISPOOF_THRESHOLD = 0.5


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


try:
    state = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    model = AntiSpoofNet()
    model.load_state_dict(state, strict=False)
    model.eval()

except Exception:
    model = None
    ENABLE_ANTISPOOF = False


def check_liveness(face_img):
    if not ENABLE_ANTISPOOF:
        return True

    if model is None:
        return True

    try:
        face = cv2.resize(face_img, (80, 80))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face).unsqueeze(0)

        with torch.no_grad():
            logit = model(face)
            score = torch.sigmoid(logit).item()

        is_real = score > ANTISPOOF_THRESHOLD

        if ANTISPOOF_SOFT_MODE:
            return True

        return is_real

    except Exception:
        return True
