import cv2
import os
from ultralytics import YOLO

# Load YOLO model - use relative path
MODEL_PATH = os.path.join("models", "yolov8n-face-lindevs.pt")
model = YOLO(MODEL_PATH)

def detect_faces(frame):
    """Return face bounding boxes in original frame size."""
    h, w = frame.shape[:2]
    inp = 480

    small = cv2.resize(frame, (inp, inp))

    results = model.predict(
        small,
        imgsz=inp,
        conf=0.5,
        iou=0.4,
        max_det=5,
        verbose=False
    )

    out = []
    for r in results:
        for (x1, y1, x2, y2, *_ ) in r.boxes.xyxy.cpu().numpy():
            out.append([
                int(x1 * w / inp),
                int(y1 * h / inp),
                int(x2 * w / inp),
                int(y2 * h / inp)
            ])
    return out


def detect_and_crop_faces(frame):
    """Backward-compatible: return cropped faces."""
    boxes = detect_faces(frame)
    faces = []

    for (x1, y1, x2, y2) in boxes:
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            faces.append(face)

    return faces


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for (x1, y1, x2, y2) in detect_faces(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("YOLO Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


