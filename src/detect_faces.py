import cv2
from ultralytics import YOLO

#load my model yolo
yolo = YOLO("models/yolov8n-face-lindevs.pt")

def detect_and_crop_faces(frame):
    """
    Detect khuôn mặt trong frame và trả về list ảnh khuôn mặt crop
    """
    results = yolo(frame)  # YOLO inference
    faces = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes [x1,y1,x2,y2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
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

        faces = detect_and_crop_faces(frame)

        # draw bounding boxes for debug
        results = yolo(frame)
        annotated = results[0].plot()

        cv2.imshow("YOLO Face Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
