from models.build import load_yolo_model, load_facenet_model, get_device

device = get_device()
print("Device:", device)

yolo = load_yolo_model()
print("YOLO model loaded:", yolo)

facenet = load_facenet_model(device)
print("FaceNet model loaded:", facenet)
