# models/build.py
import onnxruntime as ort
import numpy as np

# === Load ArcFace model ===
def load_arcface_model(model_path="models/arcface_w600k_r50.onnx"):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("[INFO] ArcFace model loaded successfully.")
    print("Inputs:", [i.name for i in session.get_inputs()])
    print("Outputs:", [o.name for o in session.get_outputs()])
    return session

# === Hàm trích xuất đặc trưng ===
def get_embedding(session, face_img):
    """
    Input: face_img (numpy BGR)
    Output: embedding vector (512-dim)
    """
    import cv2
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face / 127.5 - 1.0).astype(np.float32)
    face = np.expand_dims(np.transpose(face, (2, 0, 1)), axis=0)
    emb = session.run(None, {"data": face})[0].flatten()
    # Chuẩn hóa vector để dễ so sánh cosine similarity
    emb = emb / np.linalg.norm(emb)
    return emb

if __name__ == "__main__":
    model = load_arcface_model()
    test = np.zeros((112, 112, 3), dtype=np.uint8)
    emb = get_embedding(model, test)
    print("Embedding length:", len(emb))
