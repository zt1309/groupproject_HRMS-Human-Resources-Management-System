import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.extract_embeddings import get_embedding
import os
DB_PATH = "db/employees.json"


def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:

        return json.load(f)


def recognize(face_img, threshold=0.5):
    db = load_db()
    emb = get_embedding(face_img)
    if emb is None:
        return None, "Unknown"

    best_score = -1
    best_id = None
    for emp_id, data in db.items():
        db_emb = np.array(data["embedding"])
        score = cosine_similarity([emb], [db_emb])[0][0]
        if score > best_score:
            best_score = score
            best_id = emp_id

    if best_score >= threshold and best_id:
        return best_id, db[best_id]["name"]   # Trả về cả ID và Tên
    else:
        return None, "Unknown"


