

import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import json

app = Flask(__name__)
CORS(app)

print("🔥 SagaCoin AI Server starting...")

# -----------------------------
#  LOAD MODEL + LABELS
# -----------------------------

MODEL_PATH = "sagacoin_model.onnx"   # <- LIGGER I RODEN
LABELS_PATH = "labels.json"         # <- LIGGER I RODEN

print("🔍 Loading model:", MODEL_PATH)
print("🔍 Loading labels:", LABELS_PATH)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


# -----------------------------
#  IMAGE DECODER
# -----------------------------

def decode_image(data: str):
    data = data.replace("data:image/jpeg;base64,", "")
    data = data.replace("data:image/png;base64,", "")
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------
#  API ENDPOINT
# -----------------------------

@app.post("/analyze")
def analyze():
    try:
        req = request.get_json()
        image_front = req.get("front")
        image_back = req.get("back")

        if not image_front or not image_back:
            return jsonify({"success": False, "msg": "Missing images"}), 400

        f = decode_image(image_front)
        b = decode_image(image_back)

        embedding = (f + b) / 2.0
        inputs = {"embedding": embedding}

        output = session.run(None, inputs)[0]
        pred_idx = int(np.argmax(output))
        pred_label = list(labels.keys())[pred_idx]

        return jsonify({
            "success": True,
            "data": {
                "prediction": pred_label,
                "confidence": float(np.max(output))
            }
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"success": False, "msg": "Server error"}), 500


@app.get("/")
def home():
    return "SagaCoin AI server is running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
