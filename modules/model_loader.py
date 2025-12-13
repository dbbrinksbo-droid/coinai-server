# model_loader.py — LOCAL ONLY (SagaMoent FINAL)

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "/app/models/sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

_session = None
_labels = None


def load_labels():
    global _labels

    if _labels is not None:
        return _labels

    if not os.path.exists(LABELS_FILE):
        raise RuntimeError("labels.json missing")

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    print(f"✔ Loaded {len(_labels)} labels")

    return _labels


def load_model():
    global _session

    if _session is not None:
        return _session

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("ONNX model file not found")

    print("✔ Using local ONNX model")
    print("🔄 Loading ONNX model…")

    _session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print("✔ ONNX model ready")

    return _session


def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)   # CHW
    arr = arr[np.newaxis, :]       # NCHW
    return arr


def predict_image(img: Image.Image):
    session = load_model()
    labels = load_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: arr})
    vector = outputs[0][0]

    idx = int(np.argmax(vector))
    conf = float(vector[idx])
    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx
    }
