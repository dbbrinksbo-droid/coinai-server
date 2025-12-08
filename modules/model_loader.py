import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

_session = None
_labels = None


def load_labels():
    global _labels
    if _labels:
        return _labels

    if not os.path.exists(LABELS_FILE):
        print("⚠️ labels.json missing!")
        _labels = []
        return _labels

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    # Sort by index
    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    return _labels


def load_model():
    global _session
    if _session:
        return _session

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(img: Image.Image):
    session = load_model()
    labels = load_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: arr})
    vector = output[0][0]

    idx = int(np.argmax(vector))
    conf = float(np.max(vector))

    label = labels[idx] if labels and idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx,
        "raw": vector.tolist()
    }
