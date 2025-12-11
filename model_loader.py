import os
import json
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

_session = None
_labels = None


def download_model_if_needed():
    model_url = os.getenv("MODEL_URL")

    if not model_url:
        print("❌ MODEL_URL not set!")
        return False

    if os.path.exists(MODEL_PATH):
        print("✔ Model already exists")
        return True

    print(f"⬇ Downloading model from: {model_url}")

    try:
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print("✔ Model downloaded")
        return True
    except Exception as e:
        print("❌ MODEL DOWNLOAD FAILED:", e)
        return False


def load_labels():
    global _labels

    if _labels:
        return _labels

    if not os.path.exists(LABELS_FILE):
        print("⚠ labels.json missing")
        _labels = []
        return _labels

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    print(f"✔ Loaded {len(_labels)} labels")
    return _labels


def load_model():
    global _session

    if _session:
        return _session

    download_model_if_needed()

    print("🔄 Loading ONNX model...")
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("✔ ONNX ready")

    return _session


def preprocess(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype("float32") / 255
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, :]
    return arr


def predict_image(img):
    session = load_model()
    labels = load_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    out = session.run(None, {input_name: arr})
    vector = out[0][0]

    idx = int(np.argmax(vector))
    conf = float(vector[idx])
    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx
    }
