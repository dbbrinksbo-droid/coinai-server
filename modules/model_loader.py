import os
import json
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image


# ----------------------------------------------------------
#  SETTINGS
# ----------------------------------------------------------
MODEL_URL = os.getenv("MODEL_URL")  # Railway variable
MODEL_PATH = "models/sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

_session = None
_labels = None


# ----------------------------------------------------------
#  DOWNLOAD MODEL AUTOMATICALLY
# ----------------------------------------------------------
def ensure_model():
    """Downloader modellen én gang hvis den ikke findes i /models."""
    if not os.path.exists("models"):
        os.makedirs("models")

    if os.path.exists(MODEL_PATH):
        print("✔ Model already exists")
        return

    if not MODEL_URL:
        raise RuntimeError("❌ MODEL_URL mangler i Railway Environment Variables")

    print("⬇ Downloader model fra:", MODEL_URL)

    try:
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download model, status: {r.status_code}")

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

        print("✔ Model downloaded & saved:", MODEL_PATH)

    except Exception as e:
        raise RuntimeError(f"❌ Model download failed: {e}")


# ----------------------------------------------------------
#  LOAD LABELS.JSON
# ----------------------------------------------------------
def load_labels():
    """Loader labels fra labels.json eller fallback."""
    global _labels
    if _labels is not None:
        return _labels

    if not os.path.exists(LABELS_FILE):
        print("⚠ labels.json missing — using fallback labels")
        _labels = [f"label_{i}" for i in range(500)]
        return _labels

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    # sort by index value
    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    return _labels


# ----------------------------------------------------------
#  LOAD ONNX MODEL INTO MEMORY
# ----------------------------------------------------------
def load_model():
    global _session
    if _session is not None:
        return _session

    ensure_model()

    print("📦 Initializing ONNX session…")
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("✔ ONNX model loaded")
    return _session


# ----------------------------------------------------------
#  IMAGE PREPROCESSING
# ----------------------------------------------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr


# ----------------------------------------------------------
#  PREDICT IMAGE USING MODEL
# ----------------------------------------------------------
def predict_image(img: Image.Image):
    """Return prediction {label, confidence, index}"""
    session = load_model()
    labels = load_labels()

    input_name = session.get_inputs()[0].name
    array = preprocess(img)

    output = session.run(None, {input_name: array})
    vector = output[0][0]

    idx = int(np.argmax(vector))
    conf = float(np.max(vector))

    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx
    }

