import os
import json
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image

# ----------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------

# Din Google Drive model-URL (fallback)
DEFAULT_MODEL_URL = (
    "https://drive.google.com/uc?id=1qtwsFR6uLA4qcSxhfRZ7vWsZ8PB_XSDB"
)

# Brug Railway variable hvis den findes, ellers fallback
MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sagacoin_full_model.onnx")
LABELS_FILE = "labels.json"

_session = None
_labels = None


# ----------------------------------------------------------
# ENSURE MODEL EXISTS (DOWNLOAD IF MISSING)
# ----------------------------------------------------------
def ensure_model_exists():
    """
    Sørger for at ONNX-modellen findes.
    Hvis ikke, downloades den fra Google Drive.
    """
    if os.path.exists(MODEL_PATH):
        print(f"✔ Model already exists: {MODEL_PATH}")
        return

    if not MODEL_URL:
        # Dette burde aldrig ske, fordi vi har DEFAULT_MODEL_URL
        raise RuntimeError("❌ MODEL_URL er tom – ingen download-URL til modellen!")

    print(f"⬇ Downloader model fra: {MODEL_URL}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    resp = requests.get(MODEL_URL, allow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(
            f"❌ Kunne ikke downloade modellen (status {resp.status_code})"
        )

    with open(MODEL_PATH, "wb") as f:
        f.write(resp.content)

    print(f"✔ Model downloaded og gemt: {MODEL_PATH}")


# ----------------------------------------------------------
# LOAD LABELS
# ----------------------------------------------------------
def get_labels():
    """
    Loader labels fra labels.json, eller laver fallback hvis filen mangler.
    """
    global _labels
    if _labels is not None:
        return _labels

    if not os.path.exists(LABELS_FILE):
        print("⚠ labels.json mangler — fallback labels bruges")
        _labels = [f"label_{i}" for i in range(500)]
        return _labels

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    print(f"✔ Labels loaded: {len(_labels)}")
    return _labels


# ----------------------------------------------------------
# GET ONNX SESSION
# ----------------------------------------------------------
def get_session():
    """
    Returnerer en global ONNX-session, loader model og session første gang.
    """
    global _session

    if _session is not None:
        return _session

    ensure_model_exists()

    print(f"🔄 Loader ONNX model: {MODEL_PATH}")
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("✔ ONNX model loaded")

    return _session


# ----------------------------------------------------------
# PREPROCESS IMAGE
# ----------------------------------------------------------
def preprocess(img: Image.Image):
    """
    Konverterer PIL Image til [1, 3, 224, 224] float32 tensor i [0,1].
    """
    img = img.resize((224, 224)).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))   # HWC → CHW
    arr = np.expand_dims(arr, axis=0)    # [C, H, W] → [1, C, H, W]
    return arr


# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------
def predict_image(img: Image.Image):
    """
    Hovedfunktion til at lave et prediction på et PIL Image.
    Returnerer dict: {label, confidence, index}
    """
    session = get_session()
    labels = get_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: arr})
    vector = output[0][0]

    idx = int(np.argmax(vector))
    conf = float(vector[idx])

    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx,
    }
