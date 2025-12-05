import os
import json
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort

# ---------------------------------------------------------
# KONFIG
# ---------------------------------------------------------

MODEL_URL = "https://drive.google.com/uc?export=download&id=1qtwsFR6uLA4qcSxhfRZ7vWsZ8PB_XSDB"

# Brug relative paths -> virker både lokalt og Railway
LOCAL_MODEL_PATH = "sagacoin_full_model.onnx"
LABELS_PATH = "labels.json"

# ---------------------------------------------------------
# DOWNLOAD MODEL HVIS IKKE FINDES
# ---------------------------------------------------------

def download_model(url: str, dest_path: str, chunk_size: int = 8192):
    print(f"Downloading ONNX model from {url} …")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    print(f"Model downloaded and saved to: {dest_path}")


def get_model_path() -> str:
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model(MODEL_URL, LOCAL_MODEL_PATH)
    else:
        print(f"Model already exists at {LOCAL_MODEL_PATH}")
    return LOCAL_MODEL_PATH

# ---------------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------------

_session = None

def load_model() -> ort.InferenceSession:
    global _session

    if _session is not None:
        return _session

    path = get_model_path()
    print("Loading ONNX model…")

    _session = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )

    print("Model loaded successfully!")
    return _session

# ---------------------------------------------------------
# LABELS (JSON FORMAT)
# ---------------------------------------------------------

def load_labels() -> list:
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels_dict = json.load(f)

            # labels.json er et dict -> vi skal sortere efter index
            sorted_labels = [label for label, idx in sorted(labels_dict.items(), key=lambda x: x[1])]

            print(f"Loaded {len(sorted_labels)} labels")
            return sorted_labels

        except Exception as e:
            print("Error loading labels:", e)

    print("No labels.json file found — using indices only.")
    return []

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0

    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------------------------
# MAIN PREDICTION
# ---------------------------------------------------------

def predict_image(image: Image.Image):
    session = load_model()
    img_array = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})

    output_vector = output[0][0]

    labels = load_labels()

    index = int(np.argmax(output_vector))
    confidence = float(np.max(output_vector))

    if labels and len(labels) > index:
        return {
            "label": labels[index],
            "confidence": confidence,
            "raw": output_vector.tolist()
        }

    # fallback
    return {
        "predicted_index": index,
        "confidence": confidence,
        "raw_output": output_vector.tolist()
    }

# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Test-loading model…")
    load_model()
