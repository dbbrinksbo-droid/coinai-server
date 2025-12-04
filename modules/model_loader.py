import os
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort

# ---------------------------------------------------------
# KONFIG
# ---------------------------------------------------------

MODEL_URL = "https://drive.google.com/uc?export=download&id=1qtwsFR6uLA4qcSxhfRZ7vWsZ8PB_XSDB"
LOCAL_MODEL_PATH = "/app/sagacoin_full_model.onnx"
LABELS_PATH = "/app/labels.txt"   # Kun hvis du har den

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
# LOAD MODEL + SESSION CACHE
# ---------------------------------------------------------

_session = None   # global cache

def load_model() -> ort.InferenceSession:
    global _session

    if _session is not None:
        return _session

    model_path = get_model_path()
    print("Loading ONNX model…")

    _session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    print("Model loaded successfully!")
    return _session

# ---------------------------------------------------------
# LABELS (HVIS DE FINDES)
# ---------------------------------------------------------

def load_labels() -> list:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(labels)} labels")
        return labels

    print("No labels file found — using raw output.")
    return []

# ---------------------------------------------------------
# BILLEDE → MODEL FORMAT
# ---------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Ændr hvis din model bruger andet
    arr = np.array(image).astype("float32") / 255.0

    # Model format: (1, 3, 224, 224)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------------------------
# FORUDSIGELSE (MAIN FUNCTION)
# ---------------------------------------------------------

def predict_image(image: Image.Image):
    session = load_model()
    img_array = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})

    output_vector = output[0][0]  # første output, første batch

    labels = load_labels()

    if labels and len(labels) == len(output_vector):
        index = int(np.argmax(output_vector))
        return {
            "label": labels[index],
            "confidence": float(np.max(output_vector)),
            "raw": output_vector.tolist()
        }

    # fallback – ingen labels
    return {
        "raw_output": output_vector.tolist(),
        "predicted_index": int(np.argmax(output_vector)),
        "confidence": float(np.max(output_vector)),
    }

# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Model loader test-run…")
    load_model()
