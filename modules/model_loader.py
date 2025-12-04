import os
import requests
import onnxruntime as ort

MODEL_URL = "https://drive.google.com/uc?export=download&id=1qtwsFR6uLA4qcSxhfRZ7vWsZ8PB_XSDB"
LOCAL_MODEL_PATH = "/app/sagacoin_full_model.onnx"
LABELS_PATH = "/app/labels.txt"  # just in case you have a labels file

def download_model(url: str, dest_path: str, chunk_size: int = 8192):
    print(f"Downloading model from {url} …")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print(f"Model downloaded and saved to {dest_path}")

def get_model_path() -> str:
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model(MODEL_URL, LOCAL_MODEL_PATH)
    else:
        print(f"Model already present at {LOCAL_MODEL_PATH}")
    return LOCAL_MODEL_PATH

def load_model() -> ort.InferenceSession:
    model_path = get_model_path()
    print(f"Loading ONNX model from {model_path} …")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model loaded successfully!")
    return session

def load_labels() -> list[str]:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels from {LABELS_PATH}")
        return labels
    else:
        print(f"No labels file found at {LABELS_PATH}, returning empty list")
        return []

if __name__ == "__main__":
    session = load_model()
    labels = load_labels()
    # eksempel — du kan ændre efter dit behov:
    print("Ready to run inference.")
