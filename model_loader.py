import onnxruntime as ort
import numpy as np
from PIL import Image

def load_onnx_model(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr

def run_onnx(session, arr):
    inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, inputs)[0]

    # assume output is [N classes]
    idx = int(np.argmax(outputs))
    conf = float(outputs[idx])
    return {
        "label": str(idx),
        "confidence": conf
    }

