import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

# ----------------------------
# LOAD MODEL + LABELS 1 GANG
# ----------------------------

MODEL_PATH = "sagacoin_model.onnx"
LABELS_PATH = "labels.json"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Load labels
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)


# ----------------------------
# PREPROCESS IMAGE FOR MODEL
# ----------------------------
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))      # HWC → CHW
    arr = np.expand_dims(arr, axis=0)       # Add batch dim
    return arr


# ----------------------------
# MAIN PREDICT FUNCTION
# ----------------------------
def predict_image(img_bytes):
    """
    Returns: { label: "...", confidence: 0.95 }
    """

    try:
        # Preprocess
        input_tensor = preprocess(img_bytes)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Inference
        preds = session.run([output_name], {input_name: input_tensor})[0][0]

        # Softmax-like normalization
        exp_scores = np.exp(preds - np.max(preds))
        probs = exp_scores / exp_scores.sum()

        # Best prediction
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        confidence = float(probs[idx])

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print("❌ ERROR in predict_image:", e)
        return {"error": str(e)}
