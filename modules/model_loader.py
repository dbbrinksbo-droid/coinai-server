import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

# ------------------------
# FILPLACERING
# ------------------------
MODEL_PATH = "sagacoin_full_model.onnx"
LABELS_PATH = "labels.json"

print("🔄 Loading FULL SagaCoin ONNX model...")

# ------------------------
# LOAD MODEL
# ------------------------
try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    print("✅ ONNX model loaded")
except Exception as e:
    print("❌ ERROR loading ONNX model:", e)
    session = None

# ------------------------
# LOAD LABELS
# ------------------------
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)

    # Ensure index → label mapping
    if isinstance(LABELS, dict):
        LABELS = [LABELS[str(i)] for i in range(len(LABELS))]
    print("✅ Labels loaded")
except Exception as e:
    print("⚠️ Could not load labels:", e)
    LABELS = None

# ------------------------
# ViT PREPROCESS
# ------------------------
def preprocess_vit(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0

    # ViT normalization
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    arr = (arr - mean) / std

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    # add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr


# ------------------------
# PREDICT
# ------------------------
def predict_image(image_bytes):
    if session is None:
        return {"error": "Model not loaded"}

    try:
        input_data = preprocess_vit(image_bytes)

        input_name = session.get_inputs()[0].name

        outputs = session.run(None, {input_name: input_data})
        scores = outputs[0][0]

        top_idx = int(np.argmax(scores))
        top_label = LABELS[top_idx] if LABELS else str(top_idx)

        return {
            "scores": scores.tolist(),
            "top_index": top_idx,
            "top_label": top_label
        }

    except Exception as e:
        print("❌ ONNX ERROR:", e)
        return {"error": str(e)}
