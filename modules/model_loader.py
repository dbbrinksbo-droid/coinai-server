import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "sagacoin_model.onnx"
LABELS_PATH = "labels.json"

print("🔄 Initializing ONNX model...")

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    print("✅ ONNX model loaded:", MODEL_PATH)
except Exception as e:
    print("❌ ERROR loading ONNX model:", e)
    session = None

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)
    print("✅ Labels loaded:", LABELS_PATH)
except Exception as e:
    print("⚠️ Could not load labels.json:", e)
    LABELS = None


def _preprocess(image_bytes):
    """
    Convert raw image bytes -> NCHW float32 [0,1] array for ONNX.
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_bytes):
    """
    Kører ONNX-modellen på et billede (bytes).
    Returnerer et simpelt dict med:
      - raw scores
      - top_index
      - top_label (hvis labels findes)
    """
    if session is None:
        return {"error": "ONNX session not initialized"}

    try:
        input_data = _preprocess(image_bytes)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})

        scores = outputs[0][0]
        scores_list = scores.tolist()

        top_idx = int(np.argmax(scores))
        top_label = None

        if isinstance(LABELS, list) and top_idx < len(LABELS):
            top_label = LABELS[top_idx]
        elif isinstance(LABELS, dict):
            # hvis labels er dict med index -> name
            top_label = LABELS.get(str(top_idx)) or LABELS.get(top_idx)

        return {
            "scores": scores_list,
            "top_index": top_idx,
            "top_label": top_label,
        }

    except Exception as e:
        print("❌ ONNX prediction error:", e)
        return {"error": str(e)}
