import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Load ONNX model once
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)

    return arr

def predict_image(image_bytes):
    try:
        input_tensor = preprocess_image(image_bytes)

        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)

        return outputs[0].tolist()

    except Exception as e:
        return {"error": str(e)}
