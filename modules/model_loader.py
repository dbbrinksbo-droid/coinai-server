import io
import torch
from PIL import Image
import numpy as np

# Load model on startup
def load_model():
    try:
        model = torch.jit.load("model.pt", map_location=torch.device("cpu"))
        model.eval()
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print("❌ ERROR loading model:", e)
        return None

model = load_model()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    tensor = torch.tensor(arr).unsqueeze(0)
    return tensor

def predict_image(image_bytes):
    """
    Takes image bytes → preprocess → model prediction.
    Returns model output as Python list.
    """
    try:
        if model is None:
            return {"error": "Model not loaded"}

        input_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            output = model(input_tensor)

        if isinstance(output, torch.Tensor):
            output = output.numpy().tolist()

        return output

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}
