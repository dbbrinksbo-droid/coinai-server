import onnxruntime as ort
import numpy as np
from PIL import Image

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    return output
