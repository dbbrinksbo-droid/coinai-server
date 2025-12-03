
import os
import io
import base64
import json
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

# ----------------------------------------------------
# FLASK APP
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)

print("🔥 SagaCoin AI Server starting...")

# ----------------------------------------------------
# LOAD MODEL + LABELS
# ----------------------------------------------------
MODEL_PATH = "sagacoin_model.onnx"
LABELS_PATH = "labels.json"

print("🔍 Loading model:", MODEL_PATH)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

print("🔍 Loading labels:", LABELS_PATH)
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

# ----------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    return img_array

# ----------------------------------------------------
# API ENDPOINT
# ----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        imgdata = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(imgdata))

        input_tensor = preprocess_image(image)

        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        ort_outs = session.run(None, ort_inputs)

        pred = int(np.argmax(ort_outs[0]))
        label = LABELS[str(pred)]

        return jsonify({
            "prediction": pred,
            "label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------
# ASGI WRAPPER FOR UVICORN / RAILWAY
# ----------------------------------------------------
from starlette.responses import Response

async def asgi_app(scope, receive, send):
    """
    Simple ASGI wrapper for Flask so Uvicorn can run it on Railway.
    """
    if scope["type"] != "http":
        raise NotImplementedError("Only HTTP supported.")

    environ = {
        "REQUEST_METHOD": scope["method"],
        "PATH_INFO": scope["path"],
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "8080",
        "wsgi.version": (1, 0),
        "wsgi.input": io.BytesIO(),
        "wsgi.errors": io.BytesIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "wsgi.url_scheme": "http",
    }

    def start_response(status, headers, exc_info=None):
        pass

    response_body = b"".join(app.wsgi_app(environ, start_response))
    response = Response(response_body, media_type="application/json")
    await response(scope, receive, send)

# ----------------------------------------------------
# LOCAL TEST MODE
# ----------------------------------------------------
if __name__ == "__main__":
    print("Running locally at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
