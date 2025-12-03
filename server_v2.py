import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.model_loader import predict_image
from modules.ocr_reader import read_text_from_image
from modules.gpt_helper import ask_gpt
from modules.metadata_builder import build_metadata
from modules.full_analyzer import full_coin_analyze

app = Flask(__name__)
CORS(app)

print("🔥 SagaCoin AI v2 Backend starting...")


# -----------------------------
# Helper: Decode Base64 → Bytes
# -----------------------------
def decode_image_base64(img_b64: str) -> bytes:
    img_b64 = img_b64.replace("data:image/jpeg;base64,", "")
    img_b64 = img_b64.replace("data:image/png;base64,", "")
    return base64.b64decode(img_b64)


# -----------------------------
# ROUTE: /predict (ONNX ONLY)
# -----------------------------
@app.post("/predict")
def predict_route():
    try:
        data = request.get_json()
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "Missing image"}), 400

        img_bytes = decode_image_base64(img_b64)
        result = predict_image(img_bytes)
        return jsonify({"success": True, "prediction": result})

    except Exception as e:
        print("ERROR /predict:", e)
        return jsonify({"error": "server error"}), 500


# -----------------------------
# ROUTE: /ocr (GPT Vision OCR)
# -----------------------------
@app.post("/ocr")
def ocr_route():
    try:
        data = request.get_json()
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "Missing image"}), 400

        text = read_text_from_image(img_b64)
        return jsonify({"success": True, "ocr_text": text})

    except Exception as e:
        print("ERROR /ocr:", e)
        return jsonify({"error": "server error"}), 500


# -----------------------------
# ROUTE: /gpt-helper
# -----------------------------
@app.post("/gpt-helper")
def gpt_route():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        result = ask_gpt(prompt)
        return jsonify({"success": True, "response": result})

    except Exception as e:
        print("ERROR /gpt-helper:", e)
        return jsonify({"error": "server error"}), 500


# -----------------------------
# ROUTE: /full-analyze
# -----------------------------
@app.post("/full-analyze")
def full_analyze_route():
    try:
        data = request.get_json()
        img_b64 = data.get("image")

        if not img_b64:
            return jsonify({"error": "Missing image"}), 400

        img_bytes = decode_image_base64(img_b64)

        # RUN FULL PIPELINE
        result = full_coin_analyze(img_bytes)

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print("ERROR /full-analyze:", e)
        return jsonify({"error": "server error"}), 500


# -----------------------------
# DEFAULT ROUTE
# -----------------------------
@app.get("/")
def home():
    return "SagaCoin AI v2 Backend is running."


# -----------------------------
# START
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
