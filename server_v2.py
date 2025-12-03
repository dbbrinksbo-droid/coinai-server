import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# OUR MODULES
from modules.model_loader import load_model, preprocess_image, predict as run_predict
from modules.ocr_reader import read_text_from_image
from modules.gpt_helper import explain_prediction
from modules.metadata_builder import build_metadata

# INIT FLASK
app = Flask(__name__)
CORS(app)

# LOAD MODEL
MODEL_PATH = "models/sagacoin_model.onnx"
session = load_model(MODEL_PATH)

# -------------- HELPERS -----------------

def decode_raw_image(base64_data):
    """Convert base64 → PIL Image"""
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]

    img_bytes = base64.b64decode(base64_data)
    return io.BytesIO(img_bytes)

# -------------- ENDPOINTS -----------------

@app.get("/")
def home():
    return "SagaCoin v2 Backend is running."

@app.post("/predict")
def api_predict():
    try:
        data = request.get_json()
        img = decode_raw_image(data["image"])
        pil_img = preprocess_image(io.BytesIO(img.read()))
        output = run_predict(session, pil_img)

        return jsonify({
            "success": True,
            "prediction": output.tolist()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/ocr")
def api_ocr():
    try:
        data = request.get_json()
        img_bytes = decode_raw_image(data["image"])

        text = read_text_from_image(img_bytes)

        return jsonify({
            "success": True,
            "text": text
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/gpt-helper")
def api_gpt():
    try:
        data = request.get_json()
        label = data["label"]
        conf = data.get("confidence", 0)

        text = explain_prediction(label, conf)

        return jsonify({
            "success": True,
            "message": text
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/full-analyze")
def api_full():
    try:
        data = request.get_json()
        img = decode_raw_image(data["image"])

        # 1) Prediction
        pil_img = preprocess_image(io.BytesIO(img.read()))
        ai_pred = run_predict(session, pil_img)

        # 2) OCR
        img.seek(0)
        ocr_text = read_text_from_image(img)

        # 3) GPT enhancement
        gpt = explain_prediction(str(ai_pred), 0)

        # 4) Build metadata
        result = build_metadata(
            prediction=str(ai_pred),
            ocr_text=ocr_text,
            gpt_notes=gpt
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/train-upload")
def api_train_upload():
    try:
        data = request.get_json()

        front = decode_raw_image(data["front"])
        back  = decode_raw_image(data["back"])
        meta  = data["meta"]

        folder = f"training_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(folder, exist_ok=True)

        # Save front
        with open(f"{folder}/front.jpg", "wb") as f:
            f.write(front.read())

        # Save back
        with open(f"{folder}/back.jpg", "wb") as f:
            f.write(back.read())

        # Save metadata
        with open(f"{folder}/meta.json", "w") as f:
            import json
            json.dump(meta, f, indent=4)

        return jsonify({"success": True, "folder": folder})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
