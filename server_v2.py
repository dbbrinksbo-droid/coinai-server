import os
import io
import base64
import json
import numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

# IMPORT MODULES
from modules.utils import decode_image
from modules.ocr_year import extract_year
from modules.type_detect import detect_type
from modules.country_detect import detect_country

# GPT
from openai import OpenAI

# -----------------------------
# INIT
# -----------------------------
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_PATH = "models/sagacoin_model.onnx"
LABELS_PATH = "models/labels.json"

# Load model once
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# -----------------------------
# HELPER – decode base64 CLEAN
# -----------------------------
def decode_raw_base64(data):
    if data.startswith("data:image"):
        return base64.b64decode(data.split(",")[1])
    return base64.b64decode(data)

# -----------------------------
# GPT HELPER
# -----------------------------
def gpt_enrich(ai_label, year, country, type_caption):
    prompt = f"""
Du er møntekspert og hjælper en AI-model.

Her er AI-data:
Label: {ai_label}
År: {year}
Land: {country}
Type/Caption: {type_caption}

OPGAVE:
1. Foreslå en forbedret label (kort).
2. Giv en kort historisk beskrivelse.
3. Lav metadata i JSON format med:
   - metal
   - ædelmetal (true/false)
   - kategori
   - oprindelse
"""

    try:
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er en møntekspert."},
                {"role": "user", "content": prompt}
            ]
        )

        return out.choices[0].message.content

    except Exception as e:
        return f"GPT ERROR: {e}"

# -----------------------------
# TRAIN UPLOAD ENDPOINT
# -----------------------------
@app.post("/train-upload")
def train_upload():
    try:
        req = request.get_json()

        front_data = req["front"]
        back_data = req["back"]
        meta = req["meta"]

        folder = f"training_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(folder, exist_ok=True)

        with open(f"{folder}/front.jpg", "wb") as f:
            f.write(decode_raw_base64(front_data))

        with open(f"{folder}/back.jpg", "wb") as f:
            f.write(decode_raw_base64(back_data))

        with open(f"{folder}/meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        return jsonify({
            "success": True,
            "msg": "training data saved",
            "folder": folder
        })

    except Exception as e:
        print("TRAIN UPLOAD ERR:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# -----------------------------
# FULL ANALYZE (AI + GPT + OCR + BLIP)
# -----------------------------
@app.post("/full-analyze")
def full_analyze():
    try:
        req = request.get_json()
        front = req["front"]
        back = req["back"]

        # 1. AI prediction
        f_arr = decode_image(front)
        b_arr = decode_image(back)
        embedding = (f_arr + b_arr) / 2.0

        out = session.run(None, {"input": embedding})[0]
        pred_idx = int(np.argmax(out))
        ai_label = labels[str(pred_idx)]
        confidence = float(np.max(out))

        # 2. OCR
        year = extract_year(front)

        # 3. Type Detection
        type_caption = detect_type(front)

        # 4. Country Detection
        country = detect_country(front)

        # 5. GPT fusion
        gpt_result = gpt_enrich(ai_label, year, country, type_caption)

        return jsonify({
            "success": True,
            "ai_label": ai_label,
            "confidence": confidence,
            "year": year,
            "country": country,
            "type": type_caption,
            "gpt": gpt_result
        })

    except Exception as e:
        print("FULL ANALYZE ERR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/")
def home():
    return "SagaCoin v2 AI SuperBrain is running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

