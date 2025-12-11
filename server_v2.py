rebuild marker 20251211
import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from modules.analyzer_v3 import analyze_full_coin_v3

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "server": "SagaMoent Backend V12",
        "status": "online",
        "mode": "full-analyze-v3"
    })

@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    try:
        if "front" not in request.files:
            return jsonify({"error": "Missing front image"}), 400

        if "back" not in request.files:
            return jsonify({"error": "Missing back image"}), 400

        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()
        user_input = request.form.get("userInput", "{}")

        result = analyze_full_coin_v3(front_bytes, back_bytes, user_input)

        return jsonify({
            "success": True,
            "engine": "SagaMoent V12",
            "result": result
        })

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# Railway needs the app object exposed
app = app
