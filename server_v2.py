import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# V3 analyzer
from modules.analyzer_v3 import analyze_full_coin_v3

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"server": "SagaMoent Backend V3", "status": "online"})

# ---------------------------------------------------------------------
# NEW V3 ENDPOINT — supports front + back + userInput
# ---------------------------------------------------------------------
@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    if "front" not in request.files or "back" not in request.files:
        return jsonify({"error": "Missing images: need front & back"}), 400

    try:
        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()

        user_input = request.form.get("userInput", "{}")

        result = analyze_full_coin_v3(front_bytes, back_bytes, user_input)
        return jsonify(result)

    except Exception as e:
        print("V3 ERROR:", e)
        return jsonify({"error": "Backend V3 failed", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
