# server_v2.py — SagaMoent Backend V12 (PORT SAFE)

from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

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
            return jsonify({
                "success": False,
                "error": "Missing front image"
            }), 400

        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read() if "back" in request.files else None

        result = analyze_full_coin_v3(
            front_bytes,
            back_bytes
        )

        return jsonify({
            "success": True,
            "engine": "SagaMoent V12",
            "result": result
        })

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    raw_port = os.environ.get("PORT")

    try:
        port = int(raw_port)
    except Exception:
        port = 5000   # SAFE FALLBACK

    print(f"🚀 Starting SagaMoent Backend on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
