# server_v2.py — SagaMoent Backend V12 (Production ready)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from services.analyzer import analyze_full_coin_v3

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
            return jsonify({"success": False, "error": "Missing front image"}), 400

        if "back" not in request.files:
            return jsonify({"success": False, "error": "Missing back image"}), 400

        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()
        user_input = request.form.get("userInput", "{}")

        result = analyze_full_coin_v3(
            front_bytes=front_bytes,
            back_bytes=back_bytes,
            user_input_raw=user_input
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


# IMPORTANT:
# Railway + Docker + Gunicorn entrypoint
# Gunicorn will import `app` from this file
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

