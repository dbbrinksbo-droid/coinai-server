import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.analyzer_v3 import analyze_full_coin_v3

print("🔥🔥🔥 SagaMoent Backend V13 ACTIVE — VISION BOTH SIDES + USER INPUT 🔥🔥🔥")

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "engine": "SagaMoent Backend V13",
        "model_exists": os.path.exists("/models/sagacoin_full_model.onnx")
    })


@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    try:
        front_file = request.files.get("front")
        back_file = request.files.get("back")
        user_input_raw = request.form.get("user_input", "")

        if not front_file:
            return jsonify({"success": False, "error": "Missing front image"}), 400

        front_bytes = front_file.read()
        back_bytes = back_file.read() if back_file else None

        result = analyze_full_coin_v3(
            front_bytes=front_bytes,
            back_bytes=back_bytes,
            user_input_raw=user_input_raw,
        )

        return jsonify({
            "success": True,
            "engine": "SagaMoent Backend V13",
            "result": result
        })

    except Exception as e:
        print("❌ ERROR /full-analyze-v3:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting SagaMoent Backend on port {port}")
    app.run(host="0.0.0.0", port=port)
