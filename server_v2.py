import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Vigtig: Analyzer som bruger model_loader → AI pipeline
from modules.analyzer_v3 import analyze_full_coin_v3

# Vigtig: Sikrer at ONNX-modellen downloades ved startup
from modules.model_loader import ensure_model_exists

# ----------------------------------------------------------
#  INITIALIZE SERVER
# ----------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Download ONNX modellen når serveren starter
print("🚀 SagaMoent Backend V12 – initialiserer model…")
ensure_model_exists()
print("✔ Model klar – server starter nu.")


# ----------------------------------------------------------
#  ROOT ENDPOINT (status check)
# ----------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "server": "SagaMoent Backend V12",
        "status": "online",
        "mode": "full-analyze-v3"
    })


# ----------------------------------------------------------
#  FULL ANALYZE ENDPOINT
# ----------------------------------------------------------
@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    try:
        # Check required files
        if "front" not in request.files:
            return jsonify({"error": "Missing front image"}), 400
        if "back" not in request.files:
            return jsonify({"error": "Missing back image"}), 400

        # Read image bytes
        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()

        # Optional user metadata
        user_input = request.form.get("userInput", "{}")

        # Run AI analysis pipeline
        result = analyze_full_coin_v3(front_bytes, back_bytes, user_input)

        # Success response
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


# ----------------------------------------------------------
#  DEBUG MODEL ENDPOINT (kan testes via browser)
# ----------------------------------------------------------
@app.route("/debug/model", methods=["GET"])
def debug_model():
    """Returnerer info om modelens filsystem på Railway."""
    try:
        exists = os.path.exists("models/sagacoin_full_model.onnx")
        files = os.listdir("models") if exists else []
        return jsonify({
            "model_path": "models/sagacoin_full_model.onnx",
            "exists": exists,
            "files_in_models_folder": files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------
#  SERVER RUNNER
# ----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"🌍 Kører SagaMoent Backend V12 på port {port}")
    app.run(host="0.0.0.0", port=port)

