import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.model_loader import predict_image
from modules.full_analyzer import full_coin_analyze

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "version": "coinai-v2", "message": "ONNX + GPT backend kører"})


@app.route("/predict", methods=["POST"])
def route_predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file field 'image'"}), 400

    img_file = request.files["image"]
    image_bytes = img_file.read()

    result = predict_image(image_bytes)
    return jsonify(result)


@app.route("/full-analyze", methods=["POST"])
def route_full_analyze():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file field 'image'"}), 400

    img_file = request.files["image"]
    image_bytes = img_file.read()

    result = full_coin_analyze(image_bytes)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
