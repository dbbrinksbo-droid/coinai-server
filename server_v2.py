from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata
from modules.full_analyzer import full_coin_analyze

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return {"status": "server_v2 running"}

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    image = request.files.get("image")
    if not image:
        return {"error": "No image uploaded"}, 400

    return jsonify(predict_image(image.read()))

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    image = request.files.get("image")
    if not image:
        return {"error": "No image uploaded"}, 400

    return jsonify({"ocr": extract_ocr(image.read())})

@app.route("/gpt", methods=["POST"])
def gpt_endpoint():
    data = request.json
    text = data.get("text", "")
    return jsonify({"response": gpt_enhance(text, "")})

@app.route("/full-analyze", methods=["POST"])
async def full_endpoint():
    image = request.files.get("image")
    if not image:
        return {"error": "No image uploaded"}, 400

    result = await full_coin_analyze(image.read())
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
