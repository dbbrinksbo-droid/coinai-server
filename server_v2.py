# server_v2.py
from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.analyzer_v3 import analyze_full_coin_v3

app = Flask(__name__)
CORS(app)


@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    try:
        front_file = request.files.get("front")
        back_file = request.files.get("back")

        if not front_file:
            return jsonify({"success": False, "error": "No front image"}), 400

        front_bytes = front_file.read()
        back_bytes = back_file.read() if back_file else None

        result = analyze_full_coin_v3(
            front_bytes=front_bytes,
            back_bytes=back_bytes,
        )

        # 🔥 RETURN EXACTLY WHAT ANALYZER RETURNS
        return jsonify({
            "success": True,
            "engine": "SagaMoent V12",
            "result": result
        })

    except Exception as e:
        print("❌ ANALYZE ERROR:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("🚀 Starting SagaMoent Backend on port 8080")
    app.run(host="0.0.0.0", port=8080)
