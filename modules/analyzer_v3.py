# modules/analyzer_v3.py
# SagaMoent Full Analyze V12 — VISION ONLY (NO OCR, NO GPT NOTES)

from io import BytesIO
from PIL import Image

from modules.model_loader import predict_image
from modules.vision_gpt_reader import read_coin_from_image


def analyze_full_coin_v3(front_bytes, back_bytes=None, user_input_raw=None):
    # --- Vision GPT FIRST (HARD RULE) ---
    visual = read_coin_from_image(front_bytes)

    # --- AI prediction (classification only) ---
    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    pred_front = predict_image(front_img)

    result = {
        "visual": {
            "front_text": visual.get("front_text", "NOT VISIBLE"),
            "back_text": visual.get("back_text", "NOT VISIBLE"),
            "symbols": visual.get("symbols", []),
        },
        "front_prediction": pred_front,
    }

    # Optional back image classification (no OCR)
    if back_bytes:
        back_img = Image.open(BytesIO(back_bytes)).convert("RGB")
        result["back_prediction"] = predict_image(back_img)

    return result
