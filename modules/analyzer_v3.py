import json
from io import BytesIO
from PIL import Image

from modules.model_loader import predict_image
from modules.vision_gpt_reader import read_coin_from_image


def analyze_full_coin_v3(front_bytes, back_bytes=None, user_input_raw=""):
    # --- Parse user input (optional) ---
    try:
        user = json.loads(user_input_raw) if user_input_raw else {}
        if not isinstance(user, dict):
            user = {}
    except Exception:
        user = {}

    # --- Model predictions (classification aid) ---
    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    pred_front = predict_image(front_img)

    pred_back = None
    if back_bytes:
        back_img = Image.open(BytesIO(back_bytes)).convert("RGB")
        pred_back = predict_image(back_img)

    # --- Vision GPT: read BOTH sides ---
    visual_front = read_coin_from_image(front_bytes, side="front", user=user, prediction=pred_front)
    visual_back = read_coin_from_image(back_bytes, side="back", user=user, prediction=pred_back) if back_bytes else {
        "side": "back",
        "text": "",
        "fields": {},
        "confidence": 0,
        "notes": "No back image provided"
    }

    # --- Unified summary (what app should show / send to Coin Expert) ---
    summary_parts = []
    if visual_front.get("text"):
        summary_parts.append(f"Front: {visual_front['text']}")
    if visual_back.get("text"):
        summary_parts.append(f"Back: {visual_back['text']}")
    summary = "\n".join(summary_parts).strip()

    return {
        # Vision output (what user wants)
        "visual": {
            "front_text": visual_front.get("text", ""),
            "back_text": visual_back.get("text", ""),
            "front_fields": visual_front.get("fields", {}),
            "back_fields": visual_back.get("fields", {}),
            "front_confidence": visual_front.get("confidence", 0),
            "back_confidence": visual_back.get("confidence", 0),
            "notes": {
                "front": visual_front.get("notes", ""),
                "back": visual_back.get("notes", ""),
            }
        },

        # Model output (supporting signal)
        "front_prediction": pred_front,
        "back_prediction": pred_back,

        # User hints
        "user_input": user,

        # One combined text to forward to Coin Expert
        "summary": summary,
    }
