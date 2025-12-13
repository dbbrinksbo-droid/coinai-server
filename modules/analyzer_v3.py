from io import BytesIO
from PIL import Image

from modules.model_loader import predict_image
from modules.vision_gpt_reader import read_coin_from_image


def analyze_full_coin_v3(front_bytes, back_bytes=None, user_input=None):
    # --- Vision analysis ---
    visual_front = read_coin_from_image(front_bytes)

    visual_back = None
    if back_bytes:
        visual_back = read_coin_from_image(back_bytes)

    # --- Model prediction (classification aid) ---
    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    front_pred = predict_image(front_img)

    back_pred = None
    if back_bytes:
        back_img = Image.open(BytesIO(back_bytes)).convert("RGB")
        back_pred = predict_image(back_img)

    # --- Assemble unified result ---
    result = {
        "visual": {
            "front_text": visual_front.get("text", ""),
            "back_text": visual_back.get("text", "") if visual_back else "",
            "front_symbols": visual_front.get("symbols", []),
            "back_symbols": visual_back.get("symbols", []) if visual_back else [],
        },
        "prediction": {
            "front": front_pred,
            "back": back_pred,
        },
        "user_input": user_input or {},
    }

    return result
