from io import BytesIO
from PIL import Image
import json

from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance


def analyze_full_coin_v3(front_bytes, back_bytes, user_input):
    """
    SagaMoent V3 analyse:
    - For + bagside
    - OCR på begge sider
    - ONNX analyse
    - GPT forklaring
    - brugerinput indbygget
    """

    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    back_img = Image.open(BytesIO(back_bytes)).convert("RGB")

    pred_front = predict_image(front_img)
    pred_back = predict_image(back_img)

    ocr_front = extract_ocr(front_bytes)
    ocr_back = extract_ocr(back_bytes)

    try:
        user_data = json.loads(user_input)
    except:
        user_data = {}

    gpt_notes = gpt_enhance(
        prediction=pred_front,
        ocr_text=f"front: {ocr_front}\nback: {ocr_back}"
    )

    return {
        "label": pred_front.get("label", "Ukendt mønt"),
        "confidence": pred_front.get("confidence", 0),

        "ocr": ocr_front or ocr_back or "",
        "gpt": gpt_notes or "Ingen GPT forklaring",

        "year": user_data.get("year", ""),
        "country": user_data.get("country", ""),
        "type": user_data.get("type", ""),

        "meta": {
            "pred_front": pred_front,
            "pred_back": pred_back,
            "ocr_front": ocr_front,
            "ocr_back": ocr_back
        }
    }
