# modules/analyzer_v3.py — SagaMoent Full Analyze V12 (Production safe)

import json
from io import BytesIO
from PIL import Image

from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata


def analyze_full_coin_v3(front_bytes, back_bytes, user_input_raw):
    # Load images
    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    back_img = Image.open(BytesIO(back_bytes)).convert("RGB")

    # AI prediction
    pred_front = predict_image(front_img)
    pred_back = predict_image(back_img)

    # OCR
    ocr_front = extract_ocr(front_img).get("text", "")
    ocr_back = extract_ocr(back_img).get("text", "")

    # User input (optional metadata from app)
    try:
        user_data = json.loads(user_input_raw) if user_input_raw else {}
    except Exception:
        user_data = {}

    # GPT enhancement
    gpt_notes = gpt_enhance(
        prediction_text=pred_front.get("label", ""),
        ocr_text=f"front:{ocr_front} | back:{ocr_back}"
    )

    # Metadata aggregation
    metadata = build_metadata(
        prediction=pred_front.get("label", ""),
        confidence=float(pred_front.get("confidence", 0)),
        ocr_text=(ocr_front or ocr_back),
        gpt_notes=gpt_notes
    )

    # Final result
    return {
        "front_prediction": pred_front,
        "back_prediction": pred_back,
        "ocr_front": ocr_front,
        "ocr_back": ocr_back,
        "gpt_notes": gpt_notes,
        "user": user_data,
        "meta": metadata
    }
