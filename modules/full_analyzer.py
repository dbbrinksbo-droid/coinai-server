from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata


def full_coin_analyze(image_bytes: bytes):
    """
    Fuldt flow:
    1) ONNX prediction
    2) OCR
    3) GPT-4o-mini forklaring
    4) Samlet metadata
    """

    # 1) ONNX prediction
    prediction_raw = predict_image(image_bytes)

    # ensure prediction_raw contains needed fields
    prediction_label = prediction_raw.get("label", "")
    prediction_confidence = prediction_raw.get("confidence", 0)

    # 2) OCR
    ocr_raw = extract_ocr(image_bytes)
    ocr_text = ocr_raw.get("text", "")

    # 3) GPT enhancement
    gpt_notes = gpt_enhance(
        prediction_text=prediction_label,
        ocr_text=ocr_text
    )

    # 4) metadata
    metadata = build_metadata(
        prediction=prediction_label,
        confidence=prediction_confidence,
        ocr_text=ocr_text,
        gpt_notes=gpt_notes,
    )

    return metadata
