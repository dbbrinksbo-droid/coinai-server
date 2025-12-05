from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata

def full_coin_analyze(pil_image):
    """
    Fuldt AI-flow:
    1) ONNX prediction (PIL image → model)
    2) OCR (PIL image → text)
    3) GPT forklaring
    4) Samlet metadata retur
    """

    # 1) ONNX prediction
    prediction_raw = predict_image(pil_image)

    prediction_label = prediction_raw.get("label", "")
    prediction_confidence = prediction_raw.get("confidence", 0)

    # 2) OCR
    ocr_raw = extract_ocr(pil_image)
    ocr_text = ocr_raw.get("text", "")

    # 3) GPT forklaring
    gpt_notes = gpt_enhance(
        prediction_text=prediction_label,
        ocr_text=ocr_text
    )

    # 4) Samlet metadata
    metadata = build_metadata(
        prediction=prediction_label,
        confidence=prediction_confidence,
        ocr_text=ocr_text,
        gpt_notes=gpt_notes,
    )

    return metadata
