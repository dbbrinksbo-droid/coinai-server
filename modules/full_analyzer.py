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

    # 1) ONNX-model
    prediction = predict_image(image_bytes)

    # 2) OCR
    ocr_text = extract_ocr(image_bytes)

    # 3) GPT-forbedring
    gpt_notes = gpt_enhance(prediction, ocr_text)

    # 4) Pak det hele sammen
    return build_metadata(
        prediction=prediction,
        ocr_text=ocr_text,
        gpt_notes=gpt_notes,
    )
