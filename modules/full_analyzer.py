from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata


async def full_coin_analyze(image_bytes):
    """
    Full pipeline:
    1. ONNX model prediction
    2. OCR detection
    3. GPT enhancement (billig GPT-4o-mini)
    4. Bundle metadata
    """

    # 1) AI prediction
    prediction = predict_image(image_bytes)

    # 2) OCR
    ocr_text = extract_ocr(image_bytes)

    # 3) GPT improvement
    gpt_notes = await gpt_enhance(prediction, ocr_text)

    # 4) Build final metadata package
    result = build_metadata(
        prediction=prediction,
        ocr_text=ocr_text,
        gpt_notes=gpt_notes
    )

    return result
