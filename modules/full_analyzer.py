from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata

async def full_coin_analyze(image_bytes):
    prediction = predict_image(image_bytes)
    ocr_text = extract_ocr(image_bytes)
    gpt_notes = gpt_enhance(prediction, ocr_text)

    return build_metadata(prediction, ocr_text, gpt_notes)
