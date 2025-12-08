from modules.model_loader import predict_image
from modules.ocr_reader import extract_ocr
from modules.gpt_helper import gpt_enhance
from modules.metadata_builder import build_metadata


def unified_analyze(pil_image):
    prediction = predict_image(pil_image)
    ocr = extract_ocr(pil_image)

    gpt_text = gpt_enhance(
        prediction_text=prediction.get("label", ""),
        ocr_text=ocr.get("text", "")
    )

    metadata = build_metadata(
        prediction=prediction.get("label", ""),
        confidence=prediction.get("confidence", 0),
        ocr_text=ocr.get("text", ""),
        gpt_notes=gpt_text
    )

    return {
        "prediction": prediction,
        "ocr": ocr,
        "gpt": gpt_text,
        "meta": metadata
    }
