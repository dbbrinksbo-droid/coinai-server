import time


def build_metadata(prediction, ocr_text=None, gpt_notes=None):
    """
    Samler al info om mønten ét sted.
    """
    return {
        "timestamp": int(time.time()),
        "model_prediction": prediction,
        "ocr_text": ocr_text or "",
        "gpt_enhancement": gpt_notes or "",
        "status": "ok",
    }
