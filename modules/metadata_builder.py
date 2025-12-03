import time

def build_metadata(prediction, ocr_text, gpt_notes):
    return {
        "timestamp": int(time.time()),
        "model_prediction": prediction,
        "ocr_text": ocr_text,
        "gpt_enhancement": gpt_notes,
        "status": "ok"
    }
