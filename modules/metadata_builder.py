import time

def build_metadata(prediction, confidence, ocr_text, gpt_notes):
    return {
        "timestamp": int(time.time()),
        "label": prediction,
        "confidence": confidence,
        "ocr": ocr_text,
        "gpt": gpt_notes,
        "status": "ok"
    }

