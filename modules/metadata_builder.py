import time

def build_metadata(prediction, confidence=0.0, ocr_text=None, gpt_notes=None):
    """
    Samler al AI-data til ét samlet svar.
    Bruges af full_analyze → sendes til appen.
    """

    return {
        "timestamp": int(time.time()),

        # Model output
        "label": prediction,
        "confidence": float(confidence),

        # OCR
        "ocr": ocr_text or "",

        # GPT forklaring
        "gpt": gpt_notes or "",

        # status til debugging
        "status": "ok",
    }
