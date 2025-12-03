import time

def build_metadata(prediction, ocr_text=None, gpt_notes=None):
    """
    Samler ALLE informationer om mønten ét sted.
    Bruges af full_analyze.
    """

    return {
        "timestamp": int(time.time()),
        "model_prediction": prediction,         # fra ONNX
        "ocr_text": ocr_text,                  # årstal / tekst på mønt
        "gpt_enhancement": gpt_notes,          # ekstra analyser fra GPT
        "status": "ok"
    }
