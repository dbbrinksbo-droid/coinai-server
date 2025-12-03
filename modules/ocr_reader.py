import io
from PIL import Image
import pytesseract


def extract_ocr(image_bytes: bytes) -> str:
    """
    Simpel OCR på billedet.
    Returnerer tekst (kan være tom string).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        print("❌ OCR ERROR:", e)
        return ""
