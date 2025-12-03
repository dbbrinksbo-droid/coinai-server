import pytesseract
from PIL import Image
import io

def extract_ocr(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        print("❌ OCR ERROR:", e)
        return ""
