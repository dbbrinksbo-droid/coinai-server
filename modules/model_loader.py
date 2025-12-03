from PIL import Image
import pytesseract
import io

def read_text_from_image(image_bytes):
    """
    Extract basic OCR text from an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        print("❌ OCR ERROR:", e)
        return ""
