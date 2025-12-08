import io
import requests
from PIL import Image

OCR_API_KEY = "helloworld"
OCR_URL = "https://api.ocr.space/parse/image"


def extract_ocr(pil_image: Image.Image):
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    try:
        res = requests.post(
            OCR_URL,
            files={"filename": ("image.jpg", img_bytes)},
            data={"apikey": OCR_API_KEY, "language": "eng"},
            timeout=30
        ).json()

        if res.get("IsErroredOnProcessing"):
            return {"success": False, "text": ""}

        parsed = res.get("ParsedResults", [])
        if not parsed:
            return {"success": False, "text": ""}

        text = parsed[0].get("ParsedText", "").strip()
        return {"success": True, "text": text}

    except Exception as e:
        print("OCR ERROR:", e)
        return {"success": False, "text": ""}

