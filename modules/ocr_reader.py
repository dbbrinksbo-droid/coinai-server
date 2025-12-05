import io
import requests
from PIL import Image

OCR_API_KEY = "helloworld"  # testnøgle
OCR_API_URL = "https://api.ocr.space/parse/image"


def extract_ocr(pil_image: Image.Image, language="eng"):
    """
    Modtager PIL-image → konverterer til bytes → sender til OCR.Space.
    Returnerer dict: { success, text, error }
    """

    print("Sending image to OCR API...")

    # PIL → bytes
    img_bytes_io = io.BytesIO()
    pil_image.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()

    try:
        response = requests.post(
            OCR_API_URL,
            files={"filename": ("image.jpg", img_bytes)},
            data={"apikey": OCR_API_KEY, "language": language},
            timeout=30
        )

        result = response.json()
        print("OCR response:", result)

        if result.get("IsErroredOnProcessing"):
            return {
                "success": False,
                "text": "",
                "error": result.get("ErrorMessage")
            }

        parsed = result.get("ParsedResults")
        if not parsed:
            return {"success": False, "text": "", "error": "No OCR results"}

        extracted_text = parsed[0].get("ParsedText", "")

        return {
            "success": True,
            "text": extracted_text.strip()
        }

    except Exception as e:
        print("OCR error:", e)
        return {
            "success": False,
            "text": "",
            "error": str(e)
        }
