import requests

OCR_API_KEY = "helloworld"  # gratis nøgle, virker med begrænsninger
OCR_API_URL = "https://api.ocr.space/parse/image"


def extract_ocr(image_bytes: bytes, language="eng"):
    """
    Sender billede-data til OCR.Space API og returnerer tekst.
    image_bytes: bytes fra et billede (ex: uploaded_file.read())
    """

    print("Sending image to OCR API...")

    try:
        response = requests.post(
            OCR_API_URL,
            files={"filename": ("image.jpg", image_bytes)},
            data={"apikey": OCR_API_KEY, "language": language},
            timeout=30
        )

        result = response.json()

        # Debug print
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
