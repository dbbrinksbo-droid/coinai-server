import pytesseract
from PIL import Image
import numpy as np

def read_text_from_image(image_bytes):
    """
    Læser tekst fra en mønt — fx årstal eller små detaljer.
    Bruges senere i full_analyze.
    """
    try:
        img = Image.open(image_bytes)
        img = img.convert("L")  # grayscale
        arr = np.array(img)

        text = pytesseract.image_to_string(arr, config="--psm 7")
        cleaned = text.strip()

        return cleaned if cleaned else None
    except Exception as e:
        return None
