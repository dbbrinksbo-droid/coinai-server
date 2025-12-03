from modules.gpt_helper import ask_gpt

def read_text_from_image(image_base64: str) -> str:
    """
    OCR using GPT-4o-mini Vision (supports images via base64)
    """
    prompt = """
    Extract ALL text you can see in this coin image.
    Keep formatting simple. Only return text.
    """

    response = ask_gpt(prompt, image_base64=image_base64)
    return response
