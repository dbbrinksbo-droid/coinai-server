import os
from openai import OpenAI

def gpt_enhance(prediction, ocr_text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Missing OPENAI_API_KEY"

    client = OpenAI(api_key=api_key)

    prompt = f"""
    Analyze this coin:
    - Model prediction: {prediction}
    - OCR text: {ocr_text}

    Give a short explanation (max 5 lines).
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message["content"]
