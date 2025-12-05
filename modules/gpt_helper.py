import os
from openai import OpenAI

def gpt_enhance(prediction_text: str, ocr_text: str) -> str:
    """
    GPT-4o-mini tekstforbedring:
    - Modtager prediction label + OCR-tekst
    - Returnerer kort forklaring
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY mangler – bruger fallback tekst.")
        return "Ingen GPT-forklaring (API-nøgle mangler)."

    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            "Du er møntekspert.\n\n"
            f"Modelens forudsigelse: {prediction_text}\n\n"
            f"OCR fundet tekst: {ocr_text}\n\n"
            "Giv en kort forklaring på dansk om:\n"
            "- Land\n"
            "- Mønttype\n"
            -" Periode / årstal\n"
            "- Eventuelle kendetegn"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er møntekspert."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT ERROR:", e)
        return f"GPT-fejl: {e}"
