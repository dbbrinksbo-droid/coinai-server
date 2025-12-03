import os
from openai import OpenAI


def gpt_enhance(prediction, ocr_text: str) -> str:
    """
    Billig GPT-4o-mini tekstforbedring.
    Hvis OPENAI_API_KEY mangler, returneres en simpel tekst, men serveren crasher ikke.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY mangler i miljøvariabler – skipper GPT.")
        return "GPT not configured (missing OPENAI_API_KEY)."

    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            "Du er en mønt-ekspert.\n\n"
            f"Model prediction data:\n{prediction}\n\n"
            f"OCR (tekst på mønten):\n{ocr_text}\n\n"
            "Giv en kort, struktureret forklaring på dansk med:\n"
            "- Land (hvis muligt)\n"
            "- Valuta / mønttype\n"
            "- Omtrentligt årti eller periode\n"
            "- Eventuelle særlige kendetegn\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er en hjælpsom mønt-ekspert."},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT ERROR:", e)
        return f"GPT error: {e}"
