# modules/vision_gpt_reader.py
# SagaMoent Vision GPT Reader — IMAGE ONLY, NO GUESSING

import os
import base64
from openai import OpenAI

# OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a visual coin transcription engine.

STRICT RULES:
- Use ONLY what is visible in the image.
- Transcribe visible text exactly as it appears.
- If text is unclear or not visible, write: NOT VISIBLE
- Do NOT guess.
- Do NOT explain.
- Output JSON ONLY in the following format:

{
  "front_text": "...",
  "back_text": "...",
  "symbols": []
}
"""


def read_coin_from_image(image_bytes: bytes) -> dict:
    """
    Reads visible text from a coin image using Vision GPT.
    """

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                    {
                        "type": "input_image",
                        "image_base64": image_b64
                    }
                ],
            }
        ],
        max_output_tokens=300,
    )

    # Extract raw text output
    output_text = response.output_text.strip()

    try:
        data = eval(output_text)
        return {
            "front_text": data.get("front_text", "NOT VISIBLE"),
            "back_text": data.get("back_text", "NOT VISIBLE"),
            "symbols": data.get("symbols", []),
        }
    except Exception:
        # Absolute safety fallback (no guessing)
        return {
            "front_text": "NOT VISIBLE",
            "back_text": "NOT VISIBLE",
            "symbols": [],
        }
