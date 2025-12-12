# modules/vision_gpt_reader.py
# SagaMoent – Vision GPT Reader (REAL VISION)

import os
import base64
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a strict visual coin transcription engine.

Rules:
- Use ONLY what is visible in the image.
- Transcribe visible text EXACTLY.
- If something is not visible, write: NOT VISIBLE
- No guessing. No explanations.
- Output JSON only.
"""

def read_coin_from_image(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": SYSTEM_PROMPT},
                {"type": "input_image", "image_base64": b64}
            ]
        }]
    )

    text = response.output_text

    try:
        return eval(text)
    except Exception:
        return {
            "front_text": "NOT VISIBLE",
            "back_text": "NOT VISIBLE",
            "symbols": []
        }

