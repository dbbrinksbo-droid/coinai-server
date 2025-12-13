import base64
import os
import json
from openai import OpenAI


def _b64_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def read_coin_from_image(image_bytes, side="front", user=None, prediction=None):
    if not image_bytes:
        return {
            "side": side,
            "confidence": 0,
            "fields": {},
            "text": ""
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "side": side,
            "confidence": 0,
            "fields": {},
            "text": ""
        }

    client = OpenAI()

    img_b64 = _b64_image(image_bytes)

    system = (
        "You are a numismatic vision expert.\n"
        "You MUST ONLY describe what is VISIBLE on the coin image.\n"
        "You MUST NOT invent, guess, or generalize.\n"
        "If something cannot be clearly seen, write 'UNCLEAR'.\n"
        "DO NOT write explanations, advice, or disclaimers.\n"
        "Return STRICT JSON ONLY."
    )

    prompt = (
        "Analyze this coin image.\n"
        "Return STRICT JSON in this exact format:\n\n"
        "{\n"
        '  "side": "front|back",\n'
        '  "confidence": 0-100,\n'
        '  "fields": {\n'
        '    "country": "TEXT|UNCLEAR",\n'
        '    "denomination": "TEXT|UNCLEAR",\n'
        '    "year": "TEXT|UNCLEAR",\n'
        '    "ruler_or_motif": "TEXT|UNCLEAR",\n'
        '    "inscriptions": ["TEXT"],\n'
        '    "mintmark": "TEXT|UNCLEAR",\n'
        '    "symbols": ["TEXT"],\n'
        '    "notable_features": ["TEXT"]\n'
        "  },\n"
        '  "text": "ONE short factual sentence describing ONLY what is visible."\n'
        "}\n"
    )

    model = os.getenv("SAGAMOENT_VISION_MODEL", "gpt-4o-mini")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                ]
            }
        ],
        temperature=0.0,
    )

    raw = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type in ("output_text", "text"):
                    raw += c.text

    try:
        data = json.loads(raw)
    except Exception:
        return {
            "side": side,
            "confidence": 10,
            "fields": {},
            "text": ""
        }

    data["side"] = side
    return data
