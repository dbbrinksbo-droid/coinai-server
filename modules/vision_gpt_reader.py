import base64
import os

from openai import OpenAI


def _b64_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def read_coin_from_image(image_bytes, side="front", user=None, prediction=None):
    """
    Vision GPT reader:
    - Reads visible text/symbols ONLY from the coin image.
    - Uses user hints (country/year/grade/etc) as guidance, not as truth.
    - Returns structured fields + a clean readable text.
    """
    if not image_bytes:
        return {"side": side, "text": "", "fields": {}, "confidence": 0, "notes": "No image bytes"}

    user = user or {}
    prediction = prediction or {}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Don't crash production if env is missing
        return {
            "side": side,
            "text": "",
            "fields": {},
            "confidence": 0,
            "notes": "OPENAI_API_KEY missing on server"
        }

    client = OpenAI()

    img_b64 = _b64_image(image_bytes)

    # Hard rules: NEVER say "no visible text" unless truly unreadable.
    system = (
        "You are a numismatics vision assistant. "
        "You MUST base your reading primarily on the provided coin photo. "
        "Extract inscriptions, denomination, year, country, ruler/coat-of-arms, mintmarks, and notable features. "
        "If something is unclear, say 'UNCLEAR' for that field, but do NOT claim 'no visible text' unless the photo truly shows none."
    )

    # User hints are guidance only (helps focus)
    hint = {
        "country_hint": user.get("country"),
        "year_hint": user.get("year"),
        "denomination_hint": user.get("denomination"),
        "grade_hint": user.get("grade"),
        "notes_hint": user.get("notes"),
        "model_label_hint": prediction.get("label"),
        "model_confidence_hint": prediction.get("confidence"),
    }

    prompt = (
        f"Analyze the {side} side of this coin photo.\n"
        f"Use these hints ONLY as guidance (not truth): {hint}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "side": "front|back",\n'
        '  "confidence": 0-100,\n'
        '  "fields": {\n'
        '    "country": "...|UNCLEAR",\n'
        '    "denomination": "...|UNCLEAR",\n'
        '    "year": "...|UNCLEAR",\n'
        '    "ruler_or_motif": "...|UNCLEAR",\n'
        '    "inscriptions": ["..."],\n'
        '    "mintmark": "...|UNCLEAR",\n'
        '    "language": "...|UNCLEAR",\n'
        '    "notable_features": ["..."],\n'
        '    "possible_errors": ["..."]\n'
        "  },\n"
        '  "text": "A short human-readable summary of what you can READ/SEE on this side."\n'
        "}\n"
    )

    # Use Responses API style (works with current OpenAI python SDK)
    # Model choice: keep it small/fast but vision-capable.
    model = os.getenv("SAGAMOENT_VISION_MODEL", "gpt-4o-mini")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ],
            },
        ],
        temperature=0.2,
    )

    # Extract text output
    out_text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type in ("output_text", "text"):
                    out_text += c.text

    # Best-effort JSON parse
    import json
    try:
        data = json.loads(out_text.strip())
    except Exception:
        # If model returns non-JSON, wrap it safely
        data = {
            "side": side,
            "confidence": 30,
            "fields": {"inscriptions": [], "notable_features": [], "possible_errors": []},
            "text": out_text.strip()[:2000]
        }

    # Normalize
    data["side"] = side
    data["notes"] = data.get("notes", "")
    if "confidence" not in data:
        data["confidence"] = 30
    if "fields" not in data or not isinstance(data["fields"], dict):
        data["fields"] = {"inscriptions": [], "notable_features": [], "possible_errors": []}
    if "text" not in data:
        data["text"] = ""

    return data
