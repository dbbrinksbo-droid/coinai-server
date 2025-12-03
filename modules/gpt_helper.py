import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_prediction(label, confidence):
    """
    GPT hjælper med at forklare hvad modellen har fundet.
    Billigt – bruger GPT-4o-mini.
    """
    prompt = f"""
    Du er en møntekspert. 
    Min AI-model har identificeret en mønt som: '{label}' med {confidence}% sikkerhed.

    Forklar kort:
    - land
    - årstal (hvis relevant)
    - hvad der typisk gør denne mønt unik
    - 1 vigtig ekstra info til en samler
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du er ekspert i mønter."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message["content"]
