from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64
from PIL import Image

app = FastAPI()

class CoinRequest(BaseModel):
    image_base64: str

@app.post("/analyze")
def analyze_coin(req: CoinRequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(img_bytes))

        result = {
            "country": "Ukendt",
            "type": "Ukendt",
            "year": "Ukendt",
            "variant": "Ukendt",
            "metal": "Ukendt",
            "grade": "Ukendt"
        }

        return {"success": True, "data": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

