from typing import Optional

from pydantic import BaseModel


class TextImageSchema(BaseModel):
    text: str
    image_base64: Optional[str] = None


class PredictionSchema(BaseModel):
    category: Optional[str] = None
