from typing import Optional

from pydantic import BaseModel


class UrlSchema(BaseModel):
    url: str


class TextImageUrlSchema(BaseModel):
    text: str
    image_url: Optional[str] = None


class PredictionSchema(BaseModel):
    category: Optional[str] = None
