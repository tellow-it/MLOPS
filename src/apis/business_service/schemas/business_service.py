
from pydantic import BaseModel


class UrlSchema(BaseModel):
    url: str


class TextImageUrlSchema(BaseModel):
    text: str
    image_url: str | None = None


class PredictionSchema(BaseModel):
    category: str | None = None
