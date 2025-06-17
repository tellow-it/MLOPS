from pydantic import BaseModel


class TextImageSchema(BaseModel):
    text: str
    image_base64: str | None = None


class PredictionSchema(BaseModel):
    category: str | None = None
