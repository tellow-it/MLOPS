
from pydantic import BaseModel


class SearchRequestSchema(BaseModel):
    url: str


class SearchResponseSchema(BaseModel):
    urls: list[str] | None = None
