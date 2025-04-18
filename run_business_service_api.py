import pandas as pd
from dotenv import load_dotenv
from fastapi import HTTPException, Response
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class UrlSchema(BaseModel):
    url: str


class UrlTextSchema(UrlSchema):
    text: str


class ResultPredictionSchema(BaseModel):
    url: str
    category_id: int


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post(
    "/predict-by-url",
    response_model=ResultPredictionSchema
)
async def predict_by_url(input_data: UrlSchema):
    return ResultPredictionSchema(url=input_data.url, category_id=0)


@app.post(
    "/predict-by-text-image",
    response_model=ResultPredictionSchema
)
async def predict_by_text_image(input_data: UrlTextSchema):
    return ResultPredictionSchema(url=input_data.url, category_id=0)
