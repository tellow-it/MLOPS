from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette import status

from ml_product_categorizer.scripts.preprocessing import process_text
from src.scripts.image_processing.get_image import download_image
from src.scripts.request_to.parse_url import fetch_page_text
from src.scripts.request_to.service_model import predict_service_model

load_dotenv()

app = FastAPI()


class UrlSchema(BaseModel):
    url: str


class UrlTextSchema(BaseModel):
    text: str
    picture_url: str


class ResultPredictionSchema(BaseModel):
    category_id: int


@app.post(
    "/predict-by-url",
    response_model=ResultPredictionSchema
)
async def predict_by_url(input_data: UrlSchema):
    text = await fetch_page_text(input_data.url)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cant parse url"
        )
    processed_text = process_text(
        url=input_data.url,
        product_text=text,
        picture_url=None
    )
    if not processed_text:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Empty text after processing"
        )
    result_service_model: dict = await predict_service_model(text=processed_text)
    predicted_category_id = int(result_service_model["category_id"])
    return ResultPredictionSchema(category_id=predicted_category_id)


@app.post(
    "/predict-by-text-image",
    response_model=ResultPredictionSchema
)
async def predict_by_text_image(input_data: UrlTextSchema):
    processed_text = process_text(
        url=None,
        product_text=input_data.text,
        picture_url=input_data.picture_url
    )
    if not processed_text:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Empty text after processing"
        )
    image_array = await download_image(url=input_data.picture_url)
    if not image_array:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Problems with getting image by product_url"
        )
    result_service_model: dict = await predict_service_model(text=processed_text)
    predicted_category_id = int(result_service_model["category_id"])
    return ResultPredictionSchema(category_id=predicted_category_id)
