from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram
from starlette import status

from src.apis.business_service.schemas.business_service import (
    PredictionSchema,
    TextImageUrlSchema,
    UrlSchema,
)
from src.scripts.parse.donwload_image import load_image_from_base64
from src.scripts.parse.parse_url import extract_product_info
from src.scripts.service_model import predict_service_model

router_business = APIRouter(tags=["Predict"])

PREDICTION_BY_URL_LATENCY = Histogram(
    "prediction_by_utl_latency_seconds",
    "Latency of prediction by url in seconds",
)
PREDICTION_BY_TEXT_IMAGE_URL_LATENCY = Histogram(
    "prediction_by_text_image_url_latency_seconds",
    "Latency of prediction by text image url in seconds",
)
PREDICTION_BY_URL_REQUEST_COUNTER = Counter(
    "prediction_by_url_request_counter",
    "Total number of prediction by url requests"
)
PREDICTION_BY_TEXT_IMAGE_URL_REQUEST_COUNTER = Counter(
    "prediction_by_text_image_url_request_counter",
    "Total number of prediction by text image url requests"
)
REQUEST_COUNTER = Counter(
    "request_counter",
    "Total number all methods request"
)


@router_business.post("/predict-by-url", response_model=PredictionSchema)
async def predict_by_url(input_data: UrlSchema):
    REQUEST_COUNTER.inc()
    PREDICTION_BY_URL_REQUEST_COUNTER.inc()
    with PREDICTION_BY_URL_LATENCY.time():
        product_info = extract_product_info(input_data.url)
        if not product_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Problems with open url {input_data.url}"
            )
        if not product_info["title"] and not product_info["description"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Cant categorize. "
                       "It was not possible to find Title or Description"
            )
        text = product_info["title"] + ". " + product_info["description"]

        image_base64 = None
        if product_info["image_url"]:
            image_base64 = load_image_from_base64(b64_string=product_info["image_url"])

        result_service_model: dict = await predict_service_model(
            text=text,
            image_base64=image_base64
        )
        return PredictionSchema(category=result_service_model["category"])


@router_business.post("/predict-by-text-image", response_model=PredictionSchema)
async def predict_by_text_image(input_data: TextImageUrlSchema):
    REQUEST_COUNTER.inc()
    PREDICTION_BY_TEXT_IMAGE_URL_REQUEST_COUNTER.inc()
    with PREDICTION_BY_TEXT_IMAGE_URL_LATENCY.time():
        if not input_data.text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Empty text, text is required param"
            )
        image_base64 = None
        if input_data.image_url:
            image_base64 = load_image_from_base64(b64_string=input_data.image_url)

        result_service_model: dict = await predict_service_model(
            text=input_data.text,
            image_base64=image_base64
        )
        return PredictionSchema(category=result_service_model["category"])
