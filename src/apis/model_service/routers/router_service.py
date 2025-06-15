from typing import Union

from fastapi import APIRouter, HTTPException, Response
from prometheus_client import Counter, Histogram
from starlette import status

from src.apis.model_service.schemas.model_service import (
    PredictionSchema,
    TextImageSchema,
)
from src.ml.model_registry import registry
from src.ml.predict import model_predict
from src.scripts.pipeline_maas import get_emb_by_data

router_service = APIRouter(tags=["Predict"])

SINGLE_PREDICTION_LATENCY = Histogram(
    "single_prediction_latency_seconds",
    "Latency of single prediction in seconds",
)
BATCH_PREDICTION_LATENCY = Histogram(
    "batch_prediction_latency_seconds",
    "Latency of batch prediction in seconds",
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size_distribution",
    "Distribution of batch sizes in batch prediction requests",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)
SINGLE_REQUEST_COUNTER = Counter(
    "single_request_counter",
    "Total number of single prediction requests"
)
BATCH_REQUEST_COUNTER = Counter(
    "batch_request_counter",
    "Total number of batch prediction requests"
)
REQUEST_COUNTER = Counter(
    "request_counter",
    "Total number of prediction for 'predict' request"
)


@router_service.get("/health")
async def check_model_status():
    if registry.status_load_models():
        return Response(status_code=200)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="models is not loaded"
    )


@router_service.post(
    "/predict",
    response_model=Union[PredictionSchema, list[PredictionSchema]]
)
async def predict(input_params: Union[TextImageSchema, list[TextImageSchema]]):
    REQUEST_COUNTER.inc()
    if not registry.status_load_models():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="models is not loaded"
        )
    if isinstance(input_params, TextImageSchema):
        SINGLE_REQUEST_COUNTER.inc()
        with SINGLE_PREDICTION_LATENCY.time():
            x = get_emb_by_data(
                text=input_params.text,
                image_base64=input_params.image_base64
            )
            predicted_category = model_predict(x=x)
            return PredictionSchema(category=predicted_category)
    if isinstance(input_params, list):
        BATCH_SIZE_HISTOGRAM.observe(len(input_params))
        BATCH_REQUEST_COUNTER.inc()
        with BATCH_PREDICTION_LATENCY.time():
            prediction_result = []
            for input_data in input_params:
                x = get_emb_by_data(
                    text=input_data.text,
                    image_base64=input_data.image_base64
                )
                predicted_category = model_predict(x=x)
                prediction_result.append(
                    PredictionSchema(category=predicted_category)
                )
        return prediction_result
    raise HTTPException(
        status_code=400,
        detail="Incorrect input params"
    )
