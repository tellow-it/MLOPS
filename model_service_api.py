from contextlib import asynccontextmanager
from typing import Union, Optional

import mlflow

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette import status

from core.logger import logger
from core.config import Settings
from src.ml.model_registry import registry
from src.ml.loader import load_models, load_base_categories
from src.ml.predict import model_predict
from src.ml.upload_model import upload_model_to_mlflow
from src.scripts.pipeline_maas import get_emb_by_data
from src.scripts.s3.downloader import download_data_from_s3


@asynccontextmanager
async def lifespan(app_: FastAPI):
    logger.info("Start loading files from S3...")
    download_data_from_s3()
    logger.info("Success loading files from S3")
    logger.info("Start upload files to mlflow...")
    upload_model_to_mlflow()
    logger.info("Success upload files to mlflow")

    base_cat = load_base_categories()
    logger.info("Start load models...")
    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
    (
        cl_model_cat_l1,
        cl_model_cat_l2,
        cl_model_cat_l3,
        cat_by_model_l1,
        cat_by_model_l2,
        cat_by_model_l3
    ) = load_models()
    registry.cat_model_l1 = cl_model_cat_l1
    registry.cat_model_l2 = cl_model_cat_l2
    registry.cat_model_l3 = cl_model_cat_l3
    registry.categories_l1 = cat_by_model_l1
    registry.categories_l2 = cat_by_model_l2
    registry.categories_l3 = cat_by_model_l3
    registry.base_categories = base_cat
    yield


app = FastAPI(lifespan=lifespan)


class TextImageSchema(BaseModel):
    text: str
    image_base64: Optional[str] = None


class PredictionSchema(BaseModel):
    category: Optional[str] = None


@app.get("/health")
async def check_model_status():
    if registry.status_load_models():
        return Response(status_code=200)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="models is not loaded"
    )


@app.post(
    "/predict",
    response_model=Union[PredictionSchema, list[PredictionSchema]]
)
async def predict(input_params: Union[TextImageSchema, list[TextImageSchema]]):
    if not registry.status_load_models():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="models is not loaded"
        )
    if isinstance(input_params, TextImageSchema):
        x = get_emb_by_data(text=input_params.text, image_base64=input_params.image_base64)
        predicted_category = model_predict(x=x)
        return PredictionSchema(category=predicted_category)
    if isinstance(input_params, list):
        prediction_result = []
        for input_data in input_params:
            x = get_emb_by_data(text=input_data.text, image_base64=input_data.image_base64)
            predicted_category = model_predict(x=x)
            prediction_result.append(
                PredictionSchema(category=predicted_category)
            )
        return prediction_result
    raise HTTPException(
        status_code=400,
        detail="Incorrect input params"
    )
