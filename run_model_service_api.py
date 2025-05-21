import os
from contextlib import asynccontextmanager
from typing import Union

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette import status

from core.logger import logger

load_dotenv()

CATEGORIZATION_TEXT_MODEL = None


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global CATEGORIZATION_TEXT_MODEL
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_uri = os.getenv("MLFLOW_INFERENCE_MODEL_NAME")
    logger.info(f"Start loading model from mlflow: {model_uri}")
    CATEGORIZATION_TEXT_MODEL = mlflow.pyfunc.load_model(model_uri)
    yield
    # Clean up the ML models and release the resources
    del CATEGORIZATION_TEXT_MODEL


app = FastAPI(lifespan=lifespan)


class InputSchema(BaseModel):
    text: str


class ResultPredictionSchema(BaseModel):
    text: str
    category_id: int


@app.get("/health")
async def check_model_status():
    if CATEGORIZATION_TEXT_MODEL is not None:
        return Response(status_code=200)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="model is not loaded"
    )


@app.post(
    "/predict",
    response_model=Union[ResultPredictionSchema, list[ResultPredictionSchema]]
)
async def predict(input_data: Union[InputSchema, list[InputSchema]]):
    if isinstance(input_data, InputSchema):
        input_df = pd.DataFrame({"total_text": [input_data.text]})
        predictions = CATEGORIZATION_TEXT_MODEL.predict(input_df)
        result = ResultPredictionSchema(
            text=input_data.text,
            category_id=predictions[0]
        )
        return result
    if isinstance(input_data, list):
        texts = [data.text for data in input_data]
        input_df = pd.DataFrame({"total_text": texts})
        predictions = CATEGORIZATION_TEXT_MODEL.predict(input_df)
        result = []
        for idx, text in enumerate(texts):
            result.append(
                ResultPredictionSchema(
                    text=text,
                    category_id=predictions[idx]
                )
            )
        return result
    raise HTTPException(
        status_code=400,
        detail="Incorrect input params"
    )
