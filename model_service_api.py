from contextlib import asynccontextmanager
import mlflow
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from core.logger import logger
from core.config import Settings
from src.ml.model_registry import registry
from src.ml.loader import load_models, load_base_categories
from src.ml.upload_model import upload_model_to_mlflow
from src.scripts.s3.downloader import download_data_from_s3
from src.apis.model_service.routers.router_service import router_service


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


app = FastAPI(
    title="Model Service API",
    description="Model Service API for categorization by text and image",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router_service)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
)
instrumentator.instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
