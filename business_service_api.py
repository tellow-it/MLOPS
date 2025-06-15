from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import Settings
from src.apis.business_service.routers.router_business import router_business

app = FastAPI(
    title=Settings.BUSINESS_SERVICE_API_TITLE,
    description=Settings.BUSINESS_SERVICE_API_DESCRIPTION,
    version=Settings.BUSINESS_SERVICE_API_VERSION
)

app.include_router(router_business)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
)
instrumentator.instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
