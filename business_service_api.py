from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.apis.business_service.routers.router_business import router_business

app = FastAPI(
    title="Business Service API",
    description="Business Service API for categorization by product by url or text + image_url",
    version="1.0.0"
)

app.include_router(router_business)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
)
instrumentator.instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")